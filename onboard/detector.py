import cv2
import logging

from onboard.saliency import SaliencyDetector
from onboard.trackers import make_tracker

logger = logging.getLogger(__name__)


# The init box is sized from the target, not fixed.
#
# This used to be a flat 72 px and that was the single worst thing in the
# tracking path.  Measured across 5 terrains x 2 motion profiles: with a 72 px
# box, EVERY tracker - CSRT, KCF, MOSSE, Nano, ViT, DaSiamRPN - loses a 10 px
# target within ~20 frames, and they all drift at exactly the aircraft's ground
# speed in pixels/frame.  That number is the diagnosis: they are tracking the
# terrain inside the box, not the target.  An 8 px target is 0.57% of a 72x72
# patch, so the correlation peak belongs to the ground.
#
# Scaling the box to the target instead takes CSRT from 1% to 75% lock survival
# at 8 px, with no change of tracker.  That is a bigger win than any model swap
# and it helps whichever backend is running.
#
# The benchmark's best box was ~2.5x the target's true footprint.  The scale
# below is 1.25, and the difference is not a disagreement - it is that we do
# not know the true footprint.  We have the saliency hint, which is already an
# inflated estimate: measured against known target sizes, the DoG response
# cannot resolve below its own filter scale and bottoms out around 20 px.
#
#   true target   6    8   10   14   20   28   40   60
#   hint         26   24   20   19   26   35   47   65
#   ratio      4.3x 3.0x 2.0x 1.4x 1.4x 1.3x 1.2x 1.1x
#
# So 2.5x the *hint* is 5x the small targets we actually fly against, which is
# most of the old problem back again.  1.25x the hint lands the box at roughly
# 1.4-2.5x the true target across the whole range.  Do not "restore" this to
# 2.5 from reading the benchmark without also replacing the size estimator.
INIT_BOX_SCALE = 1.25    # box edge = this x the proposal's *hint*, not the target
INIT_BOX_MIN = 24        # the hint floor is ~20 px; below this nothing locks
INIT_BOX_MAX = 96        # a close, large target should not swallow the frame

# Used only when the proposer finds nothing at the click, so we have no size
# estimate at all.  A guess, and deliberately far below the old 72.
INIT_PATCH_FALLBACK = 40


class YoloProposer:
    """
    Optional YOLO proposal backend.

    Kept so the saliency swap can be A/B'd or reverted, but no longer a hard
    dependency - `ultralytics` (and the ~2 GB of PyTorch behind it) is imported
    lazily and only when detection.backend is explicitly set to "yolo".
    """

    def __init__(self, cfg):
        from ultralytics import YOLO      # lazy: not required for the default path
        self.cfg = cfg
        self.model = YOLO(cfg.get('model', 'yolov8n.pt'))
        logger.info("Proposal backend: YOLO")

    def find_nearest(self, frame, ref_x, ref_y, max_dist=None):
        """Nearest YOLO detection to (ref_x, ref_y), optional distance gate."""
        try:
            results = self.model(
                frame,
                conf=self.cfg.get('confidence', 0.4),
                iou=self.cfg.get('iou', 0.45),
                device=self.cfg.get('device', 'cpu'),
                verbose=False
            )
            best, best_dist = None, float('inf')
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    d = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best = (x1, y1, x2 - x1, y2 - y1)
            if best is not None and max_dist is not None and best_dist > max_dist:
                return None
            return best
        except Exception as e:
            logger.error(f"YOLO error: {e}")
            return None


def make_proposer(cfg):
    """Build the object-proposal backend named by detection.backend."""
    backend = str(cfg.get('backend', 'saliency')).lower()
    if backend == 'yolo':
        try:
            return YoloProposer(cfg)
        except Exception as e:
            logger.error(f"YOLO backend unavailable ({e}) - falling back to saliency")
    return SaliencyDetector(cfg)


class ObjectDetector:
    """
    Click-to-track detector.

    Workflow
    ────────
    1. User clicks a pixel on GCS.
    2. The proposal backend runs once to find a distinct object at that spot.
       - If found → use its centre AND its size, so the box is scaled to the
         target rather than to a constant.  See INIT_BOX_SCALE: this is the
         single biggest lever in the whole tracking path.
       - If not   → INIT_PATCH_FALLBACK, centred on the click.
    3. The primary tracker follows that visual patch frame-by-frame.  No further
       proposal calls during normal tracking, so nothing can cause a target
       switch.
    4. If the primary fails for max_fails×2 consecutive frames → hand the last
       known box to the pixel tracker and carry on with that (once per target).
    5. If the pixel tracker then fails for the same number of frames → try one
       re-acquisition within 120 px of the last known position.  Otherwise
       enter SEARCHING.

    Step 4 is why the primary is a tracker that can report its own uncertainty:
    a handover can only be triggered by a tracker able to say "I lost it".
    """

    STATE_SEARCHING = "searching"
    STATE_TRACKING = "tracking"   # a tracker is locked on

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = self.STATE_SEARCHING
        self.last_bbox = None   # (x, y, w, h) — top-left corner + size
        self.fail_count = 0
        self.max_fails = cfg.get('max_detection_failures', 10)
        # Consecutive tracker failures tolerated before we hand over or give up.
        # Frames 1..coast_limit-1 are "coasting": the box returned is the last
        # KNOWN position, not a fresh measurement.
        self.coast_limit = self.max_fails * 2

        # Primary tracker, and the pixel tracker behind it.  `fallback` is None
        # when the primary already is the pixel tracker, i.e. nothing further
        # to degrade to.
        self._primary, self._fallback = make_tracker(cfg)
        self._tracker = None      # whichever of the two is currently live
        self.degraded = False     # True once we have handed over to pixels
        self.last_score = None    # primary's own 0..1 match score, if it has one

        # Score band used by confidence().  Below `score_lost` the tracker's own
        # update() already reports failure; between the two we derate rather
        # than pretending the position is as good as a clean lock.
        #
        # Measured on a synthetic clip: a correct ViT lock scores ~0.47-0.55,
        # and with the target absent the score collapses to ~0.07.  So the band
        # must close well below a healthy lock's score - set it at 0.60 and a
        # perfectly tracked target flies at 0.7-0.85 authority forever, which
        # is a regression dressed up as caution.  0.35 derates only a genuinely
        # marginal lock.  Worth re-checking against real footage; the numbers
        # above are from procedural texture, not a camera.
        self.score_lost = float(cfg.get('vit_score_threshold', 0.20))
        self.score_good = float(cfg.get('score_confident', 0.35))

        # Ceiling on confidence once we have fallen back to pixel tracking.
        # Without it the handover would hand full authority straight back: the
        # primary says "lost", we reset the fail count, and CSRT - which cannot
        # tell us it is wrong - reports a clean lock on whatever it latched
        # onto.  A best-effort continuation is not a fresh lock, and the
        # aircraft should not commit to it as if it were.
        self.degraded_confidence = float(cfg.get('degraded_confidence', 0.60))

        # Init box geometry — see INIT_BOX_SCALE for why this is not a constant.
        self.box_scale = float(cfg.get('init_box_scale', INIT_BOX_SCALE))
        self.box_min = int(cfg.get('init_box_min', INIT_BOX_MIN))
        self.box_max = int(cfg.get('init_box_max', INIT_BOX_MAX))
        self.box_fallback = int(cfg.get('init_patch_size', INIT_PATCH_FALLBACK))

        self.proposer = make_proposer(cfg)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def confidence(self):
        """
        How much to trust the position currently being reported, 0.0 … 1.0.

        Two separate things can lower it:

        Coasting - the tracker reported failure, and the box being returned is
        the last KNOWN position, not a fresh measurement.  Decays linearly with
        the number of consecutive failures.  This is the only mechanism
        available behind the pixel tracker, which reports no score.

        A weak lock - the tracker measured the target this frame but is not
        sure it is the right one.  Only the deep tracker can tell us this, and
        it is strictly better information than frame counting: it derates
        *before* the lock is lost rather than after.

        Callers scale control commands by this, so the aircraft winds down
        smoothly instead of either steering at full authority on a stale
        position or jerking to a stop on a single-frame hiccup.
        """
        if self.state != self.STATE_TRACKING or self.last_bbox is None:
            return 0.0

        if self.fail_count > 0:
            conf = max(0.0, 1.0 - (self.fail_count / float(self.coast_limit)))
        elif self.last_score is None:
            conf = 1.0   # no score available (pixel tracker) — a lock is a lock
        else:
            span = self.score_good - self.score_lost
            conf = 1.0 if span <= 0 else \
                max(0.0, min(1.0, (self.last_score - self.score_lost) / span))

        if self.degraded:
            conf = min(conf, self.degraded_confidence)
        return conf

    def select_target(self, click_x, click_y, frame):
        """Called when the user clicks on GCS.  Initialises the tracker now."""
        self.state = self.STATE_SEARCHING
        self.last_bbox = None
        self.fail_count = 0
        self._tracker = None
        self.degraded = False
        self.last_score = None

        # Snap to a distinct object near the click, but keep the patch a small
        # fixed size so the tracker learns the specific region pointed at.
        centre_x, centre_y = click_x, click_y
        hint = self.proposer.find_nearest(frame, click_x, click_y, max_dist=80)
        if hint is not None:
            hx, hy, hw, hh = hint
            centre_x = hx + hw // 2
            centre_y = hy + hh // 2
            logger.info(f"Proposal hint: centre ({centre_x},{centre_y}) size {hw}x{hh}")

        bbox = self._make_patch(frame, centre_x, centre_y, self._box_size(hint))

        if self._start(frame, bbox, self._primary):
            self.last_bbox = bbox
            self.state = self.STATE_TRACKING
            logger.info(f"{self._tracker.name} initialised on patch {bbox}")
            return True

        # Primary refused the patch outright.  Try pixels before giving up on
        # the operator's click — a failed selection is worse than a weak one.
        if self._start(frame, bbox, self._fallback):
            self.degraded = True
            self.last_bbox = bbox
            self.state = self.STATE_TRACKING
            logger.warning(f"Primary init failed; tracking on {self._tracker.name}")
            return True

        logger.error("Could not initialise any tracker")
        return False

    def process(self, frame):
        """
        Returns (target, state, annotated_frame).
        target = (cx, cy, w, h) in pixels, or None if not tracking.
        """
        annotated = frame.copy()

        if self.state == self.STATE_SEARCHING or self._tracker is None:
            return None, self.state, annotated

        ok, rect, score = self._tracker.update(frame)
        self.last_score = score
        if ok:
            x, y, w, h = [int(v) for v in rect]
            # Sanity-check: reject obviously degenerate boxes
            if w > 5 and h > 5:
                self.last_bbox = (x, y, w, h)
                self.fail_count = 0
            else:
                ok = False

        if not ok:
            self.fail_count += 1
            if self.fail_count >= self.coast_limit:
                # The advanced tracker has given up.  Before abandoning the
                # target, hand the last known box to pixel tracking and let it
                # carry on from there.
                if not self._handover(frame):
                    # Try re-acquisition near last known position
                    ref_cx, ref_cy = self._bbox_center(self.last_bbox)
                    new_bbox = self.proposer.find_nearest(frame, ref_cx, ref_cy, max_dist=120)
                    if new_bbox is not None:
                        nx, ny, nw, nh = new_bbox
                        centre_x = nx + nw // 2
                        centre_y = ny + nh // 2
                        patch = self._make_patch(frame, centre_x, centre_y,
                                                 self._box_size(new_bbox))
                        self._start(frame, patch, self._tracker)
                        self.last_bbox = patch
                        self.fail_count = 0
                        logger.info(f"Re-acquired; {self._tracker.name} reinitialised")
                    else:
                        logger.warning("Tracker lost target — entering SEARCHING")
                        self.state = self.STATE_SEARCHING
                        return None, self.state, annotated

        # Annotate
        if self.last_bbox:
            x, y, w, h = self.last_bbox
            cx = x + w // 2
            cy = y + h // 2
            color = (0, 200, 0) if ok else (0, 100, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.circle(annotated, (cx, cy), 5, color, -1)
            label = "TRACK" if ok else f"COAST({self.fail_count})"
            if ok and self.last_score is not None:
                label += f" {self.last_score:.2f}"
            if self.degraded:
                label += " [px]"
            cv2.putText(annotated, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            return (cx, cy, w, h), self.state, annotated

        return None, self.state, annotated

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _box_size(self, hint):
        """
        Edge length for the init box, scaled to the proposal's target footprint.

        `hint` is the proposer's (x, y, w, h) or None.  With no hint we have no
        size estimate at all and have to guess; with one, the box tracks the
        target's own size, which is what keeps the terrain out of the patch.
        """
        if hint is None:
            return self.box_fallback
        edge = int(round(self.box_scale * max(hint[2], hint[3])))
        return max(self.box_min, min(edge, self.box_max))

    @staticmethod
    def _make_patch(frame, cx, cy, size):
        """Return a clamped (x, y, w, h) patch centred on (cx, cy)."""
        h, w = frame.shape[:2]
        size = max(1, min(int(size), w, h))   # a box bigger than the frame is not a box
        half = size // 2
        x = max(0, min(cx - half, w - size))
        y = max(0, min(cy - half, h - size))
        return (x, y, size, size)

    def _start(self, frame, bbox, tracker):
        """Init `tracker` on `bbox` and make it the live one.  True on success."""
        if tracker is None:
            return False
        try:
            if tracker.init(frame, bbox):
                self._tracker = tracker
                return True
        except Exception as e:
            logger.error(f"{getattr(tracker, 'name', 'tracker')} init failed: {e}")
        return False

    def _handover(self, frame):
        """
        The primary tracker has lost lock — continue on pixel tracking.

        This is the fallback the whole backend split exists for.  It fires on
        the primary's own admission of failure, hands over the last position it
        was confident about, and is one-way: once degraded we stay on pixels
        until the next select_target, rather than flapping between backends on
        a target the deep tracker has already shown it cannot hold.

        False means there is nothing to hand over to (already degraded, or the
        pixel tracker *is* the primary), so the caller should fall through to
        re-acquisition.
        """
        if self.degraded or self._fallback is None or self.last_bbox is None:
            return False
        if not self._start(frame, self.last_bbox, self._fallback):
            return False
        self.degraded = True
        self.fail_count = 0
        self.last_score = None     # pixels report none; confidence goes back to coasting
        logger.warning(f"Primary tracker lost lock — handed over to {self._tracker.name}")
        return True

    @staticmethod
    def _bbox_center(bbox):
        if bbox is None:
            return 0, 0
        x, y, w, h = bbox
        return x + w // 2, y + h // 2
