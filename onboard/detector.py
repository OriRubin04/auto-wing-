import cv2
import logging

from onboard.saliency import SaliencyDetector

logger = logging.getLogger(__name__)


# Size of the patch CSRT is initialised on, in pixels.
# Smaller = more specific (less likely to drift); bigger = more texture = more stable.
INIT_PATCH_SIZE = 72


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
       - If found → use its centre, but cap the box to INIT_PATCH_SIZE so CSRT
         gets a tight, specific patch.
       - If not   → use a fixed-size patch centred on the click.
    3. CSRT tracks that visual patch frame-by-frame.  No further proposal calls
       during normal tracking, so nothing can cause a target switch.
    4. If CSRT fails for max_fails×2 consecutive frames → try one re-acquisition
       within 120 px of the last known position, then reinit CSRT.  Otherwise
       enter SEARCHING.
    """

    STATE_SEARCHING = "searching"
    STATE_TRACKING = "tracking"   # CSRT active

    TEMPLATE_DEFAULT = 32          # template side when there is no size hint
    TEMPLATE_MIN = 24
    TEMPLATE_SCALE = 1.5           # template side = this x largest hint side

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = self.STATE_SEARCHING
        self.last_bbox = None   # (x, y, w, h) — top-left corner + size
        self.fail_count = 0
        self.max_fails = cfg.get('max_detection_failures', 10)
        # Consecutive CSRT failures tolerated before we give up and re-acquire.
        # Frames 1..coast_limit-1 are "coasting": the box returned is the last
        # KNOWN position, not a fresh measurement.
        self.coast_limit = self.max_fails * 2
        # A CSRT "success" is not a measurement until it passes these.  CSRT
        # almost never reports failure on its own: when the target vanishes it
        # re-locks on whatever correlates best, which can be a frame corner
        # 400 px away, and reports that at full confidence forever.
        # Max centre movement between consecutive frames, in pixels.  A 60 deg
        # lens slewing at 100 deg/s moves the image ~43 px/frame at 25 fps,
        # so 100 px only rejects jumps no physical target can make.
        self.max_jump_px = float(cfg.get('max_jump_px', 100))
        # Minimum fraction of the box that must lie inside the frame.
        self.min_visible_frac = float(cfg.get('min_visible_frac', 0.5))
        # Re-acquisition gate.  A saliency proposal only becomes the new
        # target if a grayscale patch there matches the template captured at
        # selection, by normalised cross-correlation, at least this well.
        # Measured on a synthetic vehicle: same target 0.85, rotated 45 deg
        # 0.61, rotated 90 deg 0.47; a wrong blob on clutter ~0.0.  Without
        # this, "nearest salient blob" always finds SOMETHING - a rock, a
        # bush - and CSRT then tracks that at full confidence, so no lost-lock
        # logic downstream can ever fire.
        self.reacquire_min_ncc = float(cfg.get('reacquire_min_ncc', 0.3))
        # NCC falls off fast with misalignment (0.15 at 15 px), so each
        # candidate is searched over +/- this many pixels for the peak.
        self.reacquire_search_px = int(cfg.get('reacquire_search_px', 40))
        # The template is TARGET-sized, not CSRT-patch-sized.  A 72 px patch
        # around a 12 px target is 97% terrain, and terrain matches terrain,
        # so a patch-sized template happily re-acquires the ground the target
        # used to be on.  Sized from the saliency hint at selection.
        self._template = None
        self._template_size = self.TEMPLATE_DEFAULT
        self._cv_tracker = None
        self._tracker_type = cfg.get('fallback_tracker', 'CSRT')
        self.proposer = make_proposer(cfg)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def confidence(self):
        """
        How much to trust the position currently being reported, 0.0 … 1.0.

        1.0  = fresh CSRT measurement this frame.
        <1.0 = coasting on the last known box; decays linearly with the number
               of consecutive failures.
        0.0  = the box is as old as we are willing to act on at all.

        Callers scale control commands by this so the aircraft stops
        committing to a stale position gradually rather than either steering
        blindly for the whole coast window or jerking to a stop on every
        single-frame tracker hiccup.
        """
        if self.state != self.STATE_TRACKING or self.last_bbox is None:
            return 0.0
        if self.fail_count <= 0:
            return 1.0
        return max(0.0, 1.0 - (self.fail_count / float(self.coast_limit)))

    def select_target(self, click_x, click_y, frame):
        """Called when the user clicks on GCS.  Initialises CSRT immediately."""
        self.state = self.STATE_SEARCHING
        self.last_bbox = None
        self.fail_count = 0
        self._cv_tracker = None

        # Snap to a distinct object near the click, but keep the CSRT patch a
        # small fixed size so the tracker learns the specific region pointed at.
        centre_x, centre_y = click_x, click_y
        self._template_size = self.TEMPLATE_DEFAULT
        hint = self.proposer.find_nearest(frame, click_x, click_y, max_dist=80)
        if hint is not None:
            hx, hy, hw, hh = hint
            centre_x = hx + hw // 2
            centre_y = hy + hh // 2
            self._template_size = int(max(self.TEMPLATE_MIN,
                                          min(INIT_PATCH_SIZE,
                                              self.TEMPLATE_SCALE * max(hw, hh))))
            logger.info(f"Proposal hint: centre ({centre_x},{centre_y}) size {hw}x{hh} "
                        f"-> template {self._template_size} px")

        bbox = self._make_patch(frame, centre_x, centre_y, INIT_PATCH_SIZE)
        self._init_cv_tracker(frame, bbox)
        if self._cv_tracker is not None:
            self.last_bbox = bbox
            self._template = self._capture_template(frame, bbox)
            self.state = self.STATE_TRACKING
            logger.info(f"CSRT initialised on patch {bbox}")
            return True

        logger.error("Could not initialise any tracker")
        return False

    def process(self, frame):
        """
        Returns (target, state, annotated_frame).
        target = (cx, cy, w, h) in pixels, or None if not tracking.
        """
        annotated = frame.copy()

        if self.state == self.STATE_SEARCHING or self._cv_tracker is None:
            return None, self.state, annotated

        ok, rect = self._cv_tracker.update(frame)
        if ok:
            bbox = tuple(int(v) for v in rect)
            reason = self._reject_reason(frame, bbox)
            if reason is None:
                self.last_bbox = bbox
                self.fail_count = 0
            else:
                logger.debug(f"Tracker box rejected: {reason}")
                ok = False

        if not ok:
            self.fail_count += 1
            if self.fail_count >= self.coast_limit:
                found = self._reacquire(frame)
                if found is not None:
                    centre_x, centre_y, score = found
                    patch = self._make_patch(frame, centre_x, centre_y, INIT_PATCH_SIZE)
                    self._init_cv_tracker(frame, patch)
                    self.last_bbox = patch
                    self._template = self._capture_template(frame, patch)
                    self.fail_count = 0
                    logger.info(f"Re-acquired (ncc {score:.2f}); CSRT reinitialised")
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
            cv2.putText(annotated, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            return (cx, cy, w, h), self.state, annotated

        return None, self.state, annotated

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _reacquire(self, frame):
        """
        Find the target again near its last known position.

        Saliency proposes where something stands out; the template decides
        whether it is OUR target.  The last known position is always a
        candidate too, so a target that is present but not salient (low
        contrast, partly occluded) can still be recovered.

        Returns (cx, cy, ncc) for the best candidate above the gate, else None.
        """
        ref_cx, ref_cy = self._bbox_center(self.last_bbox)
        cands = [(ref_cx, ref_cy)]
        hint = self.proposer.find_nearest(frame, ref_cx, ref_cy, max_dist=120)
        if hint is not None:
            hx, hy, hw, hh = hint
            cands.insert(0, (hx + hw // 2, hy + hh // 2))

        if self._template is None:
            # No template to verify against: fall back to trusting saliency.
            return (cands[0][0], cands[0][1], 1.0) if hint is not None else None

        best = None
        for cx, cy in cands:
            m = self._template_match(frame, cx, cy, self.reacquire_search_px)
            if m is not None and (best is None or m[0] > best[0]):
                best = m
        if best is None or best[0] < self.reacquire_min_ncc:
            logger.info(f"Re-acquisition rejected: best ncc "
                        f"{best[0] if best else float('nan'):.2f} "
                        f"< {self.reacquire_min_ncc:.2f}")
            return None
        return best[1], best[2], best[0]

    def _template_match(self, frame, cx, cy, search):
        """
        Peak NCC of the template within +/- search px of (cx, cy).
        Returns (ncc, peak_cx, peak_cy) or None if the window is too small.
        """
        th, tw = self._template.shape[:2]
        fh, fw = frame.shape[:2]
        x0 = max(0, int(cx) - tw // 2 - search)
        y0 = max(0, int(cy) - th // 2 - search)
        x1 = min(fw, int(cx) + tw // 2 + search)
        y1 = min(fh, int(cy) + th // 2 + search)
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        win = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(win, self._template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        return float(score), x0 + loc[0] + tw // 2, y0 + loc[1] + th // 2

    def _capture_template(self, frame, bbox):
        """Grayscale template of _template_size px centred on bbox."""
        cx, cy = self._bbox_center(bbox)
        x, y, w, h = self._make_patch(frame, cx, cy, self._template_size)
        return cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    def _reject_reason(self, frame, bbox):
        """
        Why a tracker box should NOT count as a measurement, or None if it is
        acceptable.  All checks are geometric: no thresholds on appearance.
        """
        x, y, w, h = bbox
        if w <= 5 or h <= 5:
            return f"degenerate {w}x{h}"

        fh, fw = frame.shape[:2]
        vis_w = min(x + w, fw) - max(x, 0)
        vis_h = min(y + h, fh) - max(y, 0)
        visible = max(0, vis_w) * max(0, vis_h) / float(w * h)
        if visible < self.min_visible_frac:
            return f"only {visible:.0%} inside frame"

        if self.last_bbox is not None:
            px, py = self._bbox_center(self.last_bbox)
            cx, cy = self._bbox_center(bbox)
            jump = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if jump > self.max_jump_px:
                return f"jumped {jump:.0f} px in one frame"
        return None

    @staticmethod
    def _make_patch(frame, cx, cy, size):
        """Return a clamped (x, y, w, h) patch centred on (cx, cy)."""
        h, w = frame.shape[:2]
        half = size // 2
        x = max(0, min(cx - half, w - size))
        y = max(0, min(cy - half, h - size))
        return (x, y, size, size)

    def _init_cv_tracker(self, frame, bbox):
        factories = []
        for ns in [cv2, getattr(cv2, 'legacy', None)]:
            if ns is None:
                continue
            for name in [f'Tracker{self._tracker_type}_create',
                         'TrackerCSRT_create', 'TrackerKCF_create', 'TrackerMOSSE_create']:
                fn = getattr(ns, name, None)
                if fn and fn not in factories:
                    factories.append(fn)
        for factory in factories:
            try:
                t = factory()
                t.init(frame, bbox)
                self._cv_tracker = t
                logger.info(f"Tracker: {factory.__name__}")
                return
            except Exception:
                continue
        logger.error("No working OpenCV tracker found")
        self._cv_tracker = None

    @staticmethod
    def _bbox_center(bbox):
        if bbox is None:
            return 0, 0
        x, y, w, h = bbox
        return x + w // 2, y + h // 2
