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

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = self.STATE_SEARCHING
        self.last_bbox = None   # (x, y, w, h) — top-left corner + size
        self.fail_count = 0
        self.max_fails = cfg.get('max_detection_failures', 10)
        self._cv_tracker = None
        self._tracker_type = cfg.get('fallback_tracker', 'CSRT')
        self.proposer = make_proposer(cfg)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def select_target(self, click_x, click_y, frame):
        """Called when the user clicks on GCS.  Initialises CSRT immediately."""
        self.state = self.STATE_SEARCHING
        self.last_bbox = None
        self.fail_count = 0
        self._cv_tracker = None

        # Snap to a distinct object near the click, but keep the CSRT patch a
        # small fixed size so the tracker learns the specific region pointed at.
        centre_x, centre_y = click_x, click_y
        hint = self.proposer.find_nearest(frame, click_x, click_y, max_dist=80)
        if hint is not None:
            hx, hy, hw, hh = hint
            centre_x = hx + hw // 2
            centre_y = hy + hh // 2
            logger.info(f"Proposal hint: centre ({centre_x},{centre_y}) size {hw}x{hh}")

        bbox = self._make_patch(frame, centre_x, centre_y, INIT_PATCH_SIZE)
        self._init_cv_tracker(frame, bbox)
        if self._cv_tracker is not None:
            self.last_bbox = bbox
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
            x, y, w, h = [int(v) for v in rect]
            # Sanity-check: reject obviously degenerate boxes
            if w > 5 and h > 5:
                self.last_bbox = (x, y, w, h)
                self.fail_count = 0
            else:
                ok = False

        if not ok:
            self.fail_count += 1
            if self.fail_count >= self.max_fails * 2:
                # Try re-acquisition near last known position
                ref_cx, ref_cy = self._bbox_center(self.last_bbox)
                new_bbox = self.proposer.find_nearest(frame, ref_cx, ref_cy, max_dist=120)
                if new_bbox is not None:
                    nx, ny, nw, nh = new_bbox
                    centre_x = nx + nw // 2
                    centre_y = ny + nh // 2
                    patch = self._make_patch(frame, centre_x, centre_y, INIT_PATCH_SIZE)
                    self._init_cv_tracker(frame, patch)
                    self.last_bbox = patch
                    self.fail_count = 0
                    logger.info("Re-acquired; CSRT reinitialised")
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
