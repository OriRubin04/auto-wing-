import cv2
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed, YOLO unavailable")


class ObjectDetector:
    """
    YOLO-based detector with OpenCV tracker fallback.

    Workflow:
    1. User selects a point (click) on GCS.
    2. First YOLO frame finds the bounding box nearest that point.
    3. Track that box. If YOLO fails N frames, switch to OpenCV CSRT tracker.
    4. If CSRT also fails, enter SEARCHING state.
    """

    STATE_SEARCHING = "searching"
    STATE_YOLO = "yolo"
    STATE_FALLBACK = "fallback"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.state = self.STATE_SEARCHING
        self.target_class = None
        self.target_center = None  # (x, y) pixel user clicked
        self.last_bbox = None  # (x, y, w, h)
        self.fail_count = 0
        self.max_fails = cfg.get('max_detection_failures', 10)
        self._cv_tracker = None
        self._tracker_type = cfg.get('fallback_tracker', 'CSRT')

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(cfg.get('model', 'yolov8n.pt'))
                logger.info("YOLO model loaded")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")

    def select_target(self, click_x, click_y, frame):
        """Called when user clicks on GCS. Lock CSRT immediately on the clicked object."""
        self.target_center = (click_x, click_y)
        self.state = self.STATE_SEARCHING
        self.last_bbox = None
        self.fail_count = 0
        self._cv_tracker = None

        if self.model is not None:
            bbox = self._yolo_nearest(frame, click_x, click_y)
            if bbox is not None:
                self.last_bbox = bbox
                # Go straight to CSRT so we track THIS specific object's appearance,
                # not whatever YOLO happens to detect nearest on the next frame.
                self._init_cv_tracker(frame, bbox)
                self.state = self.STATE_FALLBACK
                logger.info(f"Target acquired via YOLO: {bbox}")
                return True

        # Fallback: use clicked region as initial bbox (80x80 box around click)
        h, w = frame.shape[:2]
        size = 80
        x1 = max(0, click_x - size // 2)
        y1 = max(0, click_y - size // 2)
        x1 = min(x1, w - size)
        y1 = min(y1, h - size)
        self.last_bbox = (x1, y1, size, size)
        self._init_cv_tracker(frame, self.last_bbox)
        self.state = self.STATE_FALLBACK
        logger.info(f"Target acquired via manual ROI: {self.last_bbox}")
        return True

    def process(self, frame):
        """
        Returns (bbox, state, annotated_frame) or (None, state, annotated_frame).
        bbox is (cx, cy, w, h) in pixels.
        """
        annotated = frame.copy()

        if self.state == self.STATE_SEARCHING:
            return None, self.state, annotated

        if self.state == self.STATE_YOLO:
            # Legacy path — shouldn't normally be reached now since select_target
            # goes straight to CSRT, but kept as a safety net.
            ref_cx, ref_cy = self._bbox_center(self.last_bbox)
            bbox = self._yolo_nearest(frame, ref_cx, ref_cy, max_dist=150)
            if bbox is not None:
                self.last_bbox = bbox
                self.fail_count = 0
                self._init_cv_tracker(frame, bbox)
                self.state = self.STATE_FALLBACK
            else:
                self.fail_count += 1
                bbox = self.last_bbox
                if self.fail_count >= self.max_fails:
                    logger.warning("YOLO lost target, switching to CSRT")
                    self._init_cv_tracker(frame, self.last_bbox)
                    self.state = self.STATE_FALLBACK
                    self.fail_count = 0

        elif self.state == self.STATE_FALLBACK:
            if self._cv_tracker is not None:
                ok, rect = self._cv_tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in rect]
                    bbox = (x, y, w, h)
                    self.last_bbox = bbox
                    self.fail_count = 0
                else:
                    self.fail_count += 1
                    bbox = self.last_bbox
                    if self.fail_count >= self.max_fails * 2:
                        # Try YOLO re-acquisition near last known position
                        ref_cx, ref_cy = self._bbox_center(self.last_bbox)
                        new_bbox = self._yolo_nearest(frame, ref_cx, ref_cy, max_dist=120)
                        if new_bbox is not None:
                            logger.info("YOLO re-acquired target")
                            self.last_bbox = new_bbox
                            self._init_cv_tracker(frame, new_bbox)
                            self.fail_count = 0
                        else:
                            logger.warning("CSRT lost target, entering SEARCHING")
                            self.state = self.STATE_SEARCHING
                            return None, self.state, annotated
            else:
                return None, self.state, annotated

        # Annotate frame
        if self.last_bbox:
            x, y, w, h = self.last_bbox
            cx = x + w // 2
            cy = y + h // 2
            color = (0, 255, 0) if self.state == self.STATE_YOLO else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.circle(annotated, (cx, cy), 5, color, -1)
            label = f"YOLO" if self.state == self.STATE_YOLO else "CSRT"
            cv2.putText(annotated, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return (cx, cy, w, h), self.state, annotated

        return None, self.state, annotated

    def _yolo_nearest(self, frame, ref_x, ref_y, max_dist=None):
        """Find nearest YOLO detection to (ref_x, ref_y). Reject if farther than max_dist."""
        if self.model is None:
            return None
        try:
            results = self.model(frame, conf=self.cfg.get('confidence', 0.4),
                                 iou=self.cfg.get('iou', 0.45),
                                 device=self.cfg.get('device', 'cpu'),
                                 verbose=False)
            best_bbox = None
            best_dist = float('inf')
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    dist = math.hypot(cx - ref_x, cy - ref_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_bbox = (x1, y1, x2 - x1, y2 - y1)
            if best_bbox is not None and max_dist is not None and best_dist > max_dist:
                return None  # nearest object is too far — don't jump
            return best_bbox
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return None

    def _init_cv_tracker(self, frame, bbox):
        # Try tracker APIs in order — newer OpenCV moved them to cv2.legacy
        factories = []
        tracker_type = self._tracker_type
        for ns in [cv2, getattr(cv2, 'legacy', None)]:
            if ns is None:
                continue
            for name in [f'Tracker{tracker_type}_create',
                         'TrackerCSRT_create', 'TrackerKCF_create', 'TrackerMOSSE_create']:
                fn = getattr(ns, name, None)
                if fn:
                    factories.append(fn)
        for factory in factories:
            try:
                t = factory()
                t.init(frame, bbox)
                self._cv_tracker = t
                logger.info(f"Tracker initialized: {factory.__name__}")
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
