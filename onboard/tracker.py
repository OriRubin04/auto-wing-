"""
tracker.py - Thin state-machine wrapper around ObjectDetector.

This module is the public entry point for tracking state management.
It delegates detection/fallback logic to ObjectDetector and exposes
a simple interface to main.py.
"""
import logging
from onboard.detector import ObjectDetector

logger = logging.getLogger(__name__)


class TargetTracker:
    """
    High-level tracker that wraps ObjectDetector.

    States
    ------
    idle      : no target selected
    acquiring : target selected, waiting for first detection lock
    tracking  : actively tracking a confirmed target
    lost      : previously tracking but target disappeared
    """

    STATE_IDLE = "idle"
    STATE_ACQUIRING = "acquiring"
    STATE_TRACKING = "tracking"
    STATE_LOST = "lost"

    def __init__(self, detection_cfg):
        self.detector = ObjectDetector(detection_cfg)
        self.state = self.STATE_IDLE
        self._lost_frames = 0
        self._lost_limit = 30  # frames before transition from lost -> idle

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_target(self, x, y, frame):
        """
        Triggered by a GCS click event.

        Parameters
        ----------
        x, y  : pixel coordinates in the camera frame
        frame : current BGR frame (numpy array)

        Returns
        -------
        bool  : True if a candidate target was initialised
        """
        ok = self.detector.select_target(x, y, frame)
        if ok:
            self.state = self.STATE_ACQUIRING
            self._lost_frames = 0
            logger.info(f"TargetTracker: acquiring target at ({x}, {y})")
        return ok

    def reset(self):
        """Drop the current target and return to idle."""
        self.detector.state = ObjectDetector.STATE_SEARCHING
        self.state = self.STATE_IDLE
        self._lost_frames = 0
        logger.info("TargetTracker: reset to idle")

    def update(self, frame):
        """
        Run one detection/tracking step.

        Parameters
        ----------
        frame : current BGR frame

        Returns
        -------
        tuple : (bbox_or_None, tracker_state_str, annotated_frame)
                bbox is (cx, cy, w, h) when a target is found, else None.
        """
        if self.state == self.STATE_IDLE:
            return None, self.state, frame.copy()

        bbox, det_state, annotated = self.detector.process(frame)

        if bbox is not None:
            # Successfully located target
            self._lost_frames = 0
            self.state = self.STATE_TRACKING
        else:
            # No detection this frame
            self._lost_frames += 1
            if self.state == self.STATE_TRACKING:
                self.state = self.STATE_LOST
                logger.warning("TargetTracker: target lost")
            if self._lost_frames >= self._lost_limit:
                logger.warning("TargetTracker: giving up, returning to idle")
                self.state = self.STATE_IDLE
                self._lost_frames = 0

        return bbox, self.state, annotated

    @property
    def is_tracking(self):
        return self.state == self.STATE_TRACKING

    @property
    def is_idle(self):
        return self.state == self.STATE_IDLE
