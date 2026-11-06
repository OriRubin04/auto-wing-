"""
Re-acquisition after a lost lock must look for the target that was selected,
not for whatever stands out.

Second Pi run: after 20 failed frames the detector re-initialised the tracker
on the most salient blob within 120 px of the last position - the table, a
vase, a person's shirt - and from then on reported a clean lock on it.

Run:  python -m unittest discover tests
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onboard.detector import ObjectDetector  # noqa: E402

FW, FH = 1280, 720


def _ground(seed):
    rng = np.random.default_rng(seed)
    g = rng.integers(90, 170, size=(FH, FW), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(g, (0, 0), 3), cv2.COLOR_GRAY2BGR)


class LostLockReacquisition(unittest.TestCase):

    def test_does_not_lock_onto_a_different_object_after_a_loss(self):
        det = ObjectDetector({})
        first = _ground(0)
        cv2.circle(first, (640, 360), 7, (20, 20, 20), -1)
        self.assertTrue(det.select_target(640, 360, first))
        for _ in range(3):
            det.process(first)

        # The target is gone; something else stands out right next to where
        # it was.
        other = np.full((FH, FW, 3), 128, np.uint8)
        cv2.rectangle(other, (690, 380), (720, 410), (250, 250, 250), -1)
        for i in range(40):
            target, state, _ = det.process(other)
            locked = state == det.STATE_TRACKING and det.fail_count == 0
            self.assertFalse(locked, f"frame {i}: reports a fresh lock at {target} "
                                     f"on an object that was never selected")


if __name__ == '__main__':
    unittest.main()
