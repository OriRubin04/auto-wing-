"""
Detector lock validity: a CSRT "success" must not count as a measurement
when it is geometrically impossible, and re-acquisition must not accept a
blob that does not look like the target.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onboard.detector import ObjectDetector  # noqa: E402

W, H = 640, 480
RNG = np.random.default_rng(0)
GROUND = RNG.integers(90, 130, (H + 600, W, 3), dtype=np.uint8)


def frame(n, vanish=10**9, dx=1, scroll=2):
    """Textured ground sliding under a dark vehicle-shaped target."""
    off = min(scroll * n, 600)
    f = GROUND[off:off + H].copy()
    if n < vanish:
        cx, cy = 320 + dx * n, 240
        pts = cv2.boxPoints(((cx, cy), (18, 10), 0)).astype(np.int32)
        cv2.fillPoly(f, [pts], (20, 20, 20))
    return f


def detector():
    return ObjectDetector({'max_detection_failures': 10})


class BoxValidity(unittest.TestCase):

    def test_jump_is_rejected(self):
        d = detector()
        d.last_bbox = (300, 220, 40, 40)
        self.assertIn('jumped', d._reject_reason(frame(0), (10, 0, 40, 40)))

    def test_small_move_is_accepted(self):
        d = detector()
        d.last_bbox = (300, 220, 40, 40)
        self.assertIsNone(d._reject_reason(frame(0), (330, 240, 40, 40)))

    def test_mostly_offscreen_is_rejected(self):
        d = detector()
        d.last_bbox = (0, 0, 40, 40)
        self.assertIn('inside frame', d._reject_reason(frame(0), (-25, -25, 40, 40)))

    def test_degenerate_is_rejected(self):
        d = detector()
        self.assertIn('degenerate', d._reject_reason(frame(0), (100, 100, 3, 40)))


class TrackerFollowsTarget(unittest.TestCase):

    def test_lock_holds_and_reports_full_confidence(self):
        d = detector()
        self.assertTrue(d.select_target(320, 240, frame(0)))
        for n in range(1, 40):
            t, state, _ = d.process(frame(n))
        self.assertEqual(state, ObjectDetector.STATE_TRACKING)
        self.assertEqual(d.fail_count, 0)
        self.assertAlmostEqual(d.confidence(), 1.0)
        self.assertLess(abs(t[0] - (320 + 39)), 12)


class ReacquisitionGate(unittest.TestCase):

    def test_vanished_target_ends_in_searching_not_false_lock(self):
        d = detector()
        d.select_target(320, 240, frame(0))
        states = []
        for n in range(1, 400):
            t, state, _ = d.process(frame(n, vanish=40))
            states.append(state)
            if state == ObjectDetector.STATE_SEARCHING:
                break
        self.assertEqual(states[-1], ObjectDetector.STATE_SEARCHING,
                         "detector kept a 'lock' after the target vanished")

    def test_reacquire_accepts_real_target(self):
        d = detector()
        d.select_target(320, 240, frame(0))
        d.process(frame(1))
        # Pretend CSRT coasted for the whole window while the target moved 25 px.
        d.fail_count = d.coast_limit
        found = d._reacquire(frame(26, dx=1))
        self.assertIsNotNone(found)
        cx, cy, score = found
        self.assertGreater(score, d.reacquire_min_ncc)
        self.assertLess(abs(cx - (320 + 26)), 6)

    def test_reacquire_rejects_clutter(self):
        d = detector()
        d.select_target(320, 240, frame(0))
        d.process(frame(1))
        d.fail_count = d.coast_limit
        self.assertIsNone(d._reacquire(frame(60, vanish=40)))

    def test_template_is_target_sized(self):
        d = detector()
        d.select_target(320, 240, frame(0))
        side = d._template.shape[0]
        self.assertGreaterEqual(side, ObjectDetector.TEMPLATE_MIN)
        self.assertLess(side, 72)


if __name__ == '__main__':
    unittest.main()
