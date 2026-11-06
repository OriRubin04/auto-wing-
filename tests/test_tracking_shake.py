"""
Tracking under camera shake - the two failures found on the first Pi run.

Both reproduce the real footage (logs/20261106_151228) on a synthetic scene so
the test is deterministic and needs no video file:

1. A camera jump bigger than CSRT's capture range.  For a 24 px box CSRT holds
   ~28 px of shift between frames and loses everything from ~48 px; the real
   losses happened at 53-194 px.

2. The CSRT corner bug.  On a failed update() opencv_contrib stores (-1,-1) as
   the target centre before returning false (trackerCSRT.cpp, `object_center =
   estimate_new_position(image)`), so the NEXT update searches the top-left
   corner of the image and usually reports success there.

Run:  python -m unittest discover tests
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onboard.trackers import make_tracker  # noqa: E402

FW, FH = 1280, 720
BOX = 24                      # INIT_BOX_MIN - what most real selections got


def _world(seed=0):
    """A large textured 'ground', and where the target starts on it."""
    rng = np.random.default_rng(seed)
    w = rng.integers(90, 170, size=(1600, 2400), dtype=np.uint8)
    w = cv2.GaussianBlur(w, (0, 0), 3)
    return cv2.cvtColor(w, cv2.COLOR_GRAY2BGR), (1200, 800)


def _view(world, cam, target=(1200, 800)):
    """Camera frame with its top-left at world point `cam`; one dark target."""
    x, y = cam
    f = world[y:y + FH, x:x + FW].copy()
    cv2.circle(f, (target[0] - x, target[1] - y), 7, (20, 20, 20), -1)
    return f


def _centre(rect):
    x, y, w, h = rect
    return x + w / 2.0, y + h / 2.0


def _start(world, target, cam):
    tracker, _ = make_tracker({})
    frame = _view(world, cam)
    tx, ty = target[0] - cam[0], target[1] - cam[1]
    assert tracker.init(frame, (tx - BOX // 2, ty - BOX // 2, BOX, BOX))
    return tracker


class CameraJump(unittest.TestCase):

    def test_holds_a_moving_target_through_a_60px_camera_jump(self):
        world, target = _world()
        cam = [560, 440]
        tracker = _start(world, target, cam)

        # Settle on a still camera, then one sharp 60 px jump (what a hand
        # shake or a roll input does), then still again - while the target
        # drives 2 px/frame.  The target has to move on its own: for a static
        # one, a tracker stuck in place plus the camera shift would land on
        # the right answer without tracking anything.
        path = [(0, 0)] * 5 + [(60, -40)] + [(0, 0)] * 15
        tgt = list(target)
        for i, (dx, dy) in enumerate(path):
            cam[0] += dx
            cam[1] += dy
            if i > 5:
                tgt[0] += 2
            ok, rect, _ = tracker.update(_view(world, cam, tgt))
            truth = (tgt[0] - cam[0], tgt[1] - cam[1])
            self.assertTrue(ok, f"lost the target at camera {cam}")
            cx, cy = _centre(rect)
            self.assertLess(np.hypot(cx - truth[0], cy - truth[1]), 6.0,
                            f"box at ({cx:.0f},{cy:.0f}), target at {truth}")


class FailedUpdate(unittest.TestCase):

    def test_recovers_in_place_after_one_failed_frame(self):
        world, target = _world()
        cam = (560, 440)
        tracker = _start(world, target, cam)
        frame = _view(world, cam)
        truth = (target[0] - cam[0], target[1] - cam[1])

        for _ in range(3):
            tracker.update(frame)

        # One frame with nothing in it (a blurred-out or dropped frame).
        ok, _, _ = tracker.update(np.full_like(frame, 128))
        self.assertFalse(ok, "a blank frame should be reported as a failure")

        # The scene comes back unchanged.  The box must come back to the
        # target, not to the top-left corner of the image.
        for _ in range(3):
            ok, rect, _ = tracker.update(frame)
            self.assertTrue(ok)
            cx, cy = _centre(rect)
            self.assertLess(np.hypot(cx - truth[0], cy - truth[1]), 6.0,
                            f"box at ({cx:.0f},{cy:.0f}), target at {truth}")


if __name__ == '__main__':
    unittest.main()
