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


class FollowedVehicle(unittest.TestCase):

    def test_holds_a_vehicle_the_camera_keeps_centred(self):
        # The main mission profile: a vehicle drives over the ground and the
        # aircraft keeps it centred, so the ground streams past at the
        # vehicle's speed while the target sits still in the frame.  Plain
        # CSRT holds this forever; a stabiliser that only ever accumulates
        # the ground's motion slides the target off its own canvas.
        # Ground with structure at the scale the motion estimate works at
        # (it sees a 320 px copy).  Fine noise alone averages out to flat grey
        # there, and the only thing left to correlate is the target itself.
        rng = np.random.default_rng(1)
        coarse = rng.integers(70, 190, size=(125, 375), dtype=np.uint8)
        world = cv2.resize(coarse, (3000, 1000), interpolation=cv2.INTER_CUBIC)
        world = cv2.cvtColor(cv2.GaussianBlur(world, (0, 0), 2), cv2.COLOR_GRAY2BGR)
        tgt = [700, 500]
        cam = [tgt[0] - FW // 2, tgt[1] - FH // 2]
        tracker = _start(world, tgt, cam)
        tracker.init(_view(world, cam, tgt),
                     (FW // 2 - BOX // 2, FH // 2 - BOX // 2, BOX, BOX))

        for i in range(200):                       # 6 px/frame, 1200 px total
            tgt[0] += 6
            cam[0] = tgt[0] - FW // 2
            ok, rect, _ = tracker.update(_view(world, cam, tgt))
            self.assertTrue(ok, f"lost the vehicle at frame {i}")
            cx, cy = _centre(rect)
            self.assertLess(np.hypot(cx - FW // 2, cy - FH // 2), 6.0,
                            f"frame {i}: box at ({cx:.0f},{cy:.0f})")


class TurningVehicle(unittest.TestCase):
    # Found in review: the lost-lock check compared every frame with the
    # patch taken at selection, so a vehicle that turned ~8 degrees no longer
    # matched it - and a backoff re-init then locked onto the ground.  Roads
    # turn and the aircraft orbits, so the target's look changes all the time.

    def test_holds_a_vehicle_through_a_slow_turn(self):
        rng = np.random.default_rng(1)
        coarse = rng.integers(70, 190, size=(125, 375), dtype=np.uint8)
        world = cv2.resize(coarse, (3000, 1000), interpolation=cv2.INTER_CUBIC)
        world = cv2.cvtColor(cv2.GaussianBlur(world, (0, 0), 2), cv2.COLOR_GRAY2BGR)

        def frame(i, x):
            f = world[500 - FH // 2:500 + FH // 2, x - FW // 2:x + FW // 2].copy()
            ang = 0.6 * i                                  # 18 deg/s at 30 fps
            body = cv2.boxPoints(((FW // 2, FH // 2), (16, 8), ang)).astype(np.int32)
            cv2.fillPoly(f, [body], (25, 25, 25))
            fx = FW // 2 + np.cos(np.radians(ang)) * 4.8
            fy = FH // 2 + np.sin(np.radians(ang)) * 4.8
            cab = cv2.boxPoints(((fx, fy), (4.8, 6.4), ang)).astype(np.int32)
            cv2.fillPoly(f, [cab], (230, 230, 230))
            return f

        x = 700
        tracker, _ = make_tracker({})
        self.assertTrue(tracker.init(frame(0, x), (FW // 2 - 12, FH // 2 - 12, 25, 25)))
        held = 0
        for i in range(1, 201):                            # 120 degrees
            x += 6
            ok, rect, _ = tracker.update(frame(i, x))
            if ok:
                cx, cy = _centre(rect)
                self.assertLess(np.hypot(cx - FW // 2, cy - FH // 2), 6.0,
                                f"frame {i} ({0.6 * i:.0f} deg): lock on "
                                f"({cx:.0f},{cy:.0f}), vehicle at the centre")
                held += 1
        self.assertGreater(held, 190, f"held only {held} of 200 frames")


class FailedUpdate(unittest.TestCase):

    def test_a_blank_frame_in_the_middle_of_a_jump_does_not_lose_the_jump(self):
        # Shake is when frames blur out.  If the camera jumps across a frame
        # that cannot be measured, the jump still has to be accounted for:
        # measure the next good frame against the last good one.
        world, target = _world()
        cam = [560, 440]
        tracker = _start(world, target, cam)
        for _ in range(3):
            tracker.update(_view(world, cam))

        tracker.update(np.full((FH, FW, 3), 128, np.uint8))   # unmeasurable
        cam[0] += 60
        cam[1] -= 40
        tgt = list(target)
        for i in range(10):                       # target drives, see above
            tgt[0] += 2
            ok, rect, _ = tracker.update(_view(world, cam, tgt))
            truth = (tgt[0] - cam[0], tgt[1] - cam[1])
            self.assertTrue(ok, f"lost the target {i} frames after the jump")
            cx, cy = _centre(rect)
            self.assertLess(np.hypot(cx - truth[0], cy - truth[1]), 6.0,
                            f"box at ({cx:.0f},{cy:.0f}), target at {truth}")

    def test_a_run_of_failures_does_not_reinit_every_frame(self):
        # Re-initialising costs about as much as an update.  During a long
        # failure the detector is coasting anyway; retry, but back off.
        world, target = _world()
        cam = (560, 440)
        tracker = _start(world, target, cam)
        calls = []
        real_init = tracker.inner.init
        tracker.inner.init = lambda *a: calls.append(1) or real_init(*a)
        tracker.update(_view(world, cam))

        blank = np.full((FH, FW, 3), 128, np.uint8)
        for _ in range(20):                        # coast_limit frames
            ok, _, _ = tracker.update(blank)
            self.assertFalse(ok)
        self.assertLessEqual(len(calls), 6, f"{len(calls)} re-inits in 20 failed frames")

    def test_a_scored_tracker_is_not_reinitialised_on_failure(self):
        # The corner bug is CSRT's.  A tracker with a real score (ViT) fails
        # for a reason the detector should keep counting, so its failures go
        # through untouched.
        from onboard.trackers import PixelTracker, StabilizedTracker
        world, target = _world()
        cam = (560, 440)
        inner = PixelTracker('CSRT')
        inner.graded = True                     # stands in for ViT here
        tracker = StabilizedTracker(inner)
        tx, ty = target[0] - cam[0], target[1] - cam[1]
        tracker.init(_view(world, cam), (tx - BOX // 2, ty - BOX // 2, BOX, BOX))
        calls = []
        real_init = inner.init
        inner.init = lambda *a: calls.append(1) or real_init(*a)

        blank = np.full((FH, FW, 3), 128, np.uint8)
        for _ in range(5):
            tracker.update(blank)
        self.assertEqual(calls, [])

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


def _blind(tracker):
    """Make the camera-motion estimate miss every jump, as it did on the
    second Pi run when the picture rotated or smeared (2312, 2488, 2802)."""
    tracker._shift = lambda frame: np.zeros(2)


class LostLock(unittest.TestCase):
    # Second Pi run: after a loss CSRT kept reporting success on blank
    # ceiling, on a lookalike lamp, or off the frame, and the box "wandered".
    # CSRT has no usable lost-lock signal, so the wrapper has to check the
    # box itself against what was selected.

    def test_reports_lost_when_the_box_lands_on_other_ground(self):
        world, target = _world()
        cam = [560, 440]
        tracker = _start(world, target, cam)
        for _ in range(3):
            self.assertTrue(tracker.update(_view(world, cam))[0])

        _blind(tracker)
        cam[0] += 150                  # beyond CSRT's reach, and not measured
        cam[1] -= 100
        frame = _view(world, cam)
        truth = (target[0] - cam[0], target[1] - cam[1])
        for i in range(5):
            ok, rect, _ = tracker.update(frame)
            if ok:
                cx, cy = _centre(rect)
                self.assertLess(np.hypot(cx - truth[0], cy - truth[1]), 6.0,
                                f"frame {i}: reports a lock at ({cx:.0f},{cy:.0f}), "
                                f"target is at {truth}")

    def test_finds_the_target_again_after_a_jump_it_could_not_follow(self):
        world, target = _world()
        cam = [560, 440]
        tracker = _start(world, target, cam)
        for _ in range(3):
            tracker.update(_view(world, cam))

        _blind(tracker)
        cam[0] += 150
        cam[1] -= 100
        frame = _view(world, cam)
        truth = (target[0] - cam[0], target[1] - cam[1])
        oks = []
        for i in range(5):
            ok, rect, _ = tracker.update(frame)
            oks.append(ok)
        self.assertTrue(ok, f"never found it again: {oks}")
        cx, cy = _centre(rect)
        self.assertLess(np.hypot(cx - truth[0], cy - truth[1]), 6.0,
                        f"box at ({cx:.0f},{cy:.0f}), target at {truth}")

    def test_finds_a_large_target_again(self):
        # A big box takes the coarse-to-fine search path (the template is
        # shrunk first), which a 24 px box never reaches.
        world, target = _world()
        cv2.circle(world, target, 28, (20, 20, 20), -1)
        cv2.circle(world, target, 12, (200, 200, 200), -1)
        cam = [560, 440]
        tracker, _ = make_tracker({})
        tx, ty = target[0] - cam[0], target[1] - cam[1]
        self.assertTrue(tracker.init(_view(world, cam), (tx - 32, ty - 32, 64, 64)))
        for _ in range(3):
            tracker.update(_view(world, cam))

        _blind(tracker)
        cam[0] += 150
        cam[1] -= 100
        frame = _view(world, cam)
        truth = (target[0] - cam[0], target[1] - cam[1])
        for _ in range(5):
            ok, rect, _ = tracker.update(frame)
        self.assertTrue(ok, "never found it again")
        cx, cy = _centre(rect)
        self.assertLess(np.hypot(cx - truth[0], cy - truth[1]), 4.0,
                        f"box at ({cx:.0f},{cy:.0f}), target at {truth}")

    def test_does_not_pick_one_of_two_lookalikes(self):
        # The second run's light bar: a row of identical lamps.  When the
        # search finds the target's pattern in more than one place it cannot
        # know which is the one selected, and must stay lost rather than guess.
        world, target = _world()
        twin = (target[0] - 150, target[1] - 90)      # earlier in raster order
        patch = world[target[1] - 30:target[1] + 30, target[0] - 30:target[0] + 30].copy()
        cv2.circle(patch, (30, 30), 7, (20, 20, 20), -1)
        world[twin[1] - 30:twin[1] + 30, twin[0] - 30:twin[0] + 30] = patch
        cam = [560, 440]
        tracker = _start(world, target, cam)
        for _ in range(3):
            tracker.update(_view(world, cam))

        _blind(tracker)
        cam[0] += 150
        cam[1] -= 100
        frame = _view(world, cam)
        truth = (target[0] - cam[0], target[1] - cam[1])
        for i in range(5):
            ok, rect, _ = tracker.update(frame)
            if ok:
                cx, cy = _centre(rect)
                self.assertLess(np.hypot(cx - truth[0], cy - truth[1]), 6.0,
                                f"frame {i}: locked onto the lookalike at "
                                f"({cx:.0f},{cy:.0f}), target is at {truth}")


if __name__ == '__main__':
    unittest.main()
