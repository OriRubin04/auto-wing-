"""
The init box must cover the object that was clicked, not a speck on it.

Third Pi run (cars from a 20th floor, ~40x80 px): the saliency hint was a
5x3 px reflection on the roof, so the box was the 24 px floor on one part of
the car.  The DoG filter (sigma 1.2 / 6) only sees structure up to ~12 px; a
uniform 40 px car shows as its rim and its highlights, in fragments.

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


def _road(seed):
    rng = np.random.default_rng(seed)
    g = rng.integers(30, 70, size=(FH, FW), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(g, (0, 0), 2), cv2.COLOR_GRAY2BGR)


def _car(img, x, y, w, h):
    """White car seen from above: body, dark windscreen and rear window."""
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (225, 225, 225), -1)
    cv2.rectangle(img, (x + 4, y + h // 5), (x + w - 5, y + h // 5 + h // 6), (40, 40, 40), -1)
    cv2.rectangle(img, (x + 5, y + 3 * h // 4), (x + w - 6, y + 3 * h // 4 + h // 10), (40, 40, 40), -1)
    cv2.circle(img, (x + w // 2 + 6, y + h // 2 - 4), 2, (255, 255, 255), -1)   # a glint
    return cv2.GaussianBlur(img, (0, 0), 1.0)


class InitBoxCoversTheObject(unittest.TestCase):

    def test_box_covers_a_large_car_clicked_near_its_edge(self):
        x, y, w, h = 600, 320, 36, 76
        frame = _car(_road(0), x, y, w, h)
        det = ObjectDetector({})
        self.assertTrue(det.select_target(x + 3, y + h // 2, frame))
        bx, by, bw, bh = det.last_bbox

        ix = max(0, min(bx + bw, x + w) - max(bx, x))
        iy = max(0, min(by + bh, y + h) - max(by, y))
        covered = ix * iy / float(w * h)
        self.assertGreaterEqual(covered, 0.9, f"box {det.last_bbox} covers {covered:.0%} of the car")
        self.assertLessEqual(bw * bh, 2.0 * w * h, f"box {det.last_bbox} is mostly road")

    def test_small_target_keeps_the_floor_box(self):
        frame = _road(1)
        cv2.rectangle(frame, (636, 357), (643, 360), (230, 230, 230), -1)   # 8x4 px
        det = ObjectDetector({})
        self.assertTrue(det.select_target(640, 359, frame))
        self.assertEqual(det.last_bbox[2:], (24, 24))


def _facade(seed):
    """Light wall, dark windows with a pale frame; the glass is flat but for
    sensor noise - the inside of a window carries no texture of its own."""
    rng = np.random.default_rng(seed)
    img = np.full((FH, FW), 175, np.uint8)
    for y in range(60, FH - 60, 110):
        for x in range(60, FW - 60, 120):
            cv2.rectangle(img, (x, y), (x + 34, y + 40), 235, -1)
            cv2.rectangle(img, (x + 4, y + 4), (x + 30, y + 36), 70, -1)
    return img, rng


def _noisy(img, rng):
    n = rng.normal(0, 6, img.shape)
    return cv2.cvtColor(np.clip(img + n, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class FlatTargetsHoldLock(unittest.TestCase):

    def test_holds_a_window_on_a_still_facade(self):
        # Fourth Pi run: every window lost lock on the first update.  CSRT
        # held it; the centre of the box is flat glass, so the check that
        # the box is still on target compared noise with noise (NCC ~0.3).
        img, rng = _facade(0)
        det = ObjectDetector({})
        x, y = 60 + 4 * 120, 60 + 2 * 110
        self.assertTrue(det.select_target(x + 17, y + 20, _noisy(img, rng)))
        locked = 0
        for _ in range(30):
            det.process(_noisy(img, rng))
            locked += det.state == det.STATE_TRACKING and det.fail_count == 0
        self.assertGreaterEqual(locked, 28, f"held the window on {locked} of 30 still frames")

    def test_holds_a_window_on_a_dark_noisy_facade(self):
        # Fifth Pi run, at night: even the whole box scored 0.57-0.74 on the
        # next frame, under the fixed 0.7 bar.
        img, rng = _facade(1)
        img = (img.astype(np.float32) * 0.18).astype(np.uint8)
        det = ObjectDetector({})
        x, y = 60 + 4 * 120, 60 + 2 * 110
        noisy = lambda: cv2.cvtColor(
            np.clip(img + rng.normal(0, 9, img.shape), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.assertTrue(det.select_target(x + 17, y + 20, noisy()))
        locked = 0
        for _ in range(30):
            det.process(noisy())
            locked += det.state == det.STATE_TRACKING and det.fail_count == 0
        self.assertGreaterEqual(locked, 28, f"held the window on {locked} of 30 still frames")


if __name__ == '__main__':
    unittest.main()
