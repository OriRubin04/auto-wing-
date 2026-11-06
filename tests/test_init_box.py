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


if __name__ == '__main__':
    unittest.main()
