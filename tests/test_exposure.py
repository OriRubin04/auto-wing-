"""
The camera must never slow the frame rate to expose a dark scene.

Sixth Pi run (USB global-shutter camera, indoors, evening): with the camera's
own auto-exposure the loop took 1-33 ms per frame, but frames arrived 400 ms
apart - 2.5 fps - whenever it pointed at something dark, because the camera
lengthened the exposure past the frame period.  The video looked choked.

Run:  python -m unittest discover tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onboard.exposure import next_setting, read_controls, EXP_SHORT, GAIN_MIN, GAIN_MAX  # noqa: E402

FPS = 30
FRAME = int(10000 / FPS)          # frame period in V4L2 units (100 us)


def settle(mean_of, exp=156, gain=32, steps=40):
    """Run the controller against a scene whose brightness follows exposure."""
    for _ in range(steps):
        exp, gain = next_setting(exp, gain, mean_of(exp, gain), FPS)
    return exp, gain


class ExposureNeverSlowsTheCamera(unittest.TestCase):

    def test_dark_scene_keeps_the_frame_rate(self):
        # Lens cap on: nothing helps, and the exposure must still stop short
        # of the frame period.
        exp, gain = settle(lambda e, g: 2.0)
        self.assertLess(exp, FRAME)
        self.assertEqual(gain, GAIN_MAX)

    def test_dim_scene_raises_gain_before_long_exposure(self):
        # Short exposure is what a global shutter is for: blur grows with it.
        # A scene that reaches the target at gain 64 must not be bought with
        # exposure past EXP_SHORT.
        exp, gain = settle(lambda e, g: min(255.0, 110.0 * e * g / (EXP_SHORT * 64)))
        self.assertLessEqual(exp, EXP_SHORT)
        self.assertGreater(gain, GAIN_MIN)

    def test_bright_scene_shortens_the_exposure(self):
        exp, gain = settle(lambda e, g: min(255.0, 110.0 * e * g / (5 * GAIN_MIN)))
        self.assertLessEqual(exp, 10)
        self.assertEqual(gain, GAIN_MIN)

    def test_any_camera_ranges_are_read_from_the_driver(self):
        # Plug and play: nothing about the camera is written in the code.
        text = ("  gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0\n"
                "  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3\n"
                "  exposure_time_absolute 0x009a0902 (int) : min=3 max=2047 step=1 default=250\n")
        c = read_controls(text)
        self.assertEqual(c['gain'], (0, 100))
        self.assertEqual(c['exposure_time_absolute'], (3, 2047))
        self.assertIn('auto_exposure', c)

    def test_camera_without_gain_still_keeps_the_frame_rate(self):
        exp, gain = 100, 1
        for _ in range(40):
            exp, gain = next_setting(exp, gain, 2.0, FPS, gain_min=1, gain_max=1)
        self.assertLess(exp, FRAME)
        self.assertEqual(gain, 1)

    def test_good_picture_is_left_alone(self):
        self.assertEqual(next_setting(EXP_SHORT, 50, 115.0, FPS), (EXP_SHORT, 50))


if __name__ == '__main__':
    unittest.main()
