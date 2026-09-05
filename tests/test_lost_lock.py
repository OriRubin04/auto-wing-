"""
Lost-lock watchdog: the aircraft must end tracking on its own, with no GCS.

TrackingSystem is built without a camera, flight controller or sockets by
bypassing __init__ and wiring in fakes.  The main loop is not run; the
per-frame decisions it makes are exercised directly.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onboard.main import TrackingSystem  # noqa: E402


class FakeMav:
    def __init__(self):
        self.overrides = []
        self.released = 0
        self.modes = []

    def send_rc_override(self, roll, pitch, thr):
        self.overrides.append((roll, pitch, thr))

    def release_rc_override(self):
        self.released += 1

    def set_rtl_mode(self):
        self.modes.append('RTL')

    def set_acro_mode(self):
        self.modes.append('ACRO')


class FakeDetector:
    STATE_SEARCHING = 'searching'
    STATE_TRACKING = 'tracking'

    def __init__(self):
        self.state = self.STATE_TRACKING
        self.fail_count = 0
        self.last_bbox = None


class FakeLog:
    def __init__(self):
        self.events = []

    def log_event(self, kind, **data):
        self.events.append((kind, data))


def make_system(lost_timeout=3.0):
    s = object.__new__(TrackingSystem)
    s.mav = FakeMav()
    s.detector = FakeDetector()
    s.flight_log = FakeLog()
    s.tracking = True
    s.lost_timeout = lost_timeout
    s._last_lock_t = 100.0
    s._last_bank_rad = 0.0
    s._last_cmd = (0.0, 0.0, 0.0)
    s._last_err = (0.0, 0.0)
    return s


class LostLockWatchdog(unittest.TestCase):

    def test_no_abort_before_timeout(self):
        s = make_system(lost_timeout=3.0)
        s._check_lost_lock(100.0 + 2.9)
        self.assertTrue(s.tracking)
        self.assertEqual(s.mav.modes, [])
        self.assertEqual(s.mav.released, 0)

    def test_abort_to_rtl_at_timeout(self):
        s = make_system(lost_timeout=3.0)
        s._check_lost_lock(100.0 + 3.0)
        self.assertFalse(s.tracking)
        self.assertEqual(s.mav.modes, ['RTL'])
        self.assertEqual(s.mav.released, 1)
        self.assertEqual(s.detector.state, FakeDetector.STATE_SEARCHING)
        kinds = [k for k, _ in s.flight_log.events]
        self.assertIn('lost_lock_rtl', kinds)

    def test_abort_fires_once(self):
        s = make_system(lost_timeout=3.0)
        s._check_lost_lock(104.0)
        s._check_lost_lock(105.0)
        s._check_lost_lock(200.0)
        self.assertEqual(s.mav.modes, ['RTL'])
        self.assertEqual(s.mav.released, 1)

    def test_not_tracking_never_aborts(self):
        s = make_system(lost_timeout=3.0)
        s.tracking = False
        s._check_lost_lock(1e9)
        self.assertEqual(s.mav.modes, [])

    def test_fresh_lock_resets_clock(self):
        s = make_system(lost_timeout=3.0)
        s._last_lock_t = 102.5          # a fresh measurement arrived later
        s._check_lost_lock(105.0)       # only 2.5 s since then
        self.assertTrue(s.tracking)

    def test_seconds_lost_reports_zero_when_idle(self):
        s = make_system()
        s.tracking = False
        self.assertEqual(s._seconds_lost(999.0), 0.0)


class LostControlLevelsWings(unittest.TestCase):

    def test_banked_aircraft_gets_counter_roll(self):
        s = make_system()
        s._last_bank_rad = 0.5            # right bank ~29 deg
        s._send_lost_control()
        roll, pitch, thr = s.mav.overrides[-1]
        self.assertLess(roll, 0.0)        # roll left to level
        self.assertAlmostEqual(roll, -TrackingSystem.LEVEL_KP * 0.5)
        self.assertEqual(pitch, 0.0)
        self.assertEqual(thr, TrackingSystem.ACRO_THROTTLE)

    def test_level_aircraft_gets_zero_roll(self):
        s = make_system()
        s._last_bank_rad = 0.02           # inside the deadband
        s._send_lost_control()
        self.assertEqual(s.mav.overrides[-1][0], 0.0)

    def test_counter_roll_is_capped(self):
        s = make_system()
        s._last_bank_rad = 3.0
        s._send_lost_control()
        self.assertEqual(s.mav.overrides[-1][0], -0.5)


class StopPathShared(unittest.TestCase):

    def test_gcs_stop_and_watchdog_do_the_same_thing(self):
        a = make_system()
        a._stop_tracking('stop_rtl')
        b = make_system()
        b._check_lost_lock(1e9)
        for s in (a, b):
            self.assertFalse(s.tracking)
            self.assertIsNone(s._last_lock_t)
            self.assertEqual(s.mav.modes, ['RTL'])
            self.assertEqual(s.mav.released, 1)


if __name__ == '__main__':
    unittest.main()
