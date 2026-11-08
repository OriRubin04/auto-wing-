"""
Auto-exposure that never costs frame rate.

Sixth Pi run, USB global-shutter camera (32e4:0144) indoors in the evening:
the loop took 1-33 ms per frame but frames arrived up to 400 ms apart -
2.5 fps - whenever the camera faced something dark.  The camera's own
auto-exposure (V4L2 "Aperture Priority") lengthens the exposure past the
frame period, and the driver has no control to cap it.  With the lens cap on
it ran at 5.5 fps.

So the camera is put in manual mode and exposure is set from here: gain is
raised before the exposure grows past EXP_SHORT (blur grows with exposure,
and short exposure is the point of a global shutter), and the exposure never
reaches the frame period.  A dark scene comes out dark and noisy, at full
frame rate - for tracking that is the right trade.

Each control write is a USB control transfer, measured 5-10 ms on the Pi, so
the writes run on their own thread with their own handle (v4l2-ctl), never on
the control loop.  Units are V4L2's: exposure in 100 us steps.
"""
import logging
import shutil
import subprocess
import threading

logger = logging.getLogger(__name__)

TARGET = 110.0        # mean grey level aimed for
DEADBAND = 25.0       # leave the picture alone within this of TARGET
EXP_MIN = 1
EXP_SHORT = 100       # 10 ms: past this, raise gain first
GAIN_MIN = 16         # defaults only; the real range is read from the camera
GAIN_MAX = 255
EVERY = 10            # frames between measurements


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def read_controls(text):
    """{name: (min, max)} from `v4l2-ctl --list-ctrls` output."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        nums = dict(p.split('=', 1) for p in parts if '=' in p)
        try:
            out[parts[0]] = (int(nums['min']), int(nums['max']))
        except (KeyError, ValueError):
            pass
    return out


def next_setting(exp, gain, mean, fps, gain_min=GAIN_MIN, gain_max=GAIN_MAX, exp_min=EXP_MIN):
    """(exposure, gain) for the next frames, given the mean of this one."""
    # Within the deadband the light is kept, but still re-split: a long
    # exposure inherited from the camera's own mode gets traded for gain.
    ratio = 1.0 if abs(mean - TARGET) <= DEADBAND else _clip(TARGET / max(mean, 1.0), 0.5, 2.0)
    light = exp * gain * ratio
    longest = int(0.9 * 10000 / fps)          # always inside the frame period
    e = _clip(light / gain_min, exp_min, min(EXP_SHORT, longest))
    g = _clip(light / e, gain_min, gain_max)
    if g >= gain_max:
        e = _clip(light / gain_max, e, longest)
    return int(round(e)), int(round(g))


class ExposureControl:
    """Feed it every frame; it adjusts the camera from its own thread."""

    def __init__(self, index, fps):
        self.device = f"/dev/video{int(index)}"
        self.fps = max(1, int(fps))
        self._n = 0
        self._mean = None
        self._force = True
        self._wake = threading.Event()
        self.enabled = False

        # Any camera: take its ranges from the driver.  A camera without a
        # manual exposure control keeps its own auto-exposure, as before.
        ctrls = {}
        if shutil.which('v4l2-ctl'):
            try:
                r = subprocess.run(['v4l2-ctl', '-d', self.device, '--list-ctrls'],
                                   capture_output=True, text=True, timeout=3)
                ctrls = read_controls(r.stdout)
            except Exception:
                pass
        if 'auto_exposure' not in ctrls or 'exposure_time_absolute' not in ctrls:
            logger.warning(f"No manual exposure on {self.device} - the camera keeps its "
                           "own auto-exposure, which can drop the frame rate in the dark")
            return
        self.exp_min = max(1, ctrls['exposure_time_absolute'][0])
        self.has_gain = 'gain' in ctrls
        self.gain_min, self.gain_max = ctrls.get('gain', (1, 1))
        self.gain_min = max(1, self.gain_min)
        self.exp, self.gain = EXP_SHORT, self.gain_min
        self.enabled = True
        threading.Thread(target=self._run, name='exposure', daemon=True).start()
        logger.info(f"Exposure: manual, at most {int(0.9 * 1000 / self.fps)} ms, "
                    f"gain {self.gain_min}..{self.gain_max} ({self.device})")

    def feed(self, frame):
        """Cheap: a strided mean every EVERY frames, no I/O."""
        if not self.enabled:
            return
        self._n += 1
        if self._n % EVERY:
            return
        self._mean = float(frame[::16, ::16].mean())
        self._wake.set()

    def reset(self):
        """After the camera is reopened its controls are back to auto."""
        self._force = True

    def _run(self):
        while self.enabled:
            self._wake.wait()
            self._wake.clear()
            mean = self._mean
            exp, gain = next_setting(self.exp, self.gain, mean, self.fps,
                                     self.gain_min, self.gain_max, self.exp_min)
            if (exp, gain) == (self.exp, self.gain) and not self._force:
                continue
            self._force = False
            # auto_exposure=1 is V4L2_EXPOSURE_MANUAL, the same on every driver.
            ctrls = f"auto_exposure=1,exposure_time_absolute={exp}"
            if self.has_gain:
                ctrls += f",gain={gain}"
            try:
                r = subprocess.run(['v4l2-ctl', '-d', self.device, '-c', ctrls],
                                   capture_output=True, text=True, timeout=2)
            except Exception as e:
                r = None
                err = str(e)
            if r is None or r.returncode != 0:
                err = r.stderr.strip() if r is not None else err
                logger.warning(f"Exposure control failed ({err}) - leaving the camera on its own")
                self.enabled = False
                return
            self.exp, self.gain = exp, gain
            logger.debug(f"Exposure {exp / 10:.1f} ms gain {gain} (mean {mean:.0f})")
