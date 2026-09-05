import time


class PID:
    def __init__(self, kp, ki, kd, output_limit=1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, error):
        error = float(error)
        now = time.time()
        dt = 0.05 if self._prev_time is None else max(now - self._prev_time, 0.001)
        self._prev_time = now

        self._integral += error * dt
        ilim = self.output_limit / self.ki if self.ki else 1e6
        self._integral = float(max(-ilim, min(ilim, self._integral)))

        derivative = float((error - self._prev_error) / dt)
        self._prev_error = error

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(max(-self.output_limit, min(self.output_limit, out)))


class TrackingController:
    """
    Converts normalised pixel error to RC override commands for a fixed-wing.

    Fixed-wing guidance rules
    - Lateral (err_x): use ROLL (bank-and-turn). This is the primary axis.
    - Vertical (err_y): use a SMALL pitch correction + throttle trim.
      Aggressive pitch on a fixed-wing -> stall / crash. Limit hard.
    - A deadband ignores tiny errors (avoids constant micro-corrections).
    """

    DEADBAND = 0.10
    SMOOTH   = 0.0

    def __init__(self, cfg):
        r = cfg['roll_pid']
        p = cfg['pitch_pid']
        self.roll_pid  = PID(r['kp'], r['ki'], r['kd'], r['output_limit'])
        self.pitch_pid = PID(p['kp'], p['ki'], p['kd'], p['output_limit'])
        self.throttle_base = float(cfg.get('throttle', 0.6))

        self._roll_out  = 0.0
        self._pitch_out = 0.0

    def reset(self):
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self._roll_out  = 0.0
        self._pitch_out = 0.0

    def compute(self, error_x_norm, error_y_norm):
        """
        error_x_norm  -1..+1  (negative = target left of centre)
        error_y_norm  -1..+1  (negative = target above centre, positive = below)

        Returns (roll_norm, pitch_norm, throttle_norm).
        """
        ex_f = float(error_x_norm)
        ey_f = float(error_y_norm)
        ex = 0.0 if abs(ex_f) < self.DEADBAND else ex_f
        ey = 0.0 if abs(ey_f) < self.DEADBAND else ey_f

        raw_roll  = self.roll_pid.compute(ex)
        raw_pitch = self.pitch_pid.compute(ey)
        throttle  = self.throttle_base

        self._roll_out  = self.SMOOTH * self._roll_out  + (1 - self.SMOOTH) * raw_roll
        self._pitch_out = self.SMOOTH * self._pitch_out + (1 - self.SMOOTH) * raw_pitch

        return float(self._roll_out), float(self._pitch_out), throttle
