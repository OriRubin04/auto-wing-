import time
import math


class PID:
    def __init__(self, kp, ki, kd, output_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, error):
        now = time.time()
        if self._prev_time is None:
            dt = 0.05
        else:
            dt = now - self._prev_time
            dt = max(dt, 0.001)
        self._prev_time = now

        self._integral += error * dt
        # Anti-windup clamp
        self._integral = max(-self.output_limit / self.ki if self.ki else -1e6,
                             min(self.output_limit / self.ki if self.ki else 1e6,
                                 self._integral))
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class TrackingController:
    """Converts normalized pixel error to roll/pitch attitude commands."""

    def __init__(self, cfg):
        r = cfg['roll_pid']
        p = cfg['pitch_pid']
        self.roll_pid = PID(r['kp'], r['ki'], r['kd'], r['output_limit'])
        self.pitch_pid = PID(p['kp'], p['ki'], p['kd'], p['output_limit'])
        self.throttle = cfg['throttle']

    def reset(self):
        self.roll_pid.reset()
        self.pitch_pid.reset()

    def compute(self, error_x_norm, error_y_norm):
        """
        error_x_norm: (target_cx - frame_cx) / (frame_w/2), range -1..1
        error_y_norm: (target_cy - frame_cy) / (frame_h/2), range -1..1
        Returns: (roll_rad, pitch_rad, throttle)
          roll  positive -> bank right (turn right toward target)
          pitch negative -> pitch down when target is below
        """
        roll = self.roll_pid.compute(error_x_norm)
        pitch = -self.pitch_pid.compute(error_y_norm)  # invert: target below -> pitch down
        return roll, pitch, self.throttle
