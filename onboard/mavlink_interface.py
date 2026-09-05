import time
import logging
from pymavlink import mavutil

logger = logging.getLogger(__name__)

RC_MIN = 1000
RC_MID = 1500
RC_MAX = 2000

# ArduPlane mode numbers (NOT ArduCopter — different numbering)
MODE_MANUAL   = 0
MODE_CIRCLE   = 1   # ← mode 1 is CIRCLE, NOT acro
MODE_STABILIZE = 2
MODE_ACRO     = 4   # actual Acro mode
MODE_FBWA     = 5   # Fly By Wire A — stabilized, best for tracking
MODE_GUIDED   = 15
MODE_RTL      = 11


class MAVLinkInterface:
    """
    pymavlink wrapper using RC_CHANNELS_OVERRIDE.

    Use FBWA (mode 5) for visual tracking — it's stabilized so the plane
    holds attitude between commands and won't diverge.  ACRO (mode 4) is
    available but requires a much tighter control loop.
    """

    # Mode requests are retried from the main loop, never by blocking on it.
    MODE_RETRY_INTERVAL = 1.0    # seconds between resends
    MODE_MAX_ATTEMPTS = 8

    def __init__(self, connection_string, baudrate=115200):
        self.conn_str = connection_string
        self.baudrate = baudrate
        self.mav = None

        self._attitude = None        # newest (roll, pitch, yaw), radians
        self.current_mode = None     # newest custom_mode from HEARTBEAT
        self._pending_mode = None
        self._pending_label = ""
        self._mode_attempts = 0
        self._mode_last_send = 0.0

    def connect(self, timeout=30):
        logger.info(f"Connecting to {self.conn_str}")
        self.mav = mavutil.mavlink_connection(self.conn_str, baud=self.baudrate)
        self.mav.wait_heartbeat(timeout=timeout)
        logger.info(
            f"Heartbeat from sysid={self.mav.target_system} "
            f"compid={self.mav.target_component}"
        )

    # ------------------------------------------------------------------ #
    #  Message pump                                                        #
    # ------------------------------------------------------------------ #

    def poll(self):
        """
        Drain every pending MAVLink message once, dispatching by type.
        Non-blocking.  Must be called each iteration of the main loop.

        This is deliberately the ONLY place messages are read.  pymavlink's
        recv_match(type=X) *discards* messages that don't match its filter,
        so two separate recv_match calls silently eat each other's traffic —
        a HEARTBEAT wait throws away ATTITUDE, and vice versa.  One drain
        that dispatches by type is the only correct shape.

        Draining the whole buffer also means the cached attitude is the
        NEWEST sample rather than the oldest one still queued.
        """
        if self.mav is None:
            return
        while True:
            msg = self.mav.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == 'ATTITUDE':
                self._attitude = (msg.roll, msg.pitch, msg.yaw)
            elif mtype == 'HEARTBEAT':
                self.current_mode = msg.custom_mode
        self._service_pending_mode()

    def _service_pending_mode(self):
        """Confirm or resend an outstanding mode request.  Never blocks."""
        if self._pending_mode is None:
            return
        if self.current_mode == self._pending_mode:
            logger.info(f"Mode {self._pending_label} confirmed ✓")
            self._pending_mode = None
            return
        if time.time() - self._mode_last_send < self.MODE_RETRY_INTERVAL:
            return
        if self._mode_attempts >= self.MODE_MAX_ATTEMPTS:
            logger.warning(
                f"Could not switch to {self._pending_label} "
                f"(current mode={self.current_mode}).\n"
                f"  In Mission Planner: click the mode box (top-left) → "
                f"select {self._pending_label}\n"
                f"  RC override still works in whatever mode the plane is in."
            )
            self._pending_mode = None
            return
        self._send_mode_request()

    def _send_mode_request(self):
        # Two message types — some ArduPlane builds accept one but not the other
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self._pending_mode, 0, 0, 0, 0, 0
        )
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self._pending_mode
        )
        self._mode_last_send = time.time()
        self._mode_attempts += 1

    # ------------------------------------------------------------------ #
    #  Mode control                                                        #
    # ------------------------------------------------------------------ #

    def set_mode(self, mode_num, mode_name=""):
        """
        Request a mode change and return IMMEDIATELY.

        Confirmation and retries are handled by poll() from the main loop.
        The previous implementation blocked for up to 12 s (8 attempts x 1.5 s)
        inside the control loop, which froze video, tracking and control
        together whenever a mode failed to confirm.
        """
        self._pending_mode = mode_num
        self._pending_label = mode_name or str(mode_num)
        self._mode_attempts = 0
        self._mode_last_send = 0.0
        logger.info(f"Requesting mode {self._pending_label} ({mode_num}) …")
        self._send_mode_request()
        return True

    def set_fbwa_mode(self):
        return self.set_mode(MODE_FBWA, "FBWA")

    def set_acro_mode(self):
        return self.set_mode(MODE_ACRO, "ACRO")

    def set_rtl_mode(self):
        return self.set_mode(MODE_RTL, "RTL")

    def arm(self):
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        logger.info("Arm command sent")

    def send_rc_override(self, roll_norm, pitch_norm, throttle_norm=0.6, yaw_norm=0.0):
        """
        Simulate stick movement via RC_CHANNELS_OVERRIDE.

        In FBWA:  roll = desired bank angle,  pitch = desired pitch angle (stabilized)
        In ACRO:  roll = roll rate,            pitch = pitch rate (not stabilized)

        roll_norm:     -1.0 (full left) … +1.0 (full right)
        pitch_norm:    -1.0 (nose down) … +1.0 (nose up)
        throttle_norm:  0.0 … 1.0
        """
        ch1 = self._norm_to_pwm(roll_norm)
        ch2 = self._norm_to_pwm(-pitch_norm)  # invert: positive norm = nose up = lower PWM
        ch3 = int(RC_MIN + throttle_norm * (RC_MAX - RC_MIN))
        ch4 = self._norm_to_pwm(yaw_norm)

        self.mav.mav.rc_channels_override_send(
            self.mav.target_system,
            self.mav.target_component,
            ch1, ch2, ch3, ch4,
            0, 0, 0, 0
        )

    def release_rc_override(self):
        self.mav.mav.rc_channels_override_send(
            self.mav.target_system,
            self.mav.target_component,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        logger.info("RC override released")

    def get_attitude(self):
        """Newest cached (roll, pitch, yaw) in radians, or None. Fed by poll()."""
        return self._attitude

    def close(self):
        if self.mav:
            try:
                self.release_rc_override()
            except Exception:
                pass
            self.mav.close()

    @staticmethod
    def _norm_to_pwm(value):
        value = max(-1.0, min(1.0, value))
        return int(RC_MID + value * (RC_MAX - RC_MID))
