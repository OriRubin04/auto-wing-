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


class MAVLinkInterface:
    """
    pymavlink wrapper using RC_CHANNELS_OVERRIDE.

    Use FBWA (mode 5) for visual tracking — it's stabilized so the plane
    holds attitude between commands and won't diverge.  ACRO (mode 4) is
    available but requires a much tighter control loop.
    """

    def __init__(self, connection_string, baudrate=115200):
        self.conn_str = connection_string
        self.baudrate = baudrate
        self.mav = None

    def connect(self, timeout=30):
        logger.info(f"Connecting to {self.conn_str}")
        self.mav = mavutil.mavlink_connection(self.conn_str, baud=self.baudrate)
        self.mav.wait_heartbeat(timeout=timeout)
        logger.info(
            f"Heartbeat from sysid={self.mav.target_system} "
            f"compid={self.mav.target_component}"
        )

    def set_mode(self, mode_num, mode_name=""):
        """Send mode command and wait for confirmation (retries up to 8×)."""
        label = mode_name or str(mode_num)
        logger.info(f"Requesting mode {label} ({mode_num}) …")

        for attempt in range(8):
            # Two message types — some ArduPlane builds accept one but not the other
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_num, 0, 0, 0, 0, 0
            )
            self.mav.mav.set_mode_send(
                self.mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_num
            )
            # Drain heartbeats until we see the mode confirmed or time out
            deadline = time.time() + 1.5
            while time.time() < deadline:
                msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=0.3)
                if msg and msg.custom_mode == mode_num:
                    logger.info(f"Mode {label} confirmed ✓")
                    return True
            logger.debug(f"Mode attempt {attempt + 1}/8 …")

        current = self._read_mode()
        logger.warning(
            f"Could not switch to {label} (current mode={current}).\n"
            f"  In Mission Planner: click the mode box (top-left) → select {label}\n"
            f"  RC override still works in whatever mode the plane is in."
        )
        return False

    def set_fbwa_mode(self):
        return self.set_mode(MODE_FBWA, "FBWA")

    def set_acro_mode(self):
        return self.set_mode(MODE_ACRO, "ACRO")

    def _set_mode(self, mode_num):
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_num, 0, 0, 0, 0, 0
        )

    def _read_mode(self):
        msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
        return msg.custom_mode if msg else '?'

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
        msg = self.mav.recv_match(type='ATTITUDE', blocking=False)
        if msg:
            return msg.roll, msg.pitch, msg.yaw
        return None

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
