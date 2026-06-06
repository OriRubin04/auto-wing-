import time
import math
import logging
from pymavlink import mavutil

logger = logging.getLogger(__name__)

# RC channel PWM range
RC_MIN = 1000
RC_MID = 1500
RC_MAX = 2000

# ArduPlane mode numbers
MODE_ACRO    = 1   # Acro — rate control, direct RC input, no attitude stabilization
MODE_FBWA    = 5   # Fly By Wire A — stabilized, accepts RC input
MODE_GUIDED  = 15


class MAVLinkInterface:
    """
    pymavlink wrapper using RC_CHANNELS_OVERRIDE in FBWA mode.

    Why RC override instead of SET_ATTITUDE_TARGET:
    - Works reliably with any ArduPilot build / SITL
    - FBWA keeps the plane stable; we just push virtual sticks
    - SET_ATTITUDE_TARGET on ArduPlane requires specific GUIDED sub-modes
      that vary by firmware version and often silently do nothing
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

    def set_fbwa_mode(self):
        """Switch ArduPlane to ACRO mode — rate control, direct RC sticks."""
        logger.info(f"Sending ACRO mode command (sysid={self.mav.target_system})")
        self._set_mode(MODE_ACRO)
        time.sleep(1.0)
        mode = self._read_mode()
        if mode == MODE_ACRO:
            logger.info("Mode confirmed: ACRO ✓")
        else:
            logger.warning(f"Mode is {mode}, expected {MODE_ACRO} (ACRO) — set it manually in Mission Planner")

    def set_guided_mode(self):
        """Switch ArduPlane to GUIDED mode (kept for compatibility)."""
        self._set_mode(MODE_GUIDED)
        time.sleep(0.5)
        logger.info("GUIDED mode command sent")

    def _set_mode(self, mode_num):
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_num,
            0, 0, 0, 0, 0
        )

    def _read_mode(self):
        msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
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
        Send RC_CHANNELS_OVERRIDE — simulates stick movement.

        Args:
            roll_norm:     -1.0 (full left) to +1.0 (full right)
            pitch_norm:    -1.0 (full down) to +1.0 (full up)
            throttle_norm:  0.0 to 1.0
            yaw_norm:      -1.0 (full left) to +1.0 (full right)

        ArduPlane channel mapping (Mode 2):
            CH1 = Roll (aileron)
            CH2 = Pitch (elevator)   NOTE: 1000=up, 2000=down on most setups
            CH3 = Throttle
            CH4 = Yaw (rudder)
        """
        ch1 = self._norm_to_pwm(roll_norm)
        ch2 = self._norm_to_pwm(-pitch_norm)   # invert: positive pitch = nose up = lower PWM
        ch3 = int(RC_MIN + throttle_norm * (RC_MAX - RC_MIN))
        ch4 = self._norm_to_pwm(yaw_norm)

        # Channels 5-8: 0 = do not override
        self.mav.mav.rc_channels_override_send(
            self.mav.target_system,
            self.mav.target_component,
            ch1, ch2, ch3, ch4,
            0, 0, 0, 0
        )

    def release_rc_override(self):
        """Release all RC overrides — hand control back to pilot."""
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
        """Convert -1..+1 to PWM 1000..2000."""
        value = max(-1.0, min(1.0, value))
        return int(RC_MID + value * (RC_MAX - RC_MID))
