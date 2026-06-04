import time
import math
import logging
from pymavlink import mavutil

logger = logging.getLogger(__name__)


class MAVLinkInterface:
    """Wraps pymavlink for ArduPlane GUIDED mode attitude control."""

    def __init__(self, connection_string, baudrate=115200):
        self.conn_str = connection_string
        self.baudrate = baudrate
        self.mav = None
        self._last_heartbeat = 0

    def connect(self, timeout=30):
        logger.info(f"Connecting to {self.conn_str}")
        self.mav = mavutil.mavlink_connection(self.conn_str, baud=self.baudrate)
        self.mav.wait_heartbeat(timeout=timeout)
        logger.info(f"Heartbeat received from sysid={self.mav.target_system}")

    def set_guided_mode(self):
        """Switch ArduPlane to GUIDED mode."""
        # Try both methods — set_mode_apm works on some builds, command_long on others
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            15,  # GUIDED for ArduPlane
            0, 0, 0, 0, 0
        )
        time.sleep(0.3)
        self.mav.set_mode_apm(15)
        time.sleep(0.3)
        # Verify
        msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
        if msg:
            mode = msg.custom_mode
            logger.info(f"Mode after set: {mode} (15=GUIDED)")
        else:
            logger.warning("Could not verify mode change")
        logger.info("GUIDED mode command sent")

    def arm(self):
        self.mav.arducopter_arm()
        logger.info("Arm command sent")

    def takeoff(self, altitude_m):
        """Command takeoff to altitude in meters (MSL)."""
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0, altitude_m
        )
        logger.info(f"Takeoff command sent: {altitude_m}m")

    def send_attitude_target(self, roll_rad, pitch_rad, yaw_rate_rad=0.0, throttle=0.6):
        """
        Send SET_ATTITUDE_TARGET to control roll/pitch directly.
        type_mask: ignore yaw rate = 0b00000100 = 4, ignore everything except roll/pitch/throttle
        We command roll+pitch+thrust, ignore yaw (bit 2) and body rates (bits 0,1,2).
        type_mask bits: 0=rollrate, 1=pitchrate, 2=yawrate, 6=throttle
        To use roll/pitch angles + throttle: ignore rates (bits 0,1,2) = 0b00000111 = 7
        """
        # Quaternion from roll/pitch (yaw=0 relative)
        q = self._euler_to_quat(roll_rad, pitch_rad, 0.0)

        # type_mask: bit set = ignore. We send attitude+throttle, ignore body rates.
        type_mask = 0b00000111  # ignore roll/pitch/yaw RATES, use attitude + throttle

        self.mav.mav.set_attitude_target_send(
            int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
            self.mav.target_system,
            self.mav.target_component,
            type_mask,
            q,              # quaternion [w, x, y, z]
            0.0,            # roll rate (ignored)
            0.0,            # pitch rate (ignored)
            yaw_rate_rad,   # yaw rate
            throttle        # 0-1
        )

    def send_velocity_ned(self, vx, vy, vz):
        """Alternative: send velocity in NED frame (m/s)."""
        self.mav.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,  # velocity only
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    def get_attitude(self):
        msg = self.mav.recv_match(type='ATTITUDE', blocking=False)
        if msg:
            return msg.roll, msg.pitch, msg.yaw
        return None

    def close(self):
        if self.mav:
            self.mav.close()

    @staticmethod
    def _euler_to_quat(roll, pitch, yaw):
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [w, x, y, z]
