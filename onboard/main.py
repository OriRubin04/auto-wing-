#!/usr/bin/env python3
"""
Onboard tracking system for fixed-wing aircraft.
Run this on the Raspberry Pi / Jetson (or laptop for SITL).
"""
import sys
import time
import logging
import yaml
import cv2
import signal

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('main')

# Allow running from repo root
sys.path.insert(0, '.')
from onboard.detector import ObjectDetector
from onboard.controller import TrackingController
from onboard.mavlink_interface import MAVLinkInterface
from onboard.streamer import VideoStreamer, ControlServer


def load_config(path='config/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


class TrackingSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.running = False
        self.tracking = False

        cam_cfg = cfg['camera']
        cam_index = cam_cfg['index']
        self.cap = self._open_camera(cam_index, cam_cfg)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cam_cfg['width']
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cam_cfg['height']
        logger.info(f"Camera opened: index={cam_index} resolution={self.frame_w}x{self.frame_h}")

        self.detector = ObjectDetector(cfg['detection'])
        self.controller = TrackingController(cfg['control'])

        mav_cfg = cfg['mavlink']
        self.mav = MAVLinkInterface(mav_cfg['connection_string'], mav_cfg.get('baudrate', 115200))

        stream_cfg = cfg['streaming']
        self.streamer = VideoStreamer(
            stream_cfg['host'], stream_cfg['port'],
            stream_cfg['quality'], stream_cfg['fps_limit']
        )
        self.control_server = ControlServer(cfg['gcs']['control_port'])

        self.update_interval = 1.0 / cfg['control']['update_rate_hz']

    def start(self):
        logger.info("Connecting to flight controller...")
        self.mav.connect()
        logger.info("Ready. Waiting for GCS target selection.")
        self.running = True
        self._loop()

    @staticmethod
    def _open_camera(index, cam_cfg):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(5):
                cap.read()
                time.sleep(0.05)
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg['width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg['height'])
                cap.set(cv2.CAP_PROP_FPS, cam_cfg['fps'])
                logger.info(f"Camera backend: {backend}")
                return cap
            cap.release()
        raise RuntimeError(
            f"Cannot read from camera index {index}.\n"
            "Make sure no other app is using the webcam (Mission Planner, Teams, Zoom, browser).\n"
            "Try running: python tools\\camera_test.py"
        )

    def _loop(self):
        last_update = time.time()
        cam_fail_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                cam_fail_count += 1
                if cam_fail_count % 20 == 1:
                    logger.warning(f"Camera read failed ({cam_fail_count}x) - reopening...")
                    self.cap.release()
                    time.sleep(1.0)
                    try:
                        self.cap = self._open_camera(
                            self.cfg['camera']['index'], self.cfg['camera'])
                        cam_fail_count = 0
                        logger.info("Camera reopened successfully")
                    except RuntimeError as e:
                        logger.error(str(e))
                else:
                    time.sleep(0.05)
                continue
            cam_fail_count = 0

            self.streamer.check_registration()

            for msg in self.control_server.get_messages():
                self._handle_gcs_message(msg, frame)

            result = self.detector.process(frame)
            target, state, annotated = result

            self._draw_hud(annotated, target, state)
            self.streamer.send_frame(annotated)

            now = time.time()
            if now - last_update >= self.update_interval:
                last_update = now
                if target is not None and self.tracking:
                    self._send_control(target)
                elif self.tracking and target is None:
                    self.mav.send_rc_override(0.0, 0.0, self.cfg['control']['throttle'])

    def _handle_gcs_message(self, msg, frame):
        action = msg.get('action')
        if action == 'select':
            cx = msg.get('x', self.frame_w // 2)
            cy = msg.get('y', self.frame_h // 2)
            logger.info(f"GCS selected target at ({cx}, {cy})")
            self.mav.set_acro_mode()
            self.detector.select_target(cx, cy, frame)
            self.controller.reset()
            self.tracking = True
        elif action == 'stop':
            logger.info("GCS stop command - switching to RTL")
            self.tracking = False
            self.detector.state = self.detector.STATE_SEARCHING
            self.mav.release_rc_override()
            self.mav.set_rtl_mode()

    # How strongly to correct bank angle when target is centered (rad -> roll rate)
    # Lower = smoother but slower leveling. Tune up if wings level too slowly.
    LEVEL_KP = 0.3
    # Ignore bank angles smaller than this (radians ~3 deg) to avoid micro-corrections
    LEVEL_DEADBAND = 0.05

    def _send_control(self, target):
        cx, cy, tw, th = target
        error_x = float((cx - self.frame_w / 2) / (self.frame_w / 2))
        error_y = float((cy - self.frame_h / 2) / (self.frame_h / 2))
        roll, pitch, throttle = self.controller.compute(error_x, error_y)

        # When target is centered, level the wings using attitude feedback
        if abs(error_x) < self.controller.DEADBAND:
            attitude = self.mav.get_attitude()
            if attitude is not None:
                bank_rad = float(attitude[0])
                if abs(bank_rad) > self.LEVEL_DEADBAND:
                    roll = float(max(-0.5, min(0.5, -self.LEVEL_KP * bank_rad)))
                else:
                    roll = 0.0
            else:
                roll = 0.0

        self.mav.send_rc_override(roll, pitch, throttle)
        logger.info(
            f"CMD  err_x={error_x:+.2f} err_y={error_y:+.2f} | "
            f"roll={roll:+.2f}  pitch={pitch:+.2f}  thr={throttle:.2f}"
        )

    def _draw_hud(self, frame, target, state):
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 1)
        cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 1)
        status_color = (0, 255, 0) if self.tracking else (0, 0, 255)
        status = f"{'TRACKING' if self.tracking else 'STANDBY'} | {state.upper()}"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        if target:
            cx, cy = target[0], target[1]
            ex = (cx - w//2) / (w//2)
            ey = (cy - h//2) / (h//2)
            cv2.putText(frame, f"err x:{ex:+.2f} y:{ey:+.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def stop(self):
        self.running = False
        self.cap.release()
        self.streamer.close()
        self.control_server.close()
        self.mav.close()


def main():
    cfg = load_config()
    system = TrackingSystem(cfg)

    def _sig(s, f):
        logger.info("Shutting down...")
        system.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    system.start()


if __name__ == '__main__':
    main()
