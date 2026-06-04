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
        self.cap = cv2.VideoCapture(cam_cfg['index'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg['height'])
        self.cap.set(cv2.CAP_PROP_FPS, cam_cfg['fps'])
        self.frame_w = cam_cfg['width']
        self.frame_h = cam_cfg['height']

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
        self.mav.set_guided_mode()
        logger.info("Ready. Waiting for GCS target selection.")
        self.running = True
        self._loop()

    def _loop(self):
        last_update = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Camera frame read failed")
                time.sleep(0.05)
                continue

            # Check for GCS registration & stream frame
            self.streamer.check_registration()

            # Process control messages from GCS
            for msg in self.control_server.get_messages():
                self._handle_gcs_message(msg, frame)

            # Run detection
            result = self.detector.process(frame)
            target, state, annotated = result

            # Draw HUD
            self._draw_hud(annotated, target, state)
            self.streamer.send_frame(annotated)

            # Control loop at fixed rate
            now = time.time()
            if now - last_update >= self.update_interval:
                last_update = now
                if target is not None and self.tracking:
                    self._send_control(target)
                elif self.tracking and target is None:
                    # Lost target: hold wings level
                    self.mav.send_attitude_target(0.0, 0.0, throttle=self.cfg['control']['throttle'])

    def _handle_gcs_message(self, msg, frame):
        action = msg.get('action')
        if action == 'select':
            cx = msg.get('x', self.frame_w // 2)
            cy = msg.get('y', self.frame_h // 2)
            logger.info(f"GCS selected target at ({cx}, {cy})")
            self.detector.select_target(cx, cy, frame)
            self.controller.reset()
            self.tracking = True
        elif action == 'stop':
            logger.info("GCS stop command")
            self.tracking = False
            self.detector.state = self.detector.STATE_SEARCHING
            self.mav.send_attitude_target(0.0, 0.0, throttle=self.cfg['control']['throttle'])

    def _send_control(self, target):
        cx, cy, tw, th = target
        # Normalized error: -1 (left/up) to +1 (right/down)
        error_x = (cx - self.frame_w / 2) / (self.frame_w / 2)
        error_y = (cy - self.frame_h / 2) / (self.frame_h / 2)
        roll, pitch, throttle = self.controller.compute(error_x, error_y)
        self.mav.send_attitude_target(roll, pitch, throttle=throttle)
        logger.debug(f"err=({error_x:.2f},{error_y:.2f}) roll={roll:.3f} pitch={pitch:.3f}")

    def _draw_hud(self, frame, target, state):
        h, w = frame.shape[:2]
        # Crosshair at frame center
        cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 1)
        cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 1)
        # Status text
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
