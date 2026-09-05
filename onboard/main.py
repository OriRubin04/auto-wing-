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
from onboard.flight_log import FlightLogger


def load_config(path='config/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


class TrackingSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.running = False
        self.tracking = False
        self._last_bank_rad = 0.0  # cached attitude for wing leveling

        cam_cfg = cfg['camera']
        cam_index = cam_cfg['index']
        self.cap, first_frame = self._open_camera(cam_index, cam_cfg)
        # Take the size from a real frame.  cap.get() can report the requested
        # mode rather than the delivered one, and if the two disagree the HUD
        # crosshair (drawn from frame.shape) and the control error (normalised
        # against these) would silently reference different centres.
        self.frame_h, self.frame_w = first_frame.shape[:2]
        logger.info(f"Camera opened: index={cam_index} resolution={self.frame_w}x{self.frame_h}")

        self.detector = ObjectDetector(cfg['detection'])
        self.controller = TrackingController(cfg['control'])

        mav_cfg = cfg['mavlink']
        self.mav = MAVLinkInterface(mav_cfg['connection_string'], mav_cfg.get('baudrate', 115200))

        stream_cfg = cfg['streaming']
        self.streamer = VideoStreamer(
            stream_cfg['host'], stream_cfg['port'],
            stream_cfg['quality'], stream_cfg['fps_limit'],
            max_width=stream_cfg.get('max_width', 640),
        )
        self.control_server = ControlServer(cfg['gcs']['control_port'])

        self.flight_log = FlightLogger(cfg.get('logging', {}))
        # Last command issued, so the log records what we actually sent even on
        # frames where the rate-limited control loop did not run.
        self._last_cmd = (0.0, 0.0, 0.0)
        self._last_err = (0.0, 0.0)

        self.update_interval = 1.0 / cfg['control']['update_rate_hz']

    def start(self):
        logger.info("Connecting to flight controller...")
        self.mav.connect()
        self.flight_log.start(
            self.frame_w, self.frame_h,
            fps=self.cfg['camera'].get('fps', 25),
            meta={'config': self.cfg},
        )
        logger.info("Ready. Waiting for GCS target selection.")
        self.running = True
        self._loop()

    @staticmethod
    def _open_camera(index, cam_cfg):
        """
        Open the camera and return (cap, first_frame).

        Settings are applied BEFORE the warm-up read, and the caller takes the
        real resolution from the returned frame rather than from cap.get(),
        which can report the requested mode rather than the delivered one.
        """
        if sys.platform.startswith('win'):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        want_w, want_h = cam_cfg['width'], cam_cfg['height']
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # MJPEG first: uncompressed YUYV at 720p is ~442 Mbps, well over
            # practical USB 2.0, and the driver silently drops to ~5 fps
            # instead of reporting an error.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            cap.set(cv2.CAP_PROP_FPS, cam_cfg['fps'])
            # Keep only the newest frame: a queued buffer adds pure latency
            # to a control loop.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for _ in range(5):                      # warm up
                cap.read()
                time.sleep(0.05)
            ret, frame = cap.read()
            if ret and frame is not None:
                got_h, got_w = frame.shape[:2]
                logger.info(f"Camera backend: {backend}  delivering {got_w}x{got_h}")
                if (got_w, got_h) != (want_w, want_h):
                    logger.warning(
                        f"Camera gave {got_w}x{got_h}, not the requested "
                        f"{want_w}x{want_h} - using what it actually delivers.")
                return cap, frame
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
                        self.cap, _ = self._open_camera(
                            self.cfg['camera']['index'], self.cfg['camera'])
                        cam_fail_count = 0
                        logger.info("Camera reopened successfully")
                    except RuntimeError as e:
                        logger.error(str(e))
                else:
                    time.sleep(0.05)
                continue
            cam_fail_count = 0
            loop_t0 = time.time()

            # Single MAVLink drain: refreshes the attitude cache and services
            # any outstanding mode request without ever blocking the loop.
            self.mav.poll()
            attitude = self.mav.get_attitude()
            if attitude is not None:
                self._last_bank_rad = float(attitude[0])

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
                    # Lost target while tracking in ACRO: hold level sticks, motor off
                    self.mav.send_rc_override(0.0, 0.0, self.ACRO_THROTTLE)
                    self._last_cmd = (0.0, 0.0, self.ACRO_THROTTLE)
                    self._last_err = (0.0, 0.0)

            # Log the RAW frame (no HUD): annotations can be regenerated from
            # flight.csv, but they cannot be removed from burned-in pixels.
            roll, pitch, thr = self._last_cmd
            err_x, err_y = self._last_err
            self.flight_log.log_frame(frame, {
                'state': state,
                'tracking': int(self.tracking),
                'target_cx': target[0] if target else '',
                'target_cy': target[1] if target else '',
                'target_w': target[2] if target else '',
                'target_h': target[3] if target else '',
                'err_x': f"{err_x:.4f}",
                'err_y': f"{err_y:.4f}",
                'roll_cmd': f"{roll:.4f}",
                'pitch_cmd': f"{pitch:.4f}",
                'throttle_cmd': f"{thr:.3f}",
                'bank_rad': f"{self._last_bank_rad:.4f}",
                'fail_count': self.detector.fail_count,
                'confidence': f"{self.detector.confidence():.3f}",
                'loop_ms': f"{(time.time() - loop_t0) * 1000:.2f}",
            })

    def _handle_gcs_message(self, msg, frame):
        action = msg.get('action')
        if action == 'select':
            # The GCS clicks in STREAM pixels, which may be a downscaled copy
            # of the frame we actually process.  Map back before using them.
            scale = self.streamer.last_scale or 1.0
            cx = int(msg.get('x', self.frame_w * scale // 2) / scale)
            cy = int(msg.get('y', self.frame_h * scale // 2) / scale)
            cx = max(0, min(cx, self.frame_w - 1))
            cy = max(0, min(cy, self.frame_h - 1))
            logger.info(f"GCS selected target at ({cx}, {cy}) [stream scale {scale:.3f}]")
            self.mav.set_acro_mode()
            ok = self.detector.select_target(cx, cy, frame)
            self.controller.reset()
            self.tracking = True
            self.flight_log.log_event('target_select', x=cx, y=cy, ok=bool(ok),
                                      bbox=self.detector.last_bbox)
        elif action == 'stop':
            logger.info("GCS stop command - switching to RTL")
            self.tracking = False
            self.detector.state = self.detector.STATE_SEARCHING
            self.mav.release_rc_override()
            self.mav.set_rtl_mode()
            self.flight_log.log_event('stop_rtl')

    LEVEL_KP = 0.3          # bank angle (rad) -> roll rate command
    LEVEL_DEADBAND = 0.05   # ignore bank angles smaller than ~3 deg
    # Throttle (0..1) commanded while tracking in ACRO mode.
    # 0.0 = motor off (gliding tracking).  RTL is unaffected: when we stop we
    # release the RC override entirely, so ArduPilot controls throttle in RTL.
    ACRO_THROTTLE = 0.0

    def _send_control(self, target):
        cx, cy, tw, th = target
        error_x = float((cx - self.frame_w / 2) / (self.frame_w / 2))
        error_y = float((cy - self.frame_h / 2) / (self.frame_h / 2))
        roll, pitch, _ = self.controller.compute(error_x, error_y)

        # ACRO tracking: motor off (0% throttle).
        throttle = self.ACRO_THROTTLE

        # Derate by how fresh the tracker position is.  While coasting, the box
        # is the last KNOWN position rather than a measurement, so authority
        # decays linearly with consecutive tracker failures instead of steering
        # at full strength on data that can be ~2 s old.
        conf = self.detector.confidence()
        roll *= conf
        pitch *= conf

        # When target is centered, use cached bank angle to level wings every
        # frame.  Deliberately NOT derated: this comes from a live ATTITUDE
        # reading rather than from tracker pixels, and levelling is the safe
        # thing to be doing when we are unsure where the target is.
        if abs(error_x) < self.controller.DEADBAND:
            if abs(self._last_bank_rad) > self.LEVEL_DEADBAND:
                roll = float(max(-0.5, min(0.5, -self.LEVEL_KP * self._last_bank_rad)))
            else:
                roll = 0.0

        self.mav.send_rc_override(roll, pitch, throttle)
        self._last_cmd = (roll, pitch, throttle)
        self._last_err = (error_x, error_y)
        logger.info(
            f"CMD  err_x={error_x:+.2f} err_y={error_y:+.2f} "
            f"bank={self._last_bank_rad:+.2f}rad conf={conf:.2f} | "
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
        self.flight_log.close()
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
