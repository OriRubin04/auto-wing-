#!/usr/bin/env python3
"""
GCS Client — runs on your laptop.
Displays live video stream from onboard, click to select tracking target.
Press 'q' to quit, 's' to stop tracking, 'r' to reset.
"""
import sys
import socket
import struct
import threading
import json
import time
import logging
import numpy as np
import cv2
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('gcs')


def load_config(path='config/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


class GCSClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.onboard_ip = cfg['gcs']['onboard_ip']
        self.video_port = cfg['streaming']['port']
        self.control_port = cfg['gcs']['control_port']

        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        # Control TCP connection
        self.ctrl_sock = None
        self._connect_control()

        # Video receive socket (bind before starting thread)
        self._vid_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._vid_sock.bind(('0.0.0.0', self.video_port + 100))
        self._vid_sock.settimeout(1.0)

        # Video UDP receiver thread
        self._video_thread = threading.Thread(target=self._recv_video, daemon=True)
        self._video_thread.start()

        # Register with onboard streamer
        self._register_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._reg_thread = threading.Thread(target=self._keep_registered, daemon=True)
        self._reg_thread.start()

    def _connect_control(self):
        try:
            self.ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ctrl_sock.connect((self.onboard_ip, self.control_port))
            logger.info(f"Control connected to {self.onboard_ip}:{self.control_port}")
        except Exception as e:
            logger.warning(f"Control connection failed: {e}. Will retry.")
            self.ctrl_sock = None

    def _keep_registered(self):
        while self.running:
            try:
                self._register_sock.sendto(b'REGISTER', (self.onboard_ip, self.video_port))
            except Exception:
                pass
            time.sleep(2.0)

    def _recv_video(self):
        """Receive fragmented JPEG UDP packets and reassemble."""
        buffers = {}
        while self.running:
            try:
                data, _ = self._vid_sock.recvfrom(65536)
                if len(data) < 8:
                    continue
                total_size, idx = struct.unpack('>II', data[:8])
                payload = data[8:]

                if total_size not in buffers:
                    buffers[total_size] = {0: payload} if idx == 0 else {}
                buffers[total_size][idx] = payload

                # Check if complete
                assembled = b''.join(buffers[total_size][i]
                                     for i in sorted(buffers[total_size]))
                if len(assembled) >= total_size:
                    img_array = np.frombuffer(assembled[:total_size], dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.frame_lock:
                            self.frame = frame
                    del buffers[total_size]
                    # Keep buffer dict small
                    if len(buffers) > 5:
                        oldest = min(buffers.keys())
                        del buffers[oldest]
            except socket.timeout:
                pass
            except Exception as e:
                logger.debug(f"Video recv error: {e}")

    def send_command(self, msg_dict):
        if self.ctrl_sock is None:
            self._connect_control()
        if self.ctrl_sock:
            try:
                data = json.dumps(msg_dict).encode() + b'\n'
                self.ctrl_sock.sendall(data)
            except Exception as e:
                logger.warning(f"Send failed: {e}")
                self.ctrl_sock = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Target selected at ({x}, {y})")
            self.send_command({'action': 'select', 'x': x, 'y': y})
            # Draw click indicator on local frame
            with self.frame_lock:
                if self.frame is not None:
                    cv2.circle(self.frame, (x, y), 15, (0, 255, 255), 2)

    def run(self):
        cv2.namedWindow('GCS - Auto Wing', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('GCS - Auto Wing', self.on_mouse)

        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Waiting for video stream...', (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.putText(placeholder, 'Click to select target | Q=quit | S=stop | R=reset',
                    (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        while self.running:
            with self.frame_lock:
                display = self.frame.copy() if self.frame is not None else placeholder.copy()

            # Instructions overlay
            cv2.putText(display, 'Click target | Q=quit | S=stop',
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow('GCS - Auto Wing', display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                logger.info("Sending stop command")
                self.send_command({'action': 'stop'})
            elif key == ord('r'):
                logger.info("Sending reset command")
                self.send_command({'action': 'stop'})

        self.running = False
        cv2.destroyAllWindows()


def main():
    cfg = load_config()
    client = GCSClient(cfg)
    client.run()


if __name__ == '__main__':
    main()
