import socket
import struct
import cv2
import threading
import json
import logging

logger = logging.getLogger(__name__)


class VideoStreamer:
    """Streams JPEG frames over UDP to GCS."""

    def __init__(self, host, port, quality=70, fps_limit=25):
        self.host = host
        self.port = port
        self.quality = quality
        self.min_interval = 1.0 / fps_limit
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._gcs_addr = None
        self._last_send = 0
        # Listen for GCS registration packet
        self._reg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._reg_sock.bind(('0.0.0.0', port))
        self._reg_sock.settimeout(0.01)

    def check_registration(self):
        """Non-blocking: check if GCS has registered."""
        try:
            data, addr = self._reg_sock.recvfrom(64)
            if data == b'REGISTER':
                self._gcs_addr = (addr[0], self.port + 100)  # send stream to port+100
                logger.info(f"GCS registered from {addr[0]}")
        except socket.timeout:
            pass

    def send_frame(self, frame):
        import time
        now = time.time()
        if now - self._last_send < self.min_interval:
            return
        if self._gcs_addr is None:
            return
        self._last_send = now

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        ok, buf = cv2.imencode('.jpg', frame, encode_param)
        if not ok:
            return

        data = buf.tobytes()
        # Send in chunks with size header
        chunk_size = 60000
        total = len(data)
        header = struct.pack('>II', total, 0)
        try:
            self._sock.sendto(header + data[:chunk_size - len(header)], self._gcs_addr)
            sent = chunk_size - len(header)
            idx = 1
            while sent < total:
                chunk = data[sent:sent + chunk_size]
                self._sock.sendto(struct.pack('>II', total, idx) + chunk, self._gcs_addr)
                sent += len(chunk)
                idx += 1
        except Exception as e:
            logger.debug(f"Stream send error: {e}")

    def close(self):
        self._sock.close()
        self._reg_sock.close()


class ControlServer:
    """TCP server receiving target selection from GCS."""

    def __init__(self, port):
        self.port = port
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('0.0.0.0', port))
        self._server.listen(1)
        self._server.settimeout(0.1)
        self._client = None
        self._pending = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Control server listening on port {port}")

    def _run(self):
        while True:
            try:
                conn, addr = self._server.accept()
                logger.info(f"GCS control connected from {addr}")
                self._client = conn
                conn.settimeout(1.0)
                buf = b''
                while True:
                    try:
                        data = conn.recv(1024)
                        if not data:
                            break
                        buf += data
                        while b'\n' in buf:
                            line, buf = buf.split(b'\n', 1)
                            try:
                                msg = json.loads(line.decode())
                                with self._lock:
                                    self._pending.append(msg)
                            except json.JSONDecodeError:
                                pass
                    except socket.timeout:
                        pass
            except socket.timeout:
                pass
            except Exception as e:
                logger.debug(f"Control server error: {e}")

    def get_messages(self):
        with self._lock:
            msgs = list(self._pending)
            self._pending.clear()
        return msgs

    def close(self):
        self._server.close()
