import socket
import struct
import cv2
import threading
import json
import logging

logger = logging.getLogger(__name__)


class VideoStreamer:
    """Streams JPEG frames over UDP to GCS."""

    def __init__(self, host, port, quality=70, fps_limit=25, max_width=640):
        self.host = host
        self.port = port
        self.quality = quality
        self.min_interval = 1.0 / fps_limit
        # The GCS view only needs to be good enough to click on.  Processing
        # can run at full camera resolution while the link carries a small
        # frame, which keeps JPEG encoding out of the control loop's budget.
        self.max_width = max_width
        # Scale applied to the last streamed frame.  The GCS clicks in STREAM
        # pixels, so the onboard side must divide by this to get frame pixels.
        self.last_scale = 1.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._gcs_addr = None
        self._last_send = 0
        # Listen for GCS registration packet
        self._reg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._reg_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._reg_sock.bind(('0.0.0.0', port))
        # Truly non-blocking.  With a 10 ms timeout this cost ~10 ms of EVERY
        # frame once registration had happened, since the common case is that
        # no packet is waiting - about 30% of the frame budget spent waiting
        # for nothing, in a loop that must never block on the network.
        self._reg_sock.setblocking(False)

    def check_registration(self):
        """Non-blocking: check if GCS has registered."""
        try:
            data, addr = self._reg_sock.recvfrom(64)
            if data == b'REGISTER':
                new_addr = (addr[0], self.port + 100)
                if new_addr != self._gcs_addr:
                    self._gcs_addr = new_addr
                    logger.info(f"GCS registered from {addr[0]}")
        except (BlockingIOError, socket.timeout):
            pass
        except OSError:
            # Windows raises WSAECONNRESET on UDP when a previous send was
            # refused.  Not fatal, and never worth stalling the loop over.
            pass

    def send_frame(self, frame):
        import time
        now = time.time()
        if now - self._last_send < self.min_interval:
            return
        if self._gcs_addr is None:
            return
        self._last_send = now

        # Downscale for the link only.  Encoding a 640-wide frame costs about
        # half what a 720p frame costs, and the operator only needs enough
        # picture to click on.
        if self.max_width and frame.shape[1] > self.max_width:
            self.last_scale = self.max_width / float(frame.shape[1])
            frame = cv2.resize(frame,
                               (self.max_width, int(frame.shape[0] * self.last_scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            self.last_scale = 1.0

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
