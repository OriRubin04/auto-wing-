import logging
import json
import select
import socket
import struct
import threading
import time

import cv2

logger = logging.getLogger(__name__)


class VideoStreamer:
    """
    Streams JPEG frames over UDP to the GCS, off the control loop.

    CLAUDE.md section 1: nothing in the tracking or control path may depend on
    the link, and streaming belongs off the hot path.  So `send_frame` does the
    cheapest possible thing - hand the newest frame to a sender thread - and
    the resize, the JPEG encode and the sendto all happen on that thread.

    Measured at 1280x720, x86: encode 0.68 ms, resize 0.09 ms, copy 0.21 ms.
    So this buys back about half a millisecond per streamed frame here, which
    is small; the reason to do it anyway is that the encode is the part that
    grows on the Pi's weaker NEON path, and that a stalled `sendto` can no
    longer reach the loop at all.  The loop now pays one memcpy and nothing
    else, whatever the network does.

    Only the newest frame is kept.  A queue would add latency to a picture
    whose only job is to be current enough to click on.
    """

    def __init__(self, host, port, quality=70, fps_limit=25, max_width=640):
        self.host = host
        self.port = port
        self.quality = quality
        self.min_interval = 1.0 / fps_limit
        # The GCS view only needs to be good enough to click on.  Processing
        # can run at full camera resolution while the link carries a small
        # frame.
        self.max_width = max_width
        # Scale applied to the last streamed frame.  The GCS clicks in STREAM
        # pixels, so the onboard side must divide by this to get frame pixels.
        #
        # Deliberately computed on the control loop in `send_frame`, not on the
        # sender thread: it depends only on the frame shape and max_width, both
        # known without resizing, and `_handle_gcs_message` reads it when a
        # click arrives.  Setting it from the thread would be a race on the one
        # value that decides where the operator actually pointed.
        self.last_scale = 1.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._gcs_addr = None
        self._last_send = 0.0

        # Listen for the GCS registration packet.
        self._reg_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._reg_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._reg_sock.bind(('0.0.0.0', port))
        # Truly non-blocking.  With a 10 ms timeout this cost ~10 ms of EVERY
        # frame once registration had happened, since the common case is that
        # no packet is waiting - about 30% of the frame budget spent waiting
        # for nothing, in a loop that must never block on the network.
        self._reg_sock.setblocking(False)

        self._pending = None
        self._pending_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sender, daemon=True)
        self._thread.start()

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

    # ------------------------------------------------------------------ #
    #  Control loop side - must stay cheap and must never block            #
    # ------------------------------------------------------------------ #

    def send_frame(self, frame):
        """Queue the newest frame for the sender thread.  Never blocks."""
        if self._gcs_addr is None:
            return
        now = time.time()
        if now - self._last_send < self.min_interval:
            return
        self._last_send = now

        width = frame.shape[1]
        self.last_scale = (self.max_width / float(width)
                           if self.max_width and width > self.max_width else 1.0)

        # The loop goes on drawing into its own buffer, and the camera may hand
        # back the same one, so the sender needs an array it owns.
        with self._pending_lock:
            self._pending = frame.copy()
        self._wake.set()

    # ------------------------------------------------------------------ #
    #  Sender thread side - allowed to be slow                            #
    # ------------------------------------------------------------------ #

    def _sender(self):
        while not self._stop.is_set():
            # Timed wait rather than a bare wait, so close() is noticed even
            # if no frame ever arrives.
            if not self._wake.wait(0.2):
                continue
            self._wake.clear()
            with self._pending_lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            try:
                self._encode_and_send(frame)
            except Exception as e:
                # A streaming failure is cosmetic by CLAUDE.md section 1 and
                # must never take the thread down with it.
                logger.debug(f"Stream send error: {e}")

    def _encode_and_send(self, frame):
        addr = self._gcs_addr
        if addr is None:
            return

        if self.max_width and frame.shape[1] > self.max_width:
            scale = self.max_width / float(frame.shape[1])
            frame = cv2.resize(frame,
                               (self.max_width, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode('.jpg', frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return

        data = buf.tobytes()
        total = len(data)
        # One 60 kB datagram per frame, IP-fragmented by the stack.
        #
        # Chunking this smaller looks like a robustness win and is not: a
        # ~30 kB JPEG is ~21 IP fragments at a 1500 byte MTU, or ~23 packets if
        # we chunk at 1400 ourselves, and a JPEG needs every byte either way.
        # One lost packet loses the frame in both schemes.  Fewer, larger
        # packets at least cost fewer syscalls.  Making the picture robust to
        # loss needs FEC or a lower quality setting, not a smaller chunk size.
        chunk_size = 60000
        header = struct.pack('>II', total, 0)
        self._sock.sendto(header + data[:chunk_size - len(header)], addr)
        sent = chunk_size - len(header)
        idx = 1
        while sent < total:
            chunk = data[sent:sent + chunk_size]
            self._sock.sendto(struct.pack('>II', total, idx) + chunk, addr)
            sent += len(chunk)
            idx += 1

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)
        self._sock.close()
        self._reg_sock.close()


class ControlServer:
    """
    TCP server receiving target selection from the GCS.

    This carries the one message CLAUDE.md section 1 says the link must
    deliver, so it has to keep working across a GCS restart or a network drop.

    The previous version read a single connection in a nested loop and only
    returned to accept() when recv raised or returned EOF.  A connection that
    went *silent* - no FIN, no RST, which is exactly what the aircraft sees
    when the operator's laptop loses Wi-Fi - left it spinning on recv timeouts
    forever, so a reconnecting GCS sat in the listen backlog and the operator
    could not select a target again for the rest of the flight.  Reproduced
    before fixing.

    A killed GCS process happened to recover, because the RST made recv raise.
    That is why the bug survived: the obvious way to test it passes.

    One select loop over the listener and every live client fixes both: new
    connections are accepted while old ones are still open, and a silent
    socket simply never becomes readable.
    """

    # One operator, so unbounded connections are a leak rather than a feature.
    MAX_CLIENTS = 4

    def __init__(self, port):
        self.port = port
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('0.0.0.0', port))
        self._server.listen(5)
        self._server.setblocking(False)
        self._pending = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Control server listening on port {port}")

    def _run(self):
        buffers = {}          # socket -> bytes received so far
        try:
            while not self._stop.is_set():
                watch = [self._server] + list(buffers)
                try:
                    readable, _, _ = select.select(watch, [], [], 0.5)
                except (OSError, ValueError):
                    # A socket was closed under us; drop the dead ones.
                    for sock in [s for s in buffers if s.fileno() == -1]:
                        buffers.pop(sock, None)
                    continue

                for sock in readable:
                    if sock is self._server:
                        self._accept(buffers)
                    elif not self._read(sock, buffers):
                        buffers.pop(sock, None)
                        self._shutdown(sock)
        finally:
            for sock in list(buffers):
                self._shutdown(sock)

    def _accept(self, buffers):
        try:
            conn, addr = self._server.accept()
        except (BlockingIOError, OSError):
            return
        conn.setblocking(False)
        buffers[conn] = b''
        logger.info(f"GCS control connected from {addr}")
        while len(buffers) > self.MAX_CLIENTS:
            oldest = next(iter(buffers))
            buffers.pop(oldest, None)
            logger.info("Too many GCS control connections - dropping the oldest")
            self._shutdown(oldest)

    def _read(self, sock, buffers):
        """Read and queue whole lines.  False when the client is finished."""
        try:
            data = sock.recv(4096)
        except BlockingIOError:
            return True
        except OSError as e:
            logger.debug(f"Control client dropped: {e}")
            return False
        if not data:
            logger.info("GCS control disconnected")
            return False

        buf = buffers.get(sock, b'') + data
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            if not line.strip():
                continue
            try:
                msg = json.loads(line.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            with self._lock:
                self._pending.append(msg)
        buffers[sock] = buf
        return True

    @staticmethod
    def _shutdown(sock):
        try:
            sock.close()
        except OSError:
            pass

    def get_messages(self):
        with self._lock:
            msgs = list(self._pending)
            self._pending.clear()
        return msgs

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._shutdown(self._server)
