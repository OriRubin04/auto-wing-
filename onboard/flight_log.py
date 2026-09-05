"""
Per-flight logging: raw video + per-frame telemetry + discrete events.

Every run writes a self-contained session directory:

    logs/20260905_143022/
        video.avi      raw camera frames, UNannotated
        flight.csv     one row per frame: target, errors, commands, attitude, timing
        events.jsonl   discrete events: target select, mode change, re-acquire, loss
        meta.json      config snapshot, resolution, versions, git commit

Why raw (unannotated) video
---------------------------
Annotations can always be regenerated from flight.csv, but they cannot be
removed once burned into the pixels.  Clean frames are what you need to
re-run a different detector offline, and what you need as training data.

Free training labels
--------------------
flight.csv records the tracker's box for every frame.  Frames + boxes is a
labelled dataset - noisy, since CSRT drifts, but it is real footage of real
targets from a real airframe, which is the expensive part to obtain.

Flight safety
-------------
Logging must never be able to stall the control loop.  Video frames go to a
background thread through a small bounded queue; when the queue is full the
frame is DROPPED and counted, never blocked on.  A dropped log frame is
harmless, a stalled control loop is not.
"""
import csv
import json
import logging
import os
import platform
import queue
import subprocess
import threading
import time

import cv2

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    'frame', 't', 'state', 'tracking',
    'target_cx', 'target_cy', 'target_w', 'target_h',
    'err_x', 'err_y',
    'roll_cmd', 'pitch_cmd', 'throttle_cmd',
    'bank_rad', 'fail_count', 'confidence', 'lost_s', 'loop_ms',
]


class FlightLogger:
    """Records one flight session into logs/<timestamp>/."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get('enabled', True))
        self.base_dir = cfg.get('dir', 'logs')
        self.write_video = bool(cfg.get('video', True))
        # Record every Nth frame; 1 = all.  Raise to save disk on long flights.
        self.video_stride = max(1, int(cfg.get('video_stride', 1)))
        # Hard cap so a long flight cannot fill the disk mid-air.
        self.max_video_mb = float(cfg.get('max_video_mb', 2048))

        self.session_dir = None
        self._writer = None
        self._csv_file = None
        self._csv = None
        self._events = None
        self._q = None
        self._thread = None
        self._stop = threading.Event()

        self._frame_idx = 0
        self._rows_since_flush = 0
        self._t0 = None
        self.dropped = 0
        self._video_bytes_cap_hit = False

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self, frame_w, frame_h, fps=25, meta=None):
        if not self.enabled:
            logger.info("Flight logging disabled")
            return None
        try:
            name = time.strftime('%Y%m%d_%H%M%S')
            self.session_dir = os.path.join(self.base_dir, name)
            os.makedirs(self.session_dir, exist_ok=True)
            self._t0 = time.time()

            if self.write_video:
                path = os.path.join(self.session_dir, 'video.avi')
                # MJPG in AVI: no external codec needed on Windows or Linux.
                self._writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*'MJPG'),
                    max(1.0, fps / self.video_stride), (frame_w, frame_h))
                if not self._writer.isOpened():
                    logger.warning("Could not open video writer - logging CSV only")
                    self._writer = None
                else:
                    self._q = queue.Queue(maxsize=8)
                    self._thread = threading.Thread(target=self._drain, daemon=True)
                    self._thread.start()

            self._csv_file = open(os.path.join(self.session_dir, 'flight.csv'),
                                  'w', newline='')
            self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
            self._csv.writeheader()
            self._events = open(os.path.join(self.session_dir, 'events.jsonl'), 'a')

            self._write_meta(frame_w, frame_h, fps, meta or {})

            mb_min = frame_w * frame_h * 0.1 * fps / self.video_stride * 60 / 1e6
            logger.info(f"Flight log: {self.session_dir}  (video ~{mb_min:.0f} MB/min)")
            self.log_event('session_start', width=frame_w, height=frame_h, fps=fps)
            return self.session_dir
        except Exception as e:
            logger.error(f"Could not start flight log ({e}) - continuing without it")
            self.enabled = False
            return None

    def close(self):
        if not self.enabled:
            return
        try:
            self.log_event('session_end', frames=self._frame_idx, dropped=self.dropped)
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            if self._writer is not None:
                self._writer.release()
            for f in (self._csv_file, self._events):
                if f is not None:
                    f.close()
            if self.session_dir:
                logger.info(f"Flight log closed: {self.session_dir} "
                            f"({self._frame_idx} frames, {self.dropped} video frames dropped)")
        except Exception as e:
            logger.error(f"Error closing flight log: {e}")

    # ------------------------------------------------------------------ #
    #  Recording                                                           #
    # ------------------------------------------------------------------ #

    def log_frame(self, frame, record):
        """Queue a raw frame for writing and append one telemetry row."""
        if not self.enabled:
            return
        try:
            self._frame_idx += 1

            if (self._writer is not None
                    and self._frame_idx % self.video_stride == 0
                    and not self._video_bytes_cap_hit):
                try:
                    self._q.put_nowait(frame)
                except queue.Full:
                    # Control loop takes priority: drop, never block.
                    self.dropped += 1

            row = {k: '' for k in CSV_FIELDS}
            row['frame'] = self._frame_idx
            row['t'] = f"{time.time() - self._t0:.3f}"
            for k, v in record.items():
                if k in row:
                    row[k] = v
            self._csv.writerow(row)

            # Flush periodically so a crash keeps almost everything.
            self._rows_since_flush += 1
            if self._rows_since_flush >= 25:
                self._csv_file.flush()
                self._rows_since_flush = 0
        except Exception as e:
            logger.debug(f"log_frame error: {e}")

    def log_event(self, kind, **data):
        if not self.enabled or self._events is None:
            return
        try:
            data['kind'] = kind
            data['t'] = round(time.time() - self._t0, 3) if self._t0 else 0.0
            data['wall'] = time.strftime('%H:%M:%S')
            self._events.write(json.dumps(data) + '\n')
            self._events.flush()
        except Exception as e:
            logger.debug(f"log_event error: {e}")

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _drain(self):
        """Background video writer."""
        max_bytes = self.max_video_mb * 1e6
        written = 0
        path = os.path.join(self.session_dir, 'video.avi')
        while not self._stop.is_set() or not self._q.empty():
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._writer.write(frame)
                # Cheap size check, not every frame.
                if self._frame_idx % 100 == 0 and os.path.exists(path):
                    written = os.path.getsize(path)
                    if written > max_bytes:
                        self._video_bytes_cap_hit = True
                        logger.warning(
                            f"Video log hit {self.max_video_mb:.0f} MB cap - "
                            "continuing with CSV only")
            except Exception as e:
                logger.debug(f"video write error: {e}")

    def _write_meta(self, w, h, fps, extra):
        meta = {
            'started': time.strftime('%Y-%m-%d %H:%M:%S'),
            'width': w, 'height': h, 'fps': fps,
            'video_stride': self.video_stride,
            'opencv': cv2.__version__,
            'python': platform.python_version(),
            'platform': platform.platform(),
            'git_commit': self._git_commit(),
        }
        meta.update(extra)
        with open(os.path.join(self.session_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    @staticmethod
    def _git_commit():
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        except Exception:
            return 'unknown'
