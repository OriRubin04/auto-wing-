#!/usr/bin/env python3
"""
Record camera footage to disk for offline detector benchmarking.

You cannot iterate on a detector by flying repeatedly.  Record once, then run
tools/benchmark_detector.py over the same clip as many times as you like, with
different backends and settings, and compare like for like.

Usage
-----
    python tools/record.py                          # defaults from config.yaml
    python tools/record.py --out flight1.avi
    python tools/record.py --index 1 --width 1280 --height 720
    python tools/record.py --duration 60            # stop automatically

Press 'q' in the preview window (or Ctrl-C) to stop.
"""
import argparse
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, '.')


def open_camera(index, width, height, fps):
    """Try platform-appropriate backends; return the first that yields a frame."""
    if sys.platform.startswith('win'):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # MJPEG: uncompressed YUYV at 720p+ exceeds USB 2.0 bandwidth and the
        # driver silently drops to ~5 fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(5):                      # warm up
            cap.read()
            time.sleep(0.05)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"Camera opened (backend={backend}) actual={frame.shape[1]}x{frame.shape[0]}")
            return cap, frame
        cap.release()
    raise RuntimeError(
        f"Cannot read from camera index {index}. "
        "Close any other app using it (Mission Planner, Teams, Zoom, browser)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--out', default=None, help='output video path (default: recordings/<timestamp>.avi)')
    ap.add_argument('--index', type=int, default=None)
    ap.add_argument('--width', type=int, default=None)
    ap.add_argument('--height', type=int, default=None)
    ap.add_argument('--fps', type=int, default=None)
    ap.add_argument('--duration', type=float, default=0.0, help='seconds; 0 = until q/Ctrl-C')
    ap.add_argument('--no-preview', action='store_true')
    args = ap.parse_args()

    cam = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cam = (yaml.safe_load(f) or {}).get('camera', {}) or {}

    index = args.index if args.index is not None else cam.get('index', 0)
    width = args.width if args.width is not None else cam.get('width', 640)
    height = args.height if args.height is not None else cam.get('height', 480)
    fps = args.fps if args.fps is not None else cam.get('fps', 30)

    out_path = args.out
    if out_path is None:
        os.makedirs('recordings', exist_ok=True)
        out_path = os.path.join('recordings', time.strftime('%Y%m%d_%H%M%S') + '.avi')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    cap, first = open_camera(index, width, height, fps)
    h, w = first.shape[:2]

    # MJPG in an AVI container: no external codec needed on Windows or Linux.
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for {out_path}")

    print(f"Recording to {out_path}  ({w}x{h} @ {fps}fps)")
    print("Press 'q' in the preview window to stop." if not args.no_preview else "Ctrl-C to stop.")

    n = 0
    start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, stopping.")
                break
            writer.write(frame)
            n += 1

            if not args.no_preview:
                preview = frame.copy()
                cv2.putText(preview, f"REC {n} frames  {time.time()-start:.1f}s",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('recording (q to stop)', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.duration and (time.time() - start) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"\nSaved {n} frames in {elapsed:.1f}s "
          f"({n/elapsed:.1f} fps actual) -> {out_path}")
    print(f"Benchmark it with:\n"
          f"  python tools/benchmark_detector.py --video {out_path} --backend saliency")


if __name__ == '__main__':
    main()
