#!/usr/bin/env python3
"""
Offline detector benchmark.

Runs the REAL ObjectDetector pipeline (proposal backend + CSRT + re-acquisition)
over a recorded clip, so backends can be compared like for like without flying.

Usage
-----
    # record once
    python tools/record.py --out recordings/flight1.avi

    # then compare backends over the same footage
    python tools/benchmark_detector.py --video recordings/flight1.avi --backend saliency
    python tools/benchmark_detector.py --video recordings/flight1.avi --backend yolo

    # pick the target without a GUI
    python tools/benchmark_detector.py --video f.avi --click 320,240

What it reports
---------------
  lock / coast / lost frames   how much of the clip the tracker held the target
  re-acquisition attempts+hits whether recovery after a CSRT failure works
  proposal call timing         cost of the backend itself
  annotated output video       for eyeballing what it actually latched onto

Note on rigour: without hand-labelled ground truth this measures whether the
pipeline *keeps a lock* and *recovers*, not whether it locked onto the RIGHT
object.  Watch the annotated output to confirm that part.
"""
import argparse
import csv
import os
import statistics
import sys
import time

import cv2
import yaml

sys.path.insert(0, '.')
from onboard.detector import ObjectDetector    # noqa: E402


class TimedProposer:
    """Wraps a proposal backend to count calls and measure latency."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.hits = 0
        self.times_ms = []

    def find_nearest(self, frame, ref_x, ref_y, max_dist=None):
        t0 = time.perf_counter()
        result = self.inner.find_nearest(frame, ref_x, ref_y, max_dist)
        self.times_ms.append((time.perf_counter() - t0) * 1000.0)
        self.calls += 1
        if result is not None:
            self.hits += 1
        return result


def pick_click_point(frame, provided):
    """--click x,y wins; otherwise try a GUI picker; otherwise frame centre."""
    h, w = frame.shape[:2]
    if provided:
        x, y = provided.split(',')
        return int(x), int(y)

    point = {}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            point['xy'] = (x, y)

    try:
        win = 'click the target, then any key'
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        while 'xy' not in point:
            shown = frame.copy()
            cv2.putText(shown, "Click the target", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(win, shown)
            if cv2.waitKey(20) & 0xFF != 255:
                break
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    if 'xy' in point:
        return point['xy']
    print("No click given and no GUI available - using frame centre.")
    return w // 2, h // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--backend', default='saliency', choices=['saliency', 'yolo'])
    ap.add_argument('--method', default=None, choices=['contrast', 'spectral'],
                    help='saliency map type (saliency backend only)')
    ap.add_argument('--click', default=None, help='X,Y of the target in the start frame')
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = whole clip')
    ap.add_argument('--out', default=None, help='annotated output video')
    ap.add_argument('--csv', default=None, help='per-frame state log')
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    det_cfg = dict(cfg.get('detection', {}) or {})
    det_cfg['backend'] = args.backend
    if args.method:
        det_cfg['saliency_method'] = args.method

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    for _ in range(args.start_frame):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read the start frame")

    click_x, click_y = pick_click_point(frame, args.click)
    print(f"Target click: ({click_x}, {click_y})  backend={args.backend}"
          + (f"/{args.method}" if args.method else ""))

    detector = ObjectDetector(det_cfg)
    detector.proposer = TimedProposer(detector.proposer)

    if not detector.select_target(click_x, click_y, frame):
        raise SystemExit("select_target failed - tracker could not initialise")
    select_calls = detector.proposer.calls        # calls used by selection itself

    writer = None
    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        suffix = args.backend + (f"-{args.method}" if args.method else "")
        os.makedirs('recordings', exist_ok=True)
        out_path = os.path.join('recordings', f"{base}_{suffix}_annotated.avi")

    rows = []
    n_frames = 0
    n_lock = 0
    n_coast = 0
    lost_at = None
    process_ms = []

    prev_state = detector.state
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1

        t0 = time.perf_counter()
        target, state, annotated = detector.process(frame)
        process_ms.append((time.perf_counter() - t0) * 1000.0)

        if target is None:
            if lost_at is None and prev_state != ObjectDetector.STATE_SEARCHING:
                lost_at = n_frames
        elif detector.fail_count == 0:
            n_lock += 1
        else:
            n_coast += 1
        prev_state = state

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                     cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
        writer.write(annotated)

        rows.append({
            'frame': n_frames,
            'state': state,
            'fail_count': detector.fail_count,
            'target_x': target[0] if target else '',
            'target_y': target[1] if target else '',
            'process_ms': f"{process_ms[-1]:.2f}",
        })

        if args.max_frames and n_frames >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    prop = detector.proposer
    reacq_calls = prop.calls - select_calls
    reacq_hits = prop.hits - (1 if select_calls and prop.hits else 0)
    reacq_hits = max(0, reacq_hits)

    def stat(vals):
        if not vals:
            return "n/a"
        return (f"mean {statistics.mean(vals):6.2f} ms   "
                f"median {statistics.median(vals):6.2f} ms   "
                f"max {max(vals):6.2f} ms")

    print("\n" + "=" * 62)
    print(f"  BENCHMARK  backend={args.backend}"
          + (f"/{args.method}" if args.method else ""))
    print("=" * 62)
    print(f"  frames processed      {n_frames}")
    if n_frames:
        print(f"  lock (tracking)       {n_lock:5d}  ({100.0*n_lock/n_frames:5.1f}%)")
        print(f"  coast (CSRT failing)  {n_coast:5d}  ({100.0*n_coast/n_frames:5.1f}%)")
    print(f"  target fully lost at  {lost_at if lost_at else 'never'}")
    print("-" * 62)
    print(f"  re-acquisition calls  {reacq_calls}")
    print(f"  re-acquisition hits   {reacq_hits}"
          + (f"  ({100.0*reacq_hits/reacq_calls:.0f}%)" if reacq_calls else ""))
    print("-" * 62)
    print(f"  proposal latency      {stat(prop.times_ms)}")
    print(f"  process() latency     {stat(process_ms)}")
    print("=" * 62)
    print(f"  annotated video  ->  {out_path}")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['frame'])
            w.writeheader()
            w.writerows(rows)
        print(f"  per-frame CSV    ->  {args.csv}")


if __name__ == '__main__':
    main()
