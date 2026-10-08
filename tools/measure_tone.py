#!/usr/bin/env python3
"""Per-frame tone statistics for a continued H3 shot.

Reports median luma plus shadow and highlight percentiles frame by frame, so a
tone-curve collapse (shadows falling while highlights hold) is separable from a
brightness shift (everything falling together).

With --window and --overlap it also prints the step across each window join.
That join is the sharp test: Wan2GP re-emits the previous window's carried
frames verbatim (pipeline.py, output_prefix), so the frames either side of it are
the same shot with exactly one generation pass in between.

    python measure_tone.py shot.mp4 --window 362 --overlap 18

Needs: pip install opencv-python numpy
"""

import argparse

import cv2
import numpy as np


def frame_stats(path):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"could not open {path}")
    rows = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        rows.append((np.percentile(luma, 10), np.median(luma), np.percentile(luma, 90)))
    capture.release()
    return np.asarray(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--window", type=int, help="sliding window size in frames")
    parser.add_argument("--overlap", type=int, default=18, help="sliding window overlap")
    parser.add_argument("--per-frame", action="store_true", help="dump every frame")
    args = parser.parse_args()

    stats = frame_stats(args.video)
    if not len(stats):
        raise SystemExit("no frames decoded")

    print(f"{len(stats)} frames    columns: p10 (shadows) / median / p90 (highlights)")
    if args.per_frame:
        for index, (low, mid, high) in enumerate(stats):
            print(f"  {index:5d}  {low:6.2f}  {mid:6.2f}  {high:6.2f}")

    first, last = stats[:24].mean(axis=0), stats[-24:].mean(axis=0)
    print("\nwhole shot (first 24 frames -> last 24 frames)")
    for label, index in (("p10", 0), ("median", 1), ("p90", 2)):
        change = (last[index] - first[index]) / max(first[index], 1e-6) * 100
        print(f"  {label:7s} {first[index]:6.2f} -> {last[index]:6.2f}   {change:+6.1f}%")

    if not args.window:
        return

    stride = args.window - args.overlap
    carried = max(args.overlap - 1, 0)
    print(f"\nwindow joins (stride {stride}, {carried} carried frames per window)")
    print("  the step below is one generation pass applied to the same content")
    boundary = carried
    while boundary + 8 < len(stats):
        before = stats[max(boundary - 8, 0):boundary].mean(axis=0)
        after = stats[boundary:boundary + 8].mean(axis=0)
        deltas = "  ".join(f"{label} {after[i] - before[i]:+6.2f}"
                           for i, label in enumerate(("p10", "median", "p90")))
        print(f"  frame {boundary:5d}   {deltas}")
        boundary += stride


if __name__ == "__main__":
    main()
