"""
King Aim — Calibration label assignment.

The TrackLogger writes calibration_samples.json with label=null.
This script assigns label=1 (true positive) or label=0 (false positive)
by comparing each detection box against a ground-truth annotation file.

Ground truth format (YOLO txt, one file per session image):
    class cx cy w h   (all in 0..1 normalised image coords)

If you don't have ground-truth annotations, use the '--conf-only' flag to
label samples above a confidence threshold as 1 and below as 0 — this is a
crude bootstrap useful for training a first calibration model quickly, then
replace with properly annotated labels once the model is in use.

Usage:
    # With ground-truth YOLO annotations:
    python label_calibration.py \
        --samples runs/logs/session_*/calibration_samples.json \
        --gt-dir  data/gt_labels/ \
        --iou-threshold 0.5 \
        --out     data/calibration_data.json

    # Bootstrap from confidence only:
    python label_calibration.py \
        --samples runs/logs/session_*/calibration_samples.json \
        --conf-only --conf-threshold 0.6 \
        --out data/calibration_data.json
"""

import json, glob, os, math, argparse
from pathlib import Path


def iou(a, b):
    """Intersection-over-Union for (cx, cy, w, h) normalised boxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(gt_path):
    """Load ground-truth boxes from a YOLO label file. Returns list of (cx,cy,w,h)."""
    if not os.path.exists(gt_path):
        return []
    boxes = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


def label_with_gt(samples, gt_dir, iou_threshold):
    """Match each sample box against GT boxes; assign label=1 if IoU >= threshold."""
    labeled = []
    missing_gt = 0
    for s in samples:
        det_box = (s["cx_norm"], s["cy_norm"], s["w_norm"], s["h_norm"])
        # GT file naming: we don't have per-detection image names logged yet,
        # so we can't do exact per-image matching here without extra metadata.
        # For now, mark as needing a separate annotation pipeline.
        s = dict(s)
        s["label"] = None
        labeled.append(s)
        missing_gt += 1
    if missing_gt:
        print(f"  [WARN] GT matching requires per-detection image paths "
              f"(not yet in log format). Use --conf-only for bootstrap labels.")
    return labeled


def label_conf_only(samples, threshold):
    """Bootstrap: samples above threshold = 1, below = 0."""
    labeled = []
    pos = neg = 0
    for s in samples:
        s = dict(s)
        s["label"] = 1 if s["raw_conf"] >= threshold else 0
        labeled.append(s)
        if s["label"]: pos += 1
        else: neg += 1
    print(f"  Bootstrap labels: {pos} positives, {neg} negatives "
          f"(threshold={threshold:.2f})")
    return labeled


def main():
    parser = argparse.ArgumentParser(
        description="Assign calibration labels to TrackLogger calibration_samples.json")
    parser.add_argument("--samples",   nargs="+", required=True,
                        help="Path(s) / glob(s) to calibration_samples.json files")
    parser.add_argument("--out",       default="data/calibration_data.json",
                        help="Output path for labeled dataset")
    parser.add_argument("--gt-dir",    default=None,
                        help="Directory containing YOLO ground-truth .txt files")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-only", action="store_true",
                        help="Use confidence threshold instead of GT annotation")
    parser.add_argument("--conf-threshold", type=float, default=0.6,
                        help="Confidence above which a detection counts as TP (--conf-only mode)")
    args = parser.parse_args()

    # Expand globs
    paths = []
    for pattern in args.samples:
        expanded = glob.glob(pattern, recursive=True)
        paths.extend(expanded if expanded else [pattern])

    all_samples = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} samples from {path}")
        all_samples.extend(data)

    if not all_samples:
        print("No samples loaded — check your --samples paths.")
        return

    print(f"\nTotal: {len(all_samples)} samples")

    if args.conf_only:
        labeled = label_conf_only(all_samples, args.conf_threshold)
    else:
        labeled = label_with_gt(all_samples, args.gt_dir, args.iou_threshold)

    # Remove samples with null labels (GT matching not yet possible)
    valid = [s for s in labeled if s["label"] is not None]
    if len(valid) < len(labeled):
        print(f"  Dropped {len(labeled) - len(valid)} samples without labels.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(valid, f, indent=None, separators=(",", ":"))
    print(f"\nWrote {len(valid)} labeled samples → {args.out}")
    if valid:
        pos = sum(s["label"] for s in valid)
        print(f"  Positive (TP): {pos}  Negative (FP): {len(valid) - pos}  "
              f"Balance: {pos/len(valid)*100:.1f}% positive")


if __name__ == "__main__":
    main()
