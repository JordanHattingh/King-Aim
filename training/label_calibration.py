"""
King Aim — calibration label assignment (detection-context-v2).

TrackLogger v2 records frame_id and detection_index. Ground-truth matching is
performed per SESSION + FRAME using one-to-one maximum-IoU assignment so duplicate
detections on one object cannot all become true positives.

Expected ground-truth layouts (YOLO txt):
    <gt-dir>/<session_id>/frame_<frame_id>.txt     preferred
    <gt-dir>/frame_<frame_id>.txt                  fallback for a single session

Each GT row:
    class cx cy w h

Bootstrap confidence-only labels are intentionally marked as bootstrap data and
must not be used as the final calibration evaluation set.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(gt_path: Path) -> list[tuple[float, float, float, float]]:
    if not gt_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    with gt_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"Malformed YOLO label {gt_path}:{line_number}")
            boxes.append(tuple(float(value) for value in parts[1:5]))
    return boxes


def resolve_gt_path(gt_dir: Path, session_id: str, frame_id: int) -> Path:
    preferred = gt_dir / session_id / f"frame_{frame_id}.txt"
    if preferred.exists():
        return preferred
    return gt_dir / f"frame_{frame_id}.txt"


def _read_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL {path}:{line_number}: {exc}") from exc
        return records

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def load_samples(paths: list[str]) -> list[dict]:
    samples: list[dict] = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            print(f"  [WARN] Not found: {path}")
            continue
        data = _read_records(path)
        session_id = path.parent.name
        for sample in data:
            record = dict(sample)
            record.setdefault("session_id", session_id)
            samples.append(record)
        print(f"  Loaded {len(data)} samples from {path} as session {session_id}")
    return samples


def label_with_gt(samples: list[dict], gt_dir: Path, iou_threshold: float) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for sample in samples:
        if "frame_id" not in sample or int(sample["frame_id"]) < 0:
            raise ValueError(
                "Calibration samples need frame_id from TrackLogger schema v2. "
                "Discard old feature-schema logs and collect new data."
            )
        grouped[(str(sample["session_id"]), int(sample["frame_id"]))].append(sample)

    labeled: list[dict] = []
    missing_gt_frames = 0
    for (session_id, frame_id), frame_samples in sorted(grouped.items()):
        gt_path = resolve_gt_path(gt_dir, session_id, frame_id)
        if not gt_path.exists():
            missing_gt_frames += 1
            continue
        gt_boxes = load_gt_boxes(gt_path)

        # Build all viable detection/GT pairs and greedily accept highest IoU with
        # one-to-one constraints. This prevents duplicate detections from all being TP.
        candidates: list[tuple[float, int, int]] = []
        for detection_index, sample in enumerate(frame_samples):
            det_box = (
                float(sample["cx_norm"]),
                float(sample["cy_norm"]),
                float(sample["w_norm"]),
                float(sample["h_norm"]),
            )
            for gt_index, gt_box in enumerate(gt_boxes):
                overlap = iou(det_box, gt_box)
                if overlap >= iou_threshold:
                    candidates.append((overlap, detection_index, gt_index))

        candidates.sort(reverse=True)
        matched_detections: set[int] = set()
        matched_gt: set[int] = set()
        assigned_iou: dict[int, float] = {}
        for overlap, detection_index, gt_index in candidates:
            if detection_index in matched_detections or gt_index in matched_gt:
                continue
            matched_detections.add(detection_index)
            matched_gt.add(gt_index)
            assigned_iou[detection_index] = overlap

        for detection_index, sample in enumerate(frame_samples):
            record = dict(sample)
            record["label"] = 1 if detection_index in matched_detections else 0
            record["matched_iou"] = float(assigned_iou.get(detection_index, 0.0))
            record["label_source"] = "ground_truth_iou"
            labeled.append(record)

    if missing_gt_frames:
        print(f"  [WARN] Skipped {missing_gt_frames} frames with no GT label file.")
    return labeled


def label_conf_only(samples: list[dict], threshold: float) -> list[dict]:
    print("[WARN] CONFIDENCE-ONLY LABELS ARE BOOTSTRAP DATA, NOT CALIBRATION GROUND TRUTH.")
    labeled: list[dict] = []
    for sample in samples:
        record = dict(sample)
        record["label"] = 1 if float(record["raw_conf"]) >= threshold else 0
        record["label_source"] = "bootstrap_confidence_threshold"
        labeled.append(record)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign labels to King Aim calibration samples")
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--out", default="data/calibration_data.json")
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-only", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=0.6)
    args = parser.parse_args()

    expanded_paths: list[str] = []
    for pattern in args.samples:
        expanded = glob.glob(pattern, recursive=True)
        expanded_paths.extend(expanded if expanded else [pattern])

    samples = load_samples(expanded_paths)
    if not samples:
        raise SystemExit("No samples loaded")

    print(f"\nTotal: {len(samples)} samples across {len(set(str(s['session_id']) for s in samples))} sessions")
    if args.conf_only:
        labeled = label_conf_only(samples, args.conf_threshold)
    else:
        if not args.gt_dir:
            raise SystemExit("--gt-dir is required unless --conf-only is explicitly selected")
        labeled = label_with_gt(samples, Path(args.gt_dir), args.iou_threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(labeled, handle, separators=(",", ":"))

    positives = sum(int(sample["label"]) for sample in labeled)
    print(f"\nWrote {len(labeled)} labeled samples -> {out_path}")
    if labeled:
        print(
            f"  TP={positives} FP={len(labeled)-positives} "
            f"positive_rate={positives/len(labeled)*100:.1f}%"
        )


if __name__ == "__main__":
    main()
