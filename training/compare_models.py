"""Evaluate detector and pose candidates against one locked dataset."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare YOLOv8 and YOLO11 candidates")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path, help="Locked benchmark image directory")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from ultralytics import YOLO

    images = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        raise SystemExit("No benchmark images found")
    rows: list[dict] = []
    for model_path in args.model:
        model = YOLO(model_path)
        metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device, plots=False, verbose=False)
        latencies: list[float] = []
        for image in images:
            started = time.perf_counter()
            model.predict(str(image), imgsz=args.imgsz, device=args.device, verbose=False)
            latencies.append((time.perf_counter() - started) * 1000)
        box = getattr(metrics, "box", None)
        pose = getattr(metrics, "pose", None)
        rows.append({
            "model": model_path, "images": len(images), "precision": float(getattr(box, "mp", 0.0)),
            "recall": float(getattr(box, "mr", 0.0)), "map50": float(getattr(box, "map50", 0.0)),
            "map50_95": float(getattr(box, "map", 0.0)), "pose_map50": float(getattr(pose, "map50", 0.0)) if pose else None,
            "pose_map50_95": float(getattr(pose, "map", 0.0)) if pose else None,
            "latency_p50_ms": statistics.median(latencies), "latency_p95_ms": percentile(latencies, 0.95),
            "latency_p99_ms": percentile(latencies, 0.99),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
