"""Export a frozen YOLOv8 checkpoint to static FP32 ONNX and model manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from foundation import atomic_json, sha256_file, utc_now, write_checksums


def main() -> int:
    parser = argparse.ArgumentParser(description="Export King Aim YOLOv8 baseline to FP32 ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=50)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(YOLO(str(args.checkpoint)).export(format="onnx", imgsz=args.imgsz, opset=18, simplify=True, dynamic=False, half=False))
    destination = args.output_dir / f"kingaim-yolov8-baseline-e{args.epoch:03d}-fp32.onnx"
    if exported.resolve() != destination.resolve():
        shutil.move(exported, destination)
    manifest = {
        "schema_version": "2", "id": "kingaim-yolov8-baseline-e050", "version": "1.0.0",
        "model_file": destination.name, "model_sha256": sha256_file(destination), "created_at_utc": utc_now(),
        "input_width": args.imgsz, "input_height": args.imgsz, "output_schema": "yolo-detector-v1",
        "is_pose_model": False, "keypoint_count": 0, "baseline_epoch": args.epoch,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    write_checksums(args.output_dir, [destination, args.output_dir / "manifest.json"], args.output_dir / "checksums.sha256")
    print(f"Exported {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
