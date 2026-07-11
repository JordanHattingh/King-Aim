"""Export a frozen YOLOv8 checkpoint to static FP32 ONNX and an epoch-aware manifest."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from foundation import atomic_json, sha256_file, utc_now, write_checksums


def baseline_id(epoch: int) -> str:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    return f"kingaim-yolov8-baseline-e{epoch:03d}"


def build_manifest(
    checkpoint: Path,
    destination: Path,
    epoch: int,
    imgsz: int,
    names: dict[int, str] | dict[str, str],
) -> dict:
    identifier = baseline_id(epoch)
    return {
        "schema_version": "2",
        "id": identifier,
        "version": "1.0.0-rehearsal" if epoch < 50 else "1.0.0",
        "model_path": destination.name,
        "model_file": destination.name,
        "model_sha256": sha256_file(destination),
        "checkpoint_sha256": sha256_file(checkpoint),
        "created_at_utc": utc_now(),
        "task": "detect",
        "input_width": imgsz,
        "input_height": imgsz,
        "input_shape": [1, 3, imgsz, imgsz],
        "output_schema": "yolo-detector-v1",
        "output_layout": "[1,4+classes,N]",
        "class_names": {str(key): value for key, value in names.items()},
        "is_pose_model": False,
        "keypoint_count": 0,
        "baseline_epoch": epoch,
        "precision": "fp32",
        "opset": 18,
        "dynamic_shapes": False,
        "nms_embedded": False,
        "release_status": "rehearsal" if epoch < 50 else "baseline",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export King Aim YOLOv8 baseline to FP32 ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=50)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.epoch <= 0 or args.imgsz <= 0:
        parser.error("--epoch and --imgsz must be positive")

    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    if getattr(model, "task", None) != "detect":
        raise SystemExit(f"Expected detector checkpoint, got task={getattr(model, 'task', None)!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=18,
            simplify=True,
            dynamic=False,
            half=False,
            nms=False,
        )
    )
    identifier = baseline_id(args.epoch)
    destination = args.output_dir / f"{identifier}-fp32.onnx"
    if exported.resolve() != destination.resolve():
        shutil.move(exported, destination)
    manifest = build_manifest(args.checkpoint, destination, args.epoch, args.imgsz, model.names)
    atomic_json(args.output_dir / "manifest.json", manifest)
    write_checksums(
        args.output_dir,
        [destination, args.output_dir / "manifest.json"],
        args.output_dir / "checksums.sha256",
    )
    print(f"Exported {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
