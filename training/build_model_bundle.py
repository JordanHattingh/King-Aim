"""Assemble and checksum a schema-v2 King Aim pose model bundle."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from contracts import CALIBRATION_FEATURE_SCHEMA, MOVEMENT_FEATURE_SCHEMA, TEMPORAL_FEATURE_SCHEMA
from foundation import atomic_json, sha256_file, write_checksums


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fail-closed King Aim neural model bundle")
    parser.add_argument("--vision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id", default="kingaim-yolo11-pose")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--temporal", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--movement", type=Path)
    parser.add_argument("--norm-constants", type=Path)
    args = parser.parse_args()
    sources = {"model.onnx": args.vision, "temporal.onnx": args.temporal, "calibration.onnx": args.calibration, "movement.onnx": args.movement, "norm_constants.json": args.norm_constants}
    if not args.vision.is_file():
        raise FileNotFoundError(args.vision)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Bundle output must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for filename, source in sources.items():
        if source is None:
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.output / filename
        shutil.copy2(source, destination); copied.append(destination)
    manifest = {
        "schema_version": "2", "id": args.id, "version": args.version, "model_path": "model.onnx",
        "input_width": args.imgsz, "input_height": args.imgsz, "output_schema": "yolo-pose-kpt-v1",
        "is_pose_model": True, "keypoint_count": 4, "keypoint_names": ["head", "neck", "upper_chest", "hip"],
        "keypoint_visibility_is_logit": False, "temporal_feature_schema": TEMPORAL_FEATURE_SCHEMA,
        "calibration_feature_schema": CALIBRATION_FEATURE_SCHEMA, "movement_feature_schema": MOVEMENT_FEATURE_SCHEMA,
        "temporal_model_path": "temporal.onnx" if args.temporal else None,
        "calibration_model_path": "calibration.onnx" if args.calibration else None,
        "movement_model_path": "movement.onnx" if args.movement else None,
    }
    atomic_json(args.output / "manifest.json", manifest); copied.append(args.output / "manifest.json")
    write_checksums(args.output, copied, args.output / "checksums.sha256")
    (args.output / "MODEL_CARD.md").write_text(
        f"# {args.id}\n\nVersion: {args.version}\n\nFour-keypoint King Aim pose bundle. Companion networks are optional and remain gated by their validation reports.\n",
        encoding="utf-8", newline="\n",
    )
    print(f"Built bundle {args.output} ({sha256_file(args.output / 'checksums.sha256')[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
