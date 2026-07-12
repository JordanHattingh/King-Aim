"""Create the canonical external King Aim training workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

DIRECTORIES = (
    "baseline/yolov8-e050", "source_media/self_captured", "source_media/permissioned",
    "source_media/internet_candidates", "source_media/rejected", "extracted_frames/raw",
    "extracted_frames/deduplicated", "extracted_frames/selected", "pose/images/train", "pose/images/val",
    "pose/images/test", "pose/labels/train", "pose/labels/val", "pose/labels/test", "pose/provenance",
    "pose/permissions", "pose/reports", "gold_holdout", "runs/detect", "runs/pose", "exports/pytorch",
    "exports/onnx_fp32", "exports/onnx_fp16", "exports/model_bundles",
)

YAML = """path: {root}/pose
train: images/train
val: images/val
test: images/test
kpt_shape: [4, 3]
flip_idx: [0, 1, 2, 3]
names:
  0: enemy
kpt_names:
  0: [head, neck, upper_chest, hip]
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize C:/KingAimTraining layout")
    parser.add_argument("--root", type=Path, default=Path("C:/KingAimTraining"))
    args = parser.parse_args()
    root = args.root.resolve()
    for relative in DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    yaml_path = root / "pose/kingaim_pose.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(YAML.format(root=root.as_posix()), encoding="utf-8", newline="\n")
    print(f"King Aim training workspace ready: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
