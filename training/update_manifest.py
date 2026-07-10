"""King Aim model-bundle manifest updater (schema v2)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from contracts import (
    CALIBRATION_FEATURE_SCHEMA,
    GRU_NORM_FIELDS,
    MOVEMENT_FEATURE_SCHEMA,
    TEMPORAL_FEATURE_SCHEMA,
)


def relative_path(manifest_dir: Path, model_path: Path) -> str:
    try:
        return os.path.relpath(model_path, manifest_dir)
    except ValueError:
        return str(model_path.resolve())


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        backup = path.with_suffix(path.suffix + ".bak")
        if path.exists():
            shutil.copy2(path, backup)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update King Aim schema-v2 model manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gru")
    parser.add_argument("--gru-norm")
    parser.add_argument("--cal")
    parser.add_argument("--move")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--pose", action="store_true", help="Declare the vision model as a pose model")
    parser.add_argument("--keypoint-count", type=int, default=4)
    parser.add_argument(
        "--keypoint-names",
        default="head,neck,chest,hip",
        help="Comma-separated keypoint names in exported channel order",
    )
    parser.add_argument(
        "--keypoint-visibility-is-logit",
        action="store_true",
        help="Set only after raw PyTorch/ONNX parity proves exported visibility is a raw logit",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest_dir = manifest_path.parent
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    manifest["schema_version"] = "2"
    changed = False

    def set_model_path(key: str, source_text: str | None) -> None:
        nonlocal changed
        if source_text is None:
            return
        source = Path(source_text).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if args.copy:
            destination = manifest_dir / source.name
            shutil.copy2(source, destination)
            manifest[key] = source.name
        else:
            manifest[key] = relative_path(manifest_dir, source)
        changed = True

    set_model_path("temporal_model_path", args.gru)
    set_model_path("calibration_model_path", args.cal)
    set_model_path("movement_model_path", args.move)

    if args.gru:
        manifest["temporal_feature_schema"] = TEMPORAL_FEATURE_SCHEMA
        if not args.gru_norm:
            raise ValueError("--gru-norm is mandatory when attaching a GRU model")
    if args.cal:
        manifest["calibration_feature_schema"] = CALIBRATION_FEATURE_SCHEMA
    if args.move:
        manifest["movement_feature_schema"] = MOVEMENT_FEATURE_SCHEMA

    if args.gru_norm:
        norm_path = Path(args.gru_norm).resolve()
        with norm_path.open(encoding="utf-8") as handle:
            norm = json.load(handle)
        required = set(GRU_NORM_FIELDS)
        missing = required.difference(norm)
        if missing:
            raise ValueError(f"GRU norm constants missing: {sorted(missing)}")
        if any(float(norm[key]) <= 0 for key in ("log_w_std", "log_h_std", "dt_std", "age_std")):
            raise ValueError("GRU standard deviations must be positive")
        manifest["gru_norm"] = norm
        changed = True

    if args.pose:
        names = [name.strip() for name in args.keypoint_names.split(",") if name.strip()]
        if len(names) != args.keypoint_count:
            raise ValueError("--keypoint-names count must equal --keypoint-count")
        manifest["is_pose_model"] = True
        manifest["output_schema"] = "yolo-pose-kpt-v1"
        manifest["keypoint_count"] = args.keypoint_count
        manifest["keypoint_names"] = names
        manifest["keypoint_visibility_is_logit"] = bool(args.keypoint_visibility_is_logit)
        changed = True

    if not changed:
        raise SystemExit("Nothing to update")

    atomic_write_json(manifest_path, manifest)
    print(f"Updated {manifest_path}")
    print("schema_version=2")
    print("Reload the model bundle in King Aim after validating the feature schemas.")


if __name__ == "__main__":
    main()
