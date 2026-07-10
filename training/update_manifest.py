"""
King Aim — Manifest updater.

After training the companion networks, run this to inject the trained ONNX paths
and GRU norm constants into your model's manifest.json.

Usage (typical after a full training run):
    python update_manifest.py \\
        --manifest "Models/MyModel/manifest.json" \\
        --gru      runs/gru/trajectory_gru.onnx \\
        --gru-norm runs/gru/norm_constants.json \\
        --cal      runs/calibration/calibration_mlp.onnx \\
        --move     runs/movement/movement_mlp.onnx

Paths in the manifest are stored relative to the manifest file's directory so
the model bundle stays portable.

If you only trained one network, omit the others — existing values are preserved.
"""

import json, os, shutil, argparse
from pathlib import Path


def rel(manifest_dir, onnx_path):
    """Return a path relative to manifest_dir, or absolute if on a different drive."""
    try:
        return os.path.relpath(onnx_path, manifest_dir)
    except ValueError:
        return str(Path(onnx_path).resolve())


def main():
    parser = argparse.ArgumentParser(description="Update manifest.json with trained companion models")
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest.json to update")
    parser.add_argument("--gru",      default=None,
                        help="Path to trajectory_gru.onnx (output of train_gru.py)")
    parser.add_argument("--gru-norm", default=None,
                        help="Path to norm_constants.json (output of train_gru.py)")
    parser.add_argument("--cal",      default=None,
                        help="Path to calibration_mlp.onnx (output of train_calibration.py)")
    parser.add_argument("--move",     default=None,
                        help="Path to movement_mlp.onnx (output of train_movement.py)")
    parser.add_argument("--copy",     action="store_true",
                        help="Copy ONNX files into the manifest's directory instead of using relative paths")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] manifest.json not found: {manifest_path}")
        return

    manifest_dir = manifest_path.parent

    with open(manifest_path) as f:
        manifest = json.load(f)

    changed = False

    def set_path(key, src):
        nonlocal changed
        if src is None:
            return
        src = Path(src).resolve()
        if not src.exists():
            print(f"[WARN] File not found, skipping: {src}")
            return
        if args.copy:
            dest = manifest_dir / src.name
            shutil.copy2(src, dest)
            manifest[key] = src.name
            print(f"  Copied {src.name} → {dest}")
        else:
            manifest[key] = rel(manifest_dir, src)
            print(f"  Set {key} = {manifest[key]}")
        changed = True

    set_path("temporal_model_path",     args.gru)
    set_path("calibration_model_path",  args.cal)
    set_path("movement_model_path",     args.move)

    if args.gru_norm:
        norm_path = Path(args.gru_norm).resolve()
        if not norm_path.exists():
            print(f"[WARN] norm_constants.json not found: {norm_path}")
        else:
            with open(norm_path) as f:
                norm = json.load(f)
            manifest["gru_norm"] = {
                "log_w_mean": norm["log_w_mean"],
                "log_w_std":  norm["log_w_std"],
                "log_h_mean": norm["log_h_mean"],
                "log_h_std":  norm["log_h_std"],
                "dt_mean":    norm["dt_mean"],
                "dt_std":     norm["dt_std"],
                "age_mean":   norm["age_mean"],
                "age_std":    norm["age_std"],
            }
            print(f"  Set gru_norm from {norm_path.name}")
            changed = True

    if not changed:
        print("Nothing to update — pass at least one of --gru, --cal, --move.")
        return

    # Back up the original
    backup = manifest_path.with_suffix(".json.bak")
    shutil.copy2(manifest_path, backup)
    print(f"\n  Backup saved: {backup.name}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Updated: {manifest_path}")

    print("\nReload the model in King Aim to apply the new networks.")


if __name__ == "__main__":
    main()
