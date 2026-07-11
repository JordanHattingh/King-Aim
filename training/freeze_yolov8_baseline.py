"""Copy a live YOLOv8 epoch checkpoint into an immutable King Aim baseline bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from foundation import atomic_json, environment_report, git_commit, hash_tree, sha256_file, utc_now, write_checksums

ARTIFACTS = (
    "weights/epoch50.pt", "weights/best.pt", "weights/last.pt", "args.yaml", "results.csv",
    "results.png", "confusion_matrix.png", "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png",
)


def image_count(dataset: Path, split: str) -> int:
    directory = dataset / "images" / split
    return sum(1 for path in directory.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) if directory.is_dir() else 0


def freeze(run: Path, output: Path, dataset: Path | None, epoch: int, allow_missing_reports: bool) -> dict:
    checkpoint = run / "weights" / f"epoch{epoch}.pt"
    required = [checkpoint, run / "weights/best.pt", run / "weights/last.pt", run / "args.yaml", run / "results.csv"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required baseline artifacts missing: " + ", ".join(map(str, missing)))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Baseline output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for relative in ARTIFACTS:
        source = run / relative.replace("epoch50.pt", f"epoch{epoch}.pt")
        if not source.is_file():
            if allow_missing_reports and not relative.startswith("weights/"):
                continue
            raise FileNotFoundError(source)
        destination = output / (f"kingaim-yolov8-baseline-e{epoch:03d}.pt" if relative.startswith("weights/epoch") else relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    args_text = (run / "args.yaml").read_text(encoding="utf-8", errors="replace")
    manifest = {
        "schema_version": 1, "model_family": "YOLOv8", "checkpoint_epoch": epoch,
        "created_at_utc": utc_now(), "source_run": str(run.resolve()), "dataset_path": str(dataset.resolve()) if dataset else None,
        "dataset_sha256": hash_tree(dataset, ("*.jpg", "*.jpeg", "*.png", "*.txt", "*.yaml")) if dataset and dataset.is_dir() else None,
        "train_image_count": image_count(dataset, "train") if dataset else None,
        "validation_image_count": image_count(dataset, "val") if dataset else None,
        "git_commit": git_commit(Path(__file__).resolve().parents[1]), "training_args_text": args_text,
        "checkpoint_sha256": sha256_file(output / f"kingaim-yolov8-baseline-e{epoch:03d}.pt"),
    }
    atomic_json(output / "baseline_manifest.json", manifest)
    atomic_json(output / "environment_report.json", environment_report())
    copied.extend([output / "baseline_manifest.json", output / "environment_report.json"])
    write_checksums(output, copied, output / "checksums.sha256")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a live YOLOv8 run without modifying it")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--allow-missing-reports", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epoch <= 0:
        parser.error("--epoch must be positive")
    if args.dry_run:
        missing = [str(args.run / item.replace("epoch50.pt", f"epoch{args.epoch}.pt")) for item in ARTIFACTS if not (args.run / item.replace("epoch50.pt", f"epoch{args.epoch}.pt")).is_file()]
        print(json.dumps({"ready": not missing, "missing": missing}, indent=2))
        return 0 if not missing else 2
    freeze(args.run.resolve(), args.output.resolve(), args.dataset.resolve() if args.dataset else None, args.epoch, args.allow_missing_reports)
    print(f"Frozen YOLOv8 epoch {args.epoch} baseline at {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
