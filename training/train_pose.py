"""Reproducible single-model YOLO11-Pose trainer for King Aim."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from foundation import atomic_json, environment_report, git_commit, hash_tree, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one King Aim YOLO11 pose model")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", nargs="?", const=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict:
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.imgsz < 256 or args.epochs <= 0 or args.batch == 0 or args.save_period <= 0:
        raise ValueError("imgsz must be >=256; epochs/save-period positive; batch non-zero")
    report = environment_report()
    torch_info = report.get("torch") or {}
    if str(args.device).lower() not in {"cpu", "-1"} and not torch_info.get("cuda_available"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if torch_info.get("gpu_memory_bytes") and int(torch_info["gpu_memory_bytes"]) < 3 * 1024**3:
        raise RuntimeError("GPU has less than 3 GiB VRAM; use --device cpu or reduce the workload")
    report["dataset_yaml_sha256"] = hash_tree(args.data.parent, (args.data.name,))
    return report


def main() -> int:
    args = parse_args()
    environment = validate(args)
    run_dir = args.project.resolve() / args.name
    manifest = {
        "schema_version": 1, "created_at_utc": utc_now(), "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "data": str(args.data.resolve()), "model": args.model, "imgsz": args.imgsz, "epochs": args.epochs,
        "batch": args.batch, "device": str(args.device), "workers": args.workers, "seed": args.seed,
        "save_period": args.save_period, "resume": args.resume, "environment": environment,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "kingaim_run_manifest.json", manifest)
    from ultralytics import YOLO

    model = YOLO(args.resume if isinstance(args.resume, str) else args.model)
    model.train(
        data=str(args.data.resolve()), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
        workers=args.workers, seed=args.seed, deterministic=True, optimizer="AdamW", lr0=5e-4, lrf=0.01,
        weight_decay=5e-4, warmup_epochs=3, cos_lr=True, close_mosaic=30, amp=True, hsv_h=0.015,
        hsv_s=0.5, hsv_v=0.3, degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.0,
        mosaic=0.8, mixup=0.1, copy_paste=0.05, erasing=0.3, pose=12.0, kobj=1.0,
        project=str(args.project.resolve()), name=args.name, exist_ok=bool(args.resume), save=True,
        save_period=args.save_period, patience=50, resume=bool(args.resume),
    )
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without best checkpoint: {best}")
    if not args.skip_export:
        exported = Path(YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=18, simplify=True, dynamic=False, half=False))
        destination = run_dir / "weights" / f"{args.name}-fp32.onnx"
        if exported.resolve() != destination.resolve():
            shutil.move(exported, destination)
        print(f"Exported FP32 ONNX: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
