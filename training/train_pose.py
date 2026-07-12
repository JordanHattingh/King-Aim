"""Reproducible single-model King Aim pose-candidate trainer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from foundation import atomic_json, environment_report, git_commit, hash_tree, utc_now

POSE_CANDIDATES = {
    "yolo26s-pose.pt": "primary",
    "yolo26n-pose.pt": "low-end",
    "yolo11s-pose.pt": "control",
}
DEFAULT_EXPERIMENT_CONTRACT = Path(__file__).resolve().parent / "contracts" / "pose_candidate_matrix.json"


def candidate_role(model: str, resume: object = None) -> str:
    model_name = Path(model).name.lower()
    if resume:
        return POSE_CANDIDATES.get(model_name, "resume")
    if model_name not in POSE_CANDIDATES:
        raise ValueError(f"Model must be one of the frozen candidates: {', '.join(POSE_CANDIDATES)}")
    return POSE_CANDIDATES[model_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one frozen-matrix King Aim pose candidate")
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
    parser.add_argument("--experiment-contract", type=Path, default=DEFAULT_EXPERIMENT_CONTRACT)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict:
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.imgsz < 256 or args.epochs <= 0 or args.batch == 0 or args.save_period <= 0:
        raise ValueError("imgsz must be >=256; epochs/save-period positive; batch non-zero")
    role = candidate_role(args.model, args.resume)
    if not args.experiment_contract.is_file():
        raise FileNotFoundError(f"Experiment contract not found: {args.experiment_contract}")
    contract = json.loads(args.experiment_contract.read_text(encoding="utf-8"))
    if contract.get("candidate_roles") != POSE_CANDIDATES:
        raise ValueError("Experiment contract candidate matrix differs from the trainer")
    for field in ("dataset_revision_sha256", "split_manifest_sha256", "directml_hardware_id"):
        if not contract.get(field):
            raise ValueError(f"Experiment contract is not frozen: {field} is missing")
    if args.imgsz != contract.get("input_size") or args.epochs != contract["training"]["epochs"] or args.seed != contract["training"]["seed"]:
        raise ValueError("Run arguments differ from the frozen experiment contract")
    report = environment_report()
    torch_info = report.get("torch") or {}
    if str(args.device).lower() not in {"cpu", "-1"} and not torch_info.get("cuda_available"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if torch_info.get("gpu_memory_bytes") and int(torch_info["gpu_memory_bytes"]) < 3 * 1024**3:
        raise RuntimeError("GPU has less than 3 GiB VRAM; use --device cpu or reduce the workload")
    report["dataset_yaml_sha256"] = hash_tree(args.data.parent, (args.data.name,))
    report["candidate_role"] = role
    report["experiment_contract"] = str(args.experiment_contract.resolve())
    report["experiment_contract_sha256"] = hash_tree(args.experiment_contract.parent, (args.experiment_contract.name,))
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
        exported = Path(YOLO(str(best)).export(
            format="onnx", imgsz=args.imgsz, batch=1, opset=18, simplify=True,
            dynamic=False, half=False, end2end=False,
        ))
        destination = run_dir / "weights" / f"{args.name}-fp32.onnx"
        if exported.resolve() != destination.resolve():
            shutil.move(exported, destination)
        import onnx
        graph = onnx.load(str(destination)).graph
        def dimensions(value_info: object) -> list[int | str | None]:
            dims = value_info.type.tensor_type.shape.dim
            return [dim.dim_value if dim.HasField("dim_value") else (dim.dim_param or None) for dim in dims]
        atomic_json(run_dir / "weights" / "onnx_export_contract.json", {
            "format": "onnx", "precision": "fp32", "opset": 18, "batch": 1,
            "input_size": args.imgsz, "dynamic": False, "end2end": False,
            "inputs": [{"name": item.name, "shape": dimensions(item)} for item in graph.input],
            "outputs": [{"name": item.name, "shape": dimensions(item)} for item in graph.output],
        })
        print(f"Exported FP32 ONNX: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
