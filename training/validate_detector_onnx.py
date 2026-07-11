"""Validate raw YOLO detector parity between PyTorch and ONNX Runtime CPU.

This tool compares the exact same float32 NCHW tensors through the trusted
Ultralytics checkpoint and its static FP32 ONNX export. It is deliberately
separate from pose parity because detector outputs have no keypoint channels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else 0.0


def synthetic_inputs(size: int, seed: int) -> list[tuple[str, np.ndarray]]:
    y, x = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(seed)
    gradient_hwc = np.stack(
        [
            x / max(1, size - 1),
            y / max(1, size - 1),
            ((x + y) % 256) / 255.0,
        ],
        axis=2,
    ).astype(np.float32)
    checker = (((x // 16) + (y // 16)) % 2).astype(np.float32)
    return [
        ("synthetic_zeros", np.zeros((1, 3, size, size), dtype=np.float32)),
        ("synthetic_ones", np.ones((1, 3, size, size), dtype=np.float32)),
        ("synthetic_gradient", np.transpose(gradient_hwc, (2, 0, 1))[None, ...]),
        ("synthetic_random", rng.random((1, 3, size, size), dtype=np.float32)),
        ("synthetic_checker", np.broadcast_to(checker[None, None, ...], (1, 3, size, size)).copy()),
    ]


def image_input(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def normalize_output(value: object) -> np.ndarray:
    import torch

    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor output, got {type(value)!r}")
    return value.detach().cpu().numpy()


def summarize_case(
    case: str,
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    class_count: int,
    thresholds: Iterable[float],
) -> dict:
    if pytorch_output.shape != onnx_output.shape:
        raise ValueError(f"Output shape mismatch for {case}: {pytorch_output.shape} vs {onnx_output.shape}")
    expected_channels = 4 + class_count
    if pytorch_output.ndim != 3 or pytorch_output.shape[0] != 1 or pytorch_output.shape[1] != expected_channels:
        raise ValueError(
            f"Expected detector output [1,{expected_channels},N] for yolo-detector-v1; "
            f"got {pytorch_output.shape}"
        )
    if not np.isfinite(pytorch_output).all() or not np.isfinite(onnx_output).all():
        raise ValueError(f"Non-finite detector output in case {case}")

    absolute = np.abs(pytorch_output - onnx_output)
    box_absolute = absolute[:, :4, :]
    class_absolute = absolute[:, 4:, :]
    pytorch_scores = pytorch_output[:, 4:, :].max(axis=1)
    onnx_scores = onnx_output[:, 4:, :].max(axis=1)
    counts = {}
    for threshold in thresholds:
        key = f"{threshold:g}"
        pt_count = int((pytorch_scores >= threshold).sum())
        onnx_count = int((onnx_scores >= threshold).sum())
        counts[key] = {
            "pytorch": pt_count,
            "onnx": onnx_count,
            "match": pt_count == onnx_count,
        }

    return {
        "case": case,
        "shape": list(pytorch_output.shape),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "p99_abs_error": percentile(absolute, 0.99),
        "p999_abs_error": percentile(absolute, 0.999),
        "box_max_abs_error": float(box_absolute.max()),
        "box_mean_abs_error": float(box_absolute.mean()),
        "class_max_abs_error": float(class_absolute.max()),
        "class_mean_abs_error": float(class_absolute.mean()),
        "max_confidence_pytorch": float(pytorch_scores.max()),
        "max_confidence_onnx": float(onnx_scores.max()),
        "candidate_counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YOLO detector PyTorch/ONNX raw-output parity")
    parser.add_argument("--pt", required=True, type=Path, help="Trusted Ultralytics checkpoint")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--class-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--max-abs-tolerance", type=float, default=5e-4)
    parser.add_argument("--mean-abs-tolerance", type=float, default=5e-5)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.imgsz <= 0 or args.class_count <= 0:
        parser.error("--imgsz and --class-count must be positive")
    for path in (args.pt, args.onnx, *args.image):
        if not path.is_file():
            raise FileNotFoundError(path)

    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    inputs: list[tuple[str, np.ndarray]] = []
    if not args.skip_synthetic:
        inputs.extend(synthetic_inputs(args.imgsz, args.seed))
    inputs.extend((f"image:{path.name}", image_input(path, args.imgsz)) for path in args.image)
    if not inputs:
        raise SystemExit("No parity inputs selected")

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = YOLO(str(args.pt))
    if getattr(model, "task", None) != "detect":
        raise SystemExit(f"Expected a detector checkpoint, got task={getattr(model, 'task', None)!r}")
    module = model.model.eval().cpu()
    available_providers = ort.get_available_providers()
    if args.provider not in available_providers:
        raise SystemExit(
            f"Requested ONNX Runtime provider {args.provider!r} is unavailable. "
            f"Available providers: {available_providers}"
        )
    session = ort.InferenceSession(str(args.onnx), providers=[args.provider])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise SystemExit("Expected one ONNX input and one ONNX output")
    input_meta = session.get_inputs()[0]
    expected_input = [1, 3, args.imgsz, args.imgsz]
    if list(input_meta.shape) != expected_input or input_meta.type != "tensor(float)":
        raise SystemExit(f"Unexpected ONNX input contract: {input_meta.type} {input_meta.shape}")

    rows: list[dict] = []
    for label, tensor_np in inputs:
        with torch.no_grad():
            pytorch_np = normalize_output(module(torch.from_numpy(tensor_np)))
        onnx_np = session.run(None, {input_meta.name: tensor_np})[0]
        rows.append(summarize_case(label, pytorch_np, onnx_np, args.class_count, (0.10, 0.25, 0.50)))

    max_error = max(row["max_abs_error"] for row in rows)
    max_mean_error = max(row["mean_abs_error"] for row in rows)
    counts_match = all(
        threshold["match"]
        for row in rows
        for threshold in row["candidate_counts"].values()
    )
    passed = (
        max_error <= args.max_abs_tolerance
        and max_mean_error <= args.mean_abs_tolerance
        and counts_match
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "task": "detect",
        "output_schema": "yolo-detector-v1",
        "pytorch_checkpoint": str(args.pt.resolve()),
        "onnx_model": str(args.onnx.resolve()),
        "onnx_provider": args.provider,
        "input_shape": expected_input,
        "output_shape": rows[0]["shape"],
        "class_count": args.class_count,
        "max_abs_tolerance": args.max_abs_tolerance,
        "mean_abs_tolerance": args.mean_abs_tolerance,
        "observed_max_abs_error": max_error,
        "observed_max_mean_abs_error": max_mean_error,
        "candidate_counts_match": counts_match,
        "cases": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "detector_parity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "detector_parity_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case", "max_abs_error", "mean_abs_error", "p99_abs_error", "p999_abs_error",
            "box_max_abs_error", "box_mean_abs_error", "class_max_abs_error", "class_mean_abs_error",
            "max_confidence_pytorch", "max_confidence_onnx",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    (args.output_dir / "detector_parity_summary.txt").write_text(
        f"status={report['status']}\n"
        f"observed_max_abs_error={max_error:.10g}\n"
        f"max_abs_tolerance={args.max_abs_tolerance:.10g}\n"
        f"observed_max_mean_abs_error={max_mean_error:.10g}\n"
        f"mean_abs_tolerance={args.mean_abs_tolerance:.10g}\n"
        f"candidate_counts_match={str(counts_match).lower()}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Detector parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
