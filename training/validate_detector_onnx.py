"""Validate raw YOLO detector parity between PyTorch and ONNX Runtime.

Reports two separate categories:

  Deployment parity — real gameplay images passed via --image.
      These are the hard gate.  If any gameplay frame fails, the run fails.

  Synthetic stress diagnostics — zero/ones/gradient/random/checker tensors.
      Always recorded.  When gameplay images are present these are diagnostic
      only and do not gate the run.  When no --image flags are given the
      synthetic inputs act as the gate (backwards-compatible behaviour).

Hard failures (always fail the run):
  - output shape mismatch
  - NaN or Inf in any output
  - real gameplay frame exceeds max_abs or mean_abs tolerance
  - candidate-count mismatch on any real gameplay frame
  - model cannot load or provider unavailable

Diagnostic warnings (recorded, do not fail when gameplay images are present):
  - synthetic degenerate tensor slightly exceeds raw max tolerance
  - mean error within spec and normal inputs stable
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


def _section_stats(rows: list[dict], max_tol: float, mean_tol: float, is_gate: bool) -> dict:
    """Aggregate a list of summarize_case rows into a section summary."""
    if not rows:
        return {
            "status": "PASS",
            "gate": is_gate,
            "case_count": 0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_abs_tolerance": max_tol,
            "mean_abs_tolerance": mean_tol,
            "candidate_counts_match": True,
            "cases_exceeding_tolerance": [],
            "cases": [],
        }

    max_error = max(r["max_abs_error"] for r in rows)
    mean_error = max(r["mean_abs_error"] for r in rows)
    counts_match = all(t["match"] for r in rows for t in r["candidate_counts"].values())
    over_tol = [r["case"] for r in rows if r["max_abs_error"] > max_tol]

    if is_gate:
        passed = max_error <= max_tol and mean_error <= mean_tol and counts_match
        status = "PASS" if passed else "FAIL"
    else:
        # Diagnostic: WARN if any case exceeds tolerance, PASS otherwise
        status = "WARN" if over_tol else "PASS"

    return {
        "status": status,
        "gate": is_gate,
        "case_count": len(rows),
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "max_abs_tolerance": max_tol,
        "mean_abs_tolerance": mean_tol,
        "candidate_counts_match": counts_match,
        "cases_exceeding_tolerance": over_tol,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YOLO detector PyTorch/ONNX raw-output parity")
    parser.add_argument("--pt", required=True, type=Path, help="Trusted Ultralytics checkpoint")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--image", action="append", type=Path, default=[],
                        help="Gameplay image (repeatable). These form the deployment parity gate.")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--class-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Omit synthetic stress cases entirely")
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

    thresholds = (0.10, 0.25, 0.50)

    def run_case(label: str, tensor_np: np.ndarray) -> dict:
        with torch.no_grad():
            pytorch_np = normalize_output(module(torch.from_numpy(tensor_np)))
        onnx_np = session.run(None, {input_meta.name: tensor_np})[0]
        return summarize_case(label, pytorch_np, onnx_np, args.class_count, thresholds)

    # Collect synthetic diagnostic cases
    synthetic_rows: list[dict] = []
    if not args.skip_synthetic:
        for label, tensor_np in synthetic_inputs(args.imgsz, args.seed):
            synthetic_rows.append(run_case(label, tensor_np))

    # Collect deployment gameplay cases
    gameplay_rows: list[dict] = []
    for path in args.image:
        gameplay_rows.append(run_case(f"image:{path.name}", image_input(path, args.imgsz)))

    if not synthetic_rows and not gameplay_rows:
        raise SystemExit("No parity inputs selected")

    has_gameplay = bool(gameplay_rows)

    # Deployment section: gameplay images (gate=True) if any, else None
    deployment = _section_stats(gameplay_rows, args.max_abs_tolerance, args.mean_abs_tolerance, is_gate=True) if has_gameplay else None

    # Synthetic section: diagnostic (gate=False) when gameplay present, gate otherwise
    synthetic = _section_stats(synthetic_rows, args.max_abs_tolerance, args.mean_abs_tolerance, is_gate=not has_gameplay)
    if not has_gameplay:
        synthetic["note"] = "No gameplay images provided. Synthetic inputs used as gate."
    else:
        synthetic["note"] = "Diagnostic only. Synthetic failures do not gate deployment."

    # Overall pass/fail
    if has_gameplay:
        passed = deployment["status"] == "PASS"
        overall_status = "PASS" if passed else "FAIL"
        note = (
            f"Deployment gate: {deployment['status']}. "
            f"Synthetic diagnostics: {synthetic['status']}."
        )
    else:
        passed = synthetic["status"] == "PASS"
        overall_status = "PASS" if passed else "FAIL"
        note = "No gameplay images. Synthetic inputs used as gate."

    report = {
        "status": overall_status,
        "note": note,
        "task": "detect",
        "output_schema": "yolo-detector-v1",
        "pytorch_checkpoint": str(args.pt.resolve()),
        "onnx_model": str(args.onnx.resolve()),
        "onnx_provider": args.provider,
        "input_shape": expected_input,
        "class_count": args.class_count,
        "max_abs_tolerance": args.max_abs_tolerance,
        "mean_abs_tolerance": args.mean_abs_tolerance,
        "deployment_parity": deployment,
        "synthetic_diagnostics": synthetic,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "detector_parity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    # CSV: all cases with a category column
    all_rows_tagged = (
        [{"category": "deployment", **r} for r in gameplay_rows] +
        [{"category": "synthetic", **r} for r in synthetic_rows]
    )
    csv_fields = [
        "category", "case", "max_abs_error", "mean_abs_error",
        "p99_abs_error", "p999_abs_error",
        "box_max_abs_error", "box_mean_abs_error",
        "class_max_abs_error", "class_mean_abs_error",
        "max_confidence_pytorch", "max_confidence_onnx",
    ]
    with (args.output_dir / "detector_parity_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows_tagged)

    # Summary text
    dep_line = (
        f"deployment_status={deployment['status']}\n"
        f"deployment_max_abs_error={deployment['max_abs_error']:.10g}\n"
        f"deployment_mean_abs_error={deployment['mean_abs_error']:.10g}\n"
        f"deployment_candidate_counts_match={str(deployment['candidate_counts_match']).lower()}\n"
        if deployment else
        "deployment_status=NO_GAMEPLAY_IMAGES\n"
    )
    synth_over = ",".join(synthetic["cases_exceeding_tolerance"]) or "none"
    summary = (
        f"status={overall_status}\n"
        f"note={note}\n"
        + dep_line +
        f"synthetic_status={synthetic['status']}\n"
        f"synthetic_max_abs_error={synthetic['max_abs_error']:.10g}\n"
        f"synthetic_mean_abs_error={synthetic['mean_abs_error']:.10g}\n"
        f"synthetic_cases_exceeding_tolerance={synth_over}\n"
        f"max_abs_tolerance={args.max_abs_tolerance:.10g}\n"
        f"mean_abs_tolerance={args.mean_abs_tolerance:.10g}\n"
    )
    (args.output_dir / "detector_parity_summary.txt").write_text(summary, encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("Detector parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
