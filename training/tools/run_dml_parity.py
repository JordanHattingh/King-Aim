"""DirectML vs CPU ONNX parity validator for King Aim detector baselines.

Must be run from C:\\Tmp\\dml_venv (onnxruntime-directml):
  C:\\Tmp\\dml_venv\\Scripts\\python.exe training\\tools\\run_dml_parity.py \\
      --onnx <path> --output-dir <dir> [--image <path>] [--imgsz 512]

Compares DmlExecutionProvider against CPUExecutionProvider.
Tolerance defaults: max_abs=0.005, mean_abs=0.0005.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def synthetic_inputs(size: int, seed: int) -> list[tuple[str, np.ndarray]]:
    y, x = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(seed)
    gradient_hwc = np.stack(
        [x / max(1, size - 1), y / max(1, size - 1), ((x + y) % 256) / 255.0],
        axis=2,
    ).astype(np.float32)
    checker = (((x // 16) + (y // 16)) % 2).astype(np.float32)
    return [
        ("synthetic_zeros",    np.zeros((1, 3, size, size), dtype=np.float32)),
        ("synthetic_ones",     np.ones((1, 3, size, size), dtype=np.float32)),
        ("synthetic_gradient", np.transpose(gradient_hwc, (2, 0, 1))[None, ...]),
        ("synthetic_random",   rng.random((1, 3, size, size), dtype=np.float32)),
        ("synthetic_checker",  np.broadcast_to(checker[None, None, ...], (1, 3, size, size)).copy()),
    ]


def image_input(path: Path, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description="DirectML vs CPU ONNX parity for King Aim detectors")
    parser.add_argument("--onnx",             required=True, type=Path)
    parser.add_argument("--output-dir",       required=True, type=Path)
    parser.add_argument("--image",            action="append", type=Path, default=[])
    parser.add_argument("--imgsz",            type=int, default=512)
    parser.add_argument("--seed",             type=int, default=20260711)
    parser.add_argument("--max-abs-tolerance",  type=float, default=0.005)
    parser.add_argument("--mean-abs-tolerance", type=float, default=0.0005)
    parser.add_argument("--skip-synthetic",   action="store_true")
    args = parser.parse_args()

    available = ort.get_available_providers()
    if "DmlExecutionProvider" not in available:
        print(f"DmlExecutionProvider not available. Got: {available}", file=sys.stderr)
        return 2

    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)

    cpu_session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    dml_session = ort.InferenceSession(str(args.onnx), providers=["DmlExecutionProvider"])
    input_name = cpu_session.get_inputs()[0].name

    inputs: list[tuple[str, np.ndarray]] = []
    if not args.skip_synthetic:
        inputs.extend(synthetic_inputs(args.imgsz, args.seed))
    inputs.extend((f"image:{p.name}", image_input(p, args.imgsz)) for p in args.image if p.is_file())
    if not inputs:
        raise SystemExit("No parity inputs selected")

    rows = []
    for label, tensor in inputs:
        cpu_out = cpu_session.run(None, {input_name: tensor})[0]
        dml_out = dml_session.run(None, {input_name: tensor})[0]
        if cpu_out.shape != dml_out.shape:
            raise SystemExit(f"Shape mismatch on {label}: {cpu_out.shape} vs {dml_out.shape}")
        diff = np.abs(cpu_out.astype(np.float32) - dml_out.astype(np.float32))
        rows.append({
            "case":           label,
            "shape":          list(cpu_out.shape),
            "max_abs_error":  float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "p99_abs_error":  float(np.quantile(diff, 0.99)),
        })
        print(f"  {label}: max={diff.max():.8g}  mean={diff.mean():.8g}")

    max_error  = max(r["max_abs_error"]  for r in rows)
    mean_error = max(r["mean_abs_error"] for r in rows)
    passed = max_error <= args.max_abs_tolerance and mean_error <= args.mean_abs_tolerance

    report = {
        "status":                  "PASS" if passed else "FAIL",
        "provider_reference":      "CPUExecutionProvider",
        "provider_under_test":     "DmlExecutionProvider",
        "onnx_model":              str(args.onnx.resolve()),
        "max_abs_tolerance":       args.max_abs_tolerance,
        "mean_abs_tolerance":      args.mean_abs_tolerance,
        "observed_max_abs_error":  max_error,
        "observed_max_mean_error": mean_error,
        "cases":                   rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "detector_dml_parity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "detector_dml_parity_summary.txt").write_text(
        f"status={report['status']}\n"
        f"provider_under_test=DmlExecutionProvider\n"
        f"observed_max_abs_error={max_error:.10g}\n"
        f"max_abs_tolerance={args.max_abs_tolerance:.10g}\n"
        f"observed_max_mean_error={mean_error:.10g}\n"
        f"mean_abs_tolerance={args.mean_abs_tolerance:.10g}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("DirectML parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
