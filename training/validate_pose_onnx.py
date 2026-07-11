"""
King Aim pose export parity checker.

Compares raw PyTorch and ONNX Runtime CPU outputs from the SAME preprocessed input.
Use this before setting keypoint_visibility_is_logit in manifest.json. The script
prints the raw keypoint-visibility channel distribution and parity errors; it does
not guess whether a second sigmoid is appropriate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from ultralytics import YOLO


def preprocess(image_path: Path, size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO pose PyTorch/ONNX raw-output parity")
    parser.add_argument("--pt", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--class-count", type=int, default=1)
    parser.add_argument("--keypoint-count", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    tensor_np = preprocess(Path(args.image), args.imgsz)
    tensor_torch = torch.from_numpy(tensor_np)

    yolo = YOLO(args.pt)
    module = yolo.model.eval().cpu()
    with torch.no_grad():
        pytorch_output = module(tensor_torch)
    if isinstance(pytorch_output, (tuple, list)):
        pytorch_output = pytorch_output[0]
    pytorch_np = pytorch_output.detach().cpu().numpy()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: tensor_np})[0]

    print("PyTorch shape:", pytorch_np.shape)
    print("ONNX shape:   ", onnx_output.shape)
    if pytorch_np.shape != onnx_output.shape:
        raise SystemExit("Output shape mismatch; validate export/decoder schema before deployment")

    absolute = np.abs(pytorch_np - onnx_output)
    max_error = float(absolute.max())
    mean_error = float(absolute.mean())
    print(f"max_abs_error={max_error:.8g}")
    print(f"mean_abs_error={mean_error:.8g}")

    expected_channels = 4 + args.class_count + args.keypoint_count * 3
    if pytorch_np.ndim != 3 or pytorch_np.shape[1] != expected_channels:
        raise SystemExit(
            f"Expected [1,{expected_channels},N] for yolo-pose-kpt-v1; got {pytorch_np.shape}"
        )

    visibility_rows = [4 + args.class_count + keypoint_index * 3 + 2 for keypoint_index in range(args.keypoint_count)]
    vis = onnx_output[:, visibility_rows, :]
    print(
        "raw_visibility: "
        f"min={vis.min():.6f} p01={np.quantile(vis, 0.01):.6f} "
        f"median={np.median(vis):.6f} p99={np.quantile(vis, 0.99):.6f} max={vis.max():.6f}"
    )
    if vis.min() >= 0.0 and vis.max() <= 1.0:
        print("Visibility values are already bounded 0..1 on this export. Keep keypoint_visibility_is_logit=false.")
    else:
        print("Visibility values leave 0..1. Inspect exporter semantics before setting keypoint_visibility_is_logit=true.")
    report = {
        "pytorch_shape": list(pytorch_np.shape), "onnx_shape": list(onnx_output.shape),
        "max_abs_error": max_error, "mean_abs_error": mean_error, "tolerance": args.tolerance,
        "status": "PASS" if max_error <= args.tolerance else "FAIL",
        "visibility_min": float(vis.min()), "visibility_max": float(vis.max()),
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "pose_parity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (args.output_dir / "pose_parity_report.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=report.keys()); writer.writeheader(); writer.writerow(report)
        (args.output_dir / "pose_parity_summary.txt").write_text(
            f"status={report['status']}\nmax_abs_error={max_error:.8g}\ntolerance={args.tolerance:.8g}\n", encoding="utf-8"
        )
    if max_error > args.tolerance:
        raise SystemExit(f"Parity failed: max_abs_error {max_error:.8g} > tolerance {args.tolerance:.8g}")


if __name__ == "__main__":
    main()
