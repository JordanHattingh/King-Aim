"""
King Aim — FP16 ONNX conversion for DirectML deployment.

Converts any of the four King Aim ONNX models (pose, GRU, calibration, movement)
from FP32 to FP16 using onnxconverter-common.

Usage:
    python convert_fp16.py model.onnx              # outputs model_fp16.onnx
    python convert_fp16.py model.onnx out.onnx     # explicit output path
    python convert_fp16.py --all runs/             # convert every *.onnx in a folder

Notes:
  * keep_io_types=True keeps input/output tensors as FP32 so the ORT runtime's
    DirectML EP can accept plain float arrays from C# without explicit casting.
  * GTX 1650 has no Tensor Cores — FP16 helps with memory bandwidth on DirectML
    graph execution but does not activate hardware Tensor Core multiply-accumulate.
  * GRU / calibration / movement models run on CPU (CPUExecutionProvider) and do
    NOT benefit from FP16 on the GTX 1650; skip them unless you benchmark a win.
  * Always validate FP32 ↔ FP16 outputs agree within tolerance before deploying.

Install:
    pip install onnxconverter-common onnxruntime numpy
"""

import sys
import os
import argparse
import numpy as np

try:
    import onnx
    from onnxconverter_common import convert_float_to_float16
    import onnxruntime as ort
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("       pip install onnxconverter-common onnxruntime numpy")
    sys.exit(1)


# ── Validation helpers ────────────────────────────────────────────────────────

def _dummy_inputs(session: ort.InferenceSession) -> dict:
    """Build a dict of random dummy float32 inputs sized to the model's specs."""
    inputs = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        inputs[inp.name] = np.random.rand(*shape).astype(np.float32)
    return inputs


def _run(session: ort.InferenceSession, inputs: dict) -> list:
    return session.run(None, inputs)


def validate(fp32_path: str, fp16_path: str, rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    """
    Compare FP32 and FP16 model outputs on the same random input.
    Uses the same random seed so any stochastic ops are aligned.
    Returns True if all outputs agree within (rtol, atol).
    """
    np.random.seed(42)

    sess32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess16 = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])

    inputs = _dummy_inputs(sess32)

    outs32 = _run(sess32, inputs)
    outs16 = _run(sess16, inputs)

    ok = True
    for i, (o32, o16) in enumerate(zip(outs32, outs16)):
        a32 = np.array(o32, dtype=np.float32)
        a16 = np.array(o16, dtype=np.float32)
        close = np.allclose(a32, a16, rtol=rtol, atol=atol)
        max_diff = float(np.max(np.abs(a32 - a16)))
        status = "OK" if close else "FAIL"
        print(f"  Output[{i}]: max_diff={max_diff:.6f}  [{status}]")
        if not close:
            ok = False

    return ok


# ── Conversion ────────────────────────────────────────────────────────────────

def convert(fp32_path: str, fp16_path: str | None = None) -> str:
    """
    Convert fp32_path to FP16, write to fp16_path (defaults to *_fp16.onnx).
    Returns the path of the written FP16 model.
    """
    if fp16_path is None:
        base, ext = os.path.splitext(fp32_path)
        fp16_path = base + "_fp16" + ext

    print(f"[convert_fp16] {os.path.basename(fp32_path)} → {os.path.basename(fp16_path)}")

    # 1. Validate FP32 model loads cleanly
    print("  Validating FP32 model … ", end="", flush=True)
    model_fp32 = onnx.load(fp32_path)
    onnx.checker.check_model(model_fp32)
    print("OK")

    # 2. Convert
    print("  Converting to FP16 … ", end="", flush=True)
    model_fp16 = convert_float_to_float16(
        model_fp32,
        keep_io_types=True,    # keep FP32 I/O so C# doesn't need to cast
        disable_shape_infer=False,
    )
    onnx.save(model_fp16, fp16_path)
    print("OK")

    # 3. Validate outputs agree
    print("  Validating FP32 ≈ FP16 outputs:")
    passed = validate(fp32_path, fp16_path)
    if passed:
        print(f"  Validation PASSED → {fp16_path}")
    else:
        print(f"  Validation FAILED — FP16 output diverges beyond tolerance.")
        print(f"  The FP16 file was still written; inspect before deploying.")

    size32 = os.path.getsize(fp32_path) / 1024 / 1024
    size16 = os.path.getsize(fp16_path) / 1024 / 1024
    print(f"  Size: {size32:.1f} MB → {size16:.1f} MB  ({100*(1-size16/size32):.0f}% smaller)")

    return fp16_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert King Aim ONNX models from FP32 to FP16")
    parser.add_argument("input", nargs="?", help="FP32 .onnx file to convert")
    parser.add_argument("output", nargs="?", help="Output path (default: <input>_fp16.onnx)")
    parser.add_argument("--all", metavar="DIR",
                        help="Convert every *.onnx in DIR (skips files already containing '_fp16')")
    args = parser.parse_args()

    if args.all:
        folder = args.all
        found = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".onnx") and "_fp16" not in f
        ]
        if not found:
            print(f"No FP32 .onnx files found in '{folder}'.")
            return
        for path in sorted(found):
            convert(path)
            print()
    elif args.input:
        convert(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
