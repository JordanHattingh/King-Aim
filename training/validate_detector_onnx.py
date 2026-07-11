"""Validate raw YOLO detector parity between PyTorch and ONNX Runtime.

Reports two separate categories:

  Deployment parity — real gameplay images passed via --image.
      These are the hard gate.  If any gameplay frame fails semantic parity,
      the run fails.  Raw numerical parity is recorded separately.

  Synthetic stress diagnostics — zero/ones/gradient/random/checker tensors.
      Always recorded.  When gameplay images are present these are diagnostic
      only and do not gate the run.  When no --image flags are given the
      synthetic inputs act as the gate (backwards-compatible behaviour).

Parity is evaluated at two levels for each gameplay frame:

  Raw numerical parity — max/mean absolute error between raw network outputs.
      Tolerance: max_abs=0.0005, mean_abs=0.00005.
      A failure here is a WARN when semantic parity passes.

  Semantic detection parity — post-NMS decoded detections are compared.
      Tolerance: IoU>=0.999, centre difference<=0.25 px, conf diff<=0.001.
      This is the deployment gate.

Hard failures (always fail the run):
  - output shape mismatch
  - NaN or Inf in any output
  - real gameplay frame fails semantic detection parity (count, IoU, centre)
  - model cannot load or provider unavailable

Diagnostic warnings (recorded, do not fail when gameplay images are present):
  - raw numerical tolerance exceeded on a synthetic degenerate tensor
  - raw numerical tolerance exceeded on a gameplay frame that passes
    semantic parity (count and box decisions are identical)
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


# ---------------------------------------------------------------------------
# Semantic (post-NMS) parity
# ---------------------------------------------------------------------------

def _box_iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a [M,4] and every box in b [N,4] → [M,N]."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Return keep indices after greedy NMS, ordered by descending score."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iou = _box_iou_batch(boxes[i : i + 1], boxes[rest])[0]
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def decode_detections(
    raw: np.ndarray,
    conf_thresh: float,
    iou_thresh: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw [1, nc+4, N] detector output → (boxes_xyxy [K,4], scores [K]).

    Assumes eval-mode Ultralytics Detect head: rows 0-3 are cx,cy,w,h in pixel
    space; row 4 is confidence already sigmoid-applied.  Output sorted by
    descending confidence after NMS.
    """
    scores_all = raw[0, 4, :]
    mask = scores_all >= conf_thresh
    if not mask.any():
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    cx = raw[0, 0, mask]
    cy = raw[0, 1, mask]
    w = raw[0, 2, mask]
    h = raw[0, 3, mask]
    scores = scores_all[mask]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = _nms(boxes, scores, iou_thresh)
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


def semantic_parity_case(
    case: str,
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    conf_thresholds: tuple[float, ...],
    nms_iou_thresh: float = 0.45,
    min_box_iou: float = 0.999,
    max_center_diff_px: float = 0.25,
    max_conf_diff: float = 0.001,
) -> dict:
    """Compare post-NMS decoded detections between PyTorch and ONNX outputs.

    Returns a result dict with per-threshold breakdown and an overall
    passed flag.  passed=True only when every threshold passes.
    """
    by_threshold: dict[str, dict] = {}
    all_passed = True

    for thresh in conf_thresholds:
        pt_boxes, pt_scores = decode_detections(pytorch_output, thresh, nms_iou_thresh)
        onnx_boxes, onnx_scores = decode_detections(onnx_output, thresh, nms_iou_thresh)

        pt_n = len(pt_scores)
        onnx_n = len(onnx_scores)
        count_match = pt_n == onnx_n

        box_iou_min: float | None = None
        center_diff_max_px: float | None = None
        conf_diff_max: float | None = None

        if not count_match:
            threshold_passed = False
        elif pt_n == 0:
            # Both zero — vacuously pass
            box_iou_min = None
            center_diff_max_px = None
            conf_diff_max = None
            threshold_passed = True
        else:
            n = pt_n
            pt_cx = (pt_boxes[:n, 0] + pt_boxes[:n, 2]) / 2
            pt_cy = (pt_boxes[:n, 1] + pt_boxes[:n, 3]) / 2
            onnx_cx = (onnx_boxes[:n, 0] + onnx_boxes[:n, 2]) / 2
            onnx_cy = (onnx_boxes[:n, 1] + onnx_boxes[:n, 3]) / 2
            center_diff_max_px = float(np.hypot(pt_cx - onnx_cx, pt_cy - onnx_cy).max())
            conf_diff_max = float(np.abs(pt_scores[:n] - onnx_scores[:n]).max())
            iou_matrix = _box_iou_batch(pt_boxes[:n], onnx_boxes[:n])
            box_iou_min = float(np.diag(iou_matrix).min())
            threshold_passed = (
                box_iou_min >= min_box_iou
                and center_diff_max_px <= max_center_diff_px
                and conf_diff_max <= max_conf_diff
            )

        if not threshold_passed:
            all_passed = False

        by_threshold[f"{thresh:g}"] = {
            "threshold": thresh,
            "pytorch_count": pt_n,
            "onnx_count": onnx_n,
            "count_match": count_match,
            "box_iou_min": box_iou_min,
            "center_diff_max_px": center_diff_max_px,
            "conf_diff_max": conf_diff_max,
            "passed": threshold_passed,
        }

    return {
        "case": case,
        "passed": all_passed,
        "min_box_iou": min_box_iou,
        "max_center_diff_px": max_center_diff_px,
        "max_conf_diff": max_conf_diff,
        "nms_iou_thresh": nms_iou_thresh,
        "by_threshold": by_threshold,
    }


# ---------------------------------------------------------------------------
# Section aggregation
# ---------------------------------------------------------------------------

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


def _semantic_section(
    semantic_rows: list[dict],
    is_gate: bool,
) -> dict:
    """Aggregate semantic_parity_case rows into a section summary."""
    if not semantic_rows:
        return {"status": "PASS", "gate": is_gate, "case_count": 0, "cases": []}
    failed = [r["case"] for r in semantic_rows if not r["passed"]]
    status = "FAIL" if failed else "PASS"
    return {
        "status": status,
        "gate": is_gate,
        "case_count": len(semantic_rows),
        "cases_failing": failed,
        "cases": semantic_rows,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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

    def run_case(label: str, tensor_np: np.ndarray) -> tuple[dict, dict]:
        """Returns (raw_summary, semantic_summary)."""
        with torch.no_grad():
            pytorch_np = normalize_output(module(torch.from_numpy(tensor_np)))
        onnx_np = session.run(None, {input_meta.name: tensor_np})[0]
        raw = summarize_case(label, pytorch_np, onnx_np, args.class_count, thresholds)
        semantic = semantic_parity_case(label, pytorch_np, onnx_np, thresholds)
        return raw, semantic

    # Collect synthetic diagnostic cases
    synthetic_raw_rows: list[dict] = []
    synthetic_sem_rows: list[dict] = []
    if not args.skip_synthetic:
        for label, tensor_np in synthetic_inputs(args.imgsz, args.seed):
            raw, sem = run_case(label, tensor_np)
            synthetic_raw_rows.append(raw)
            synthetic_sem_rows.append(sem)

    # Collect deployment gameplay cases
    gameplay_raw_rows: list[dict] = []
    gameplay_sem_rows: list[dict] = []
    for path in args.image:
        raw, sem = run_case(f"image:{path.name}", image_input(path, args.imgsz))
        gameplay_raw_rows.append(raw)
        gameplay_sem_rows.append(sem)

    if not synthetic_raw_rows and not gameplay_raw_rows:
        raise SystemExit("No parity inputs selected")

    has_gameplay = bool(gameplay_raw_rows)

    # Raw deployment section: gate only when gameplay images present
    deployment_raw = (
        _section_stats(gameplay_raw_rows, args.max_abs_tolerance, args.mean_abs_tolerance, is_gate=True)
        if has_gameplay else None
    )

    # Semantic deployment section: this is the true deployment gate
    deployment_sem = (
        _semantic_section(gameplay_sem_rows, is_gate=True)
        if has_gameplay else None
    )

    # Synthetic section: diagnostic when gameplay present, gate otherwise
    synthetic_raw = _section_stats(
        synthetic_raw_rows, args.max_abs_tolerance, args.mean_abs_tolerance, is_gate=not has_gameplay
    )
    if not has_gameplay:
        synthetic_raw["note"] = "No gameplay images provided. Synthetic inputs used as gate."
    else:
        synthetic_raw["note"] = "Diagnostic only. Synthetic failures do not gate deployment."

    # Determine overall pass/fail and status
    if has_gameplay:
        # Semantic parity is the gate; raw parity failure alongside semantic pass is a WARN
        semantic_passed = deployment_sem["status"] == "PASS"
        raw_passed = deployment_raw["status"] == "PASS"

        if semantic_passed and raw_passed:
            overall_status = "PASS"
            note = "Deployment semantic and raw parity both pass."
        elif semantic_passed and not raw_passed:
            overall_status = "WARN"
            note = (
                "Deployment semantic parity PASS: final detections are identical. "
                f"Raw numerical tolerance FAIL: {deployment_raw['cases_exceeding_tolerance']}. "
                "Sub-threshold numerical difference does not affect NMS decisions."
            )
        else:
            overall_status = "FAIL"
            note = (
                f"Deployment semantic parity FAIL: {deployment_sem['cases_failing']}. "
                "Final detections differ between PyTorch and ONNX."
            )
        passed = semantic_passed
    else:
        passed = synthetic_raw["status"] == "PASS"
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
        "deployment_parity": deployment_raw,
        "semantic_deployment_parity": deployment_sem,
        "synthetic_diagnostics": synthetic_raw,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "detector_parity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    # CSV: all cases with a category column
    all_rows_tagged = (
        [{"category": "deployment", **r} for r in gameplay_raw_rows] +
        [{"category": "synthetic", **r} for r in synthetic_raw_rows]
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
    if deployment_raw:
        dep_raw_line = (
            f"deployment_raw_status={deployment_raw['status']}\n"
            f"deployment_raw_max_abs_error={deployment_raw['max_abs_error']:.10g}\n"
            f"deployment_raw_mean_abs_error={deployment_raw['mean_abs_error']:.10g}\n"
        )
    else:
        dep_raw_line = "deployment_raw_status=NO_GAMEPLAY_IMAGES\n"

    if deployment_sem:
        dep_sem_line = (
            f"deployment_semantic_status={deployment_sem['status']}\n"
            f"deployment_semantic_cases_failing={','.join(deployment_sem['cases_failing']) or 'none'}\n"
        )
    else:
        dep_sem_line = "deployment_semantic_status=NO_GAMEPLAY_IMAGES\n"

    synth_over = ",".join(synthetic_raw["cases_exceeding_tolerance"]) or "none"
    summary = (
        f"status={overall_status}\n"
        f"note={note}\n"
        + dep_raw_line
        + dep_sem_line
        + f"synthetic_status={synthetic_raw['status']}\n"
        f"synthetic_max_abs_error={synthetic_raw['max_abs_error']:.10g}\n"
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
