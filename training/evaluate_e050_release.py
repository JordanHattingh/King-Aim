"""Deterministic E050 PT/ONNX deployment-release benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

INPUT_SIZE = 512
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
GT_IOU_THRESHOLD = 0.50
PARITY_IOU_THRESHOLD = 0.99
PARITY_CONFIDENCE_TOLERANCE = 0.002
MAX_DETECTIONS = 300
QUALITY_GATES = {"recall": 0.90, "precision": 0.85, "negative_fp_rate": 0.10}
IMPORTANT_TAGS = ("small", "partially_occluded", "edge", "overlap", "low_contrast", "cluttered_background", "multiple_enemies")
RELEASE_MINIMUMS = {"positive_frames": 25, "negative_frames": 10, "enemy_objects": 40, "important_tag_objects": 5}
REHEARSAL_EXPECTATIONS = {"PV001": (1, None), "PV002": (2, None), "PV003": (0, 0)}


@dataclass(frozen=True)
class Letterbox:
    ratio: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, image_dir: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("images")
    if not isinstance(entries, list):
        raise ValueError("manifest.json must contain an images array")
    if not entries:
        raise ValueError("manifest.json images array must not be empty")
    seen: set[str] = set()
    errors: list[str] = []
    for index, entry in enumerate(entries):
        prefix = f"images[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: entry must be an object")
            continue
        filename = entry.get("file")
        if not isinstance(filename, str) or not filename:
            errors.append(f"{prefix}: missing file")
            continue
        if filename in seen:
            errors.append(f"{prefix}: duplicate filename {filename}")
        seen.add(filename)
        candidate = Path(filename)
        if candidate.is_absolute() or candidate.name != filename or ".." in candidate.parts:
            errors.append(f"{prefix}: file must be a basename without path traversal")
            continue
        image_path = image_dir / filename
        if not image_path.is_file():
            errors.append(f"{prefix}: missing image {filename}")
            continue
        if entry.get("reviewed") is not True:
            errors.append(f"{prefix}: entry is not reviewed")
        if sha256(image_path).lower() != str(entry.get("sha256", "")).lower():
            errors.append(f"{prefix}: changed image hash")
        with Image.open(image_path) as image:
            width, height = image.size
        if entry.get("width") != width or entry.get("height") != height:
            errors.append(f"{prefix}: dimensions do not match image")
        tags = entry.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            errors.append(f"{prefix}: tags must be an array of non-empty strings")
            tags = []
        objects = entry.get("objects")
        if not isinstance(objects, list):
            errors.append(f"{prefix}: objects must be an array")
            continue
        negative = "negative" in tags
        if negative and objects:
            errors.append(f"{prefix}: negative frame contains objects")
        if not negative and not objects:
            errors.append(f"{prefix}: positive frame contains no objects")
        for object_index, obj in enumerate(objects):
            object_prefix = f"{prefix}.objects[{object_index}]"
            if not isinstance(obj, dict):
                errors.append(f"{object_prefix}: object must be an object")
                continue
            box = obj.get("bbox_xyxy")
            if obj.get("class") != "enemy":
                errors.append(f"{object_prefix}: class must be enemy")
            object_tags = obj.get("tags")
            if not isinstance(object_tags, list) or not all(isinstance(tag, str) and tag for tag in object_tags):
                errors.append(f"{object_prefix}: tags must be an array of non-empty strings")
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
                errors.append(f"{object_prefix}: bbox_xyxy must contain four numbers")
                continue
            if not all(np.isfinite(v) for v in box):
                errors.append(f"{object_prefix}: bbox coordinates must be finite")
                continue
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                errors.append(f"{object_prefix}: zero-area box")
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                errors.append(f"{object_prefix}: box outside image bounds")
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))
    stems = [Path(entry["file"]).stem.casefold() for entry in entries]
    if len(stems) != len(set(stems)):
        raise ValueError("Manifest validation failed:\nduplicate review-output stems")
    return manifest


def letterbox(image: Image.Image, size: int = INPUT_SIZE) -> tuple[np.ndarray, Letterbox]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    ratio = min(size / width, size / height)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    pad_x, pad_y = (size - resized_width) / 2, (size - resized_height) / 2
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (round(pad_x - 0.1), round(pad_y - 0.1)))
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), Letterbox(ratio, pad_x, pad_y, width, height)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    intersection_min = np.maximum(a[:, None, :2], b[None, :, :2])
    intersection_max = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(0, intersection_max - intersection_min), axis=2)
    area_a = np.prod(np.maximum(0, a[:, 2:] - a[:, :2]), axis=1)
    area_b = np.prod(np.maximum(0, b[:, 2:] - b[:, :2]), axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def decode(raw: np.ndarray, transform: Letterbox, confidence: float = CONFIDENCE_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    if raw.shape[0:2] != (1, 5):
        raise ValueError(f"Expected raw [1,5,N] detector output, got {raw.shape}")
    scores = raw[0, 4]
    selected = np.flatnonzero(scores >= confidence)
    boxes = raw[0, :4, selected].astype(np.float32)
    if not len(boxes):
        return np.empty((0, 4), np.float32), np.empty(0, np.float32)
    boxes = np.column_stack((boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                             boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2))
    ranked = np.argsort(scores[selected])[::-1]
    order = ranked.copy()
    keep: list[int] = []
    while len(order) and len(keep) < MAX_DETECTIONS:
        keep.append(int(order[0]))
        if len(order) == 1:
            break
        # Match PredictionFilter.ApplyNms: equality suppresses, so only IoU < 0.45 survives.
        mask = box_iou(boxes[order[:1]], boxes[order[1:]])[0] < NMS_IOU_THRESHOLD
        order = order[1:][mask]
    result = boxes[keep]
    result[:, [0, 2]] = (result[:, [0, 2]] - transform.pad_x) / transform.ratio
    result[:, [1, 3]] = (result[:, [1, 3]] - transform.pad_y) / transform.ratio
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, transform.original_width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, transform.original_height)
    return result, scores[selected][keep]


def match(predictions: np.ndarray, scores: np.ndarray, ground_truth: np.ndarray, threshold: float) -> dict:
    unmatched = set(range(len(ground_truth)))
    matches: list[dict] = []
    false_positives: list[int] = []
    ious = box_iou(predictions, ground_truth)
    for prediction in np.argsort(scores)[::-1]:
        candidates = [(float(ious[prediction, gt]), gt) for gt in unmatched if ious[prediction, gt] >= threshold]
        if not candidates:
            false_positives.append(int(prediction))
            continue
        iou, gt = max(candidates)
        unmatched.remove(gt)
        matches.append({"prediction": int(prediction), "ground_truth": gt, "iou": iou})
    return {"matches": matches, "false_positives": false_positives, "false_negatives": sorted(unmatched)}


def draw_review(path: Path, image: Image.Image, gt: np.ndarray, pt: tuple[np.ndarray, np.ndarray], onnx: tuple[np.ndarray, np.ndarray], gt_match: dict) -> None:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for i, box in enumerate(gt):
        draw.rectangle(tuple(box), outline="lime", width=3); draw.text((box[0], box[1]), f"GT {i}", fill="lime")
    matched = {row["prediction"]: row for row in gt_match["matches"]}
    for label, (boxes, scores), color in (("PT", pt, "cyan"), ("ONNX", onnx, "orange")):
        for i, (box, score) in enumerate(zip(boxes, scores)):
            status = "TP" if label == "ONNX" and i in matched else ("FP" if label == "ONNX" else "PRED")
            iou = f" IoU={matched[i]['iou']:.3f}" if i in matched else ""
            draw.rectangle(tuple(box), outline=color, width=2); draw.text((box[0], max(0, box[1] - 13)), f"{label} {score:.3f} {status}{iou}", fill=color)
    for gt_index in gt_match["false_negatives"]:
        box = gt[gt_index]; draw.text((box[0], box[3]), "FN", fill="red")
    canvas.save(path)


def evaluate(manifest_path: Path, image_dir: Path, pt_path: Path, onnx_path: Path, output_dir: Path,
             pt_runner: Callable[[np.ndarray], np.ndarray], onnx_runner: Callable[[np.ndarray], np.ndarray], ultralytics_version: str,
             mode: str = "rehearsal") -> dict:
    manifest = load_manifest(manifest_path, image_dir)
    output_dir.mkdir(parents=True, exist_ok=True); review_dir = output_dir / "review"; review_dir.mkdir(exist_ok=True)
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("pytorch", "onnx")}
    negative_frames = 0; negative_predictions = {"pytorch": 0, "onnx": 0}
    parity_failures: list[str] = []; frames: list[dict] = []; parity_ious: list[float] = []; parity_confidence_differences: list[float] = []
    pt_only = onnx_only = count_disagreement_frames = positive_frames = enemy_objects = 0
    positive_frames_with_pt_detection = positive_frames_with_onnx_detection = 0
    rehearsal_expectation_failures: list[str] = []
    observed_rehearsal_stems: set[str] = set()
    tag_counts = {tag: 0 for tag in IMPORTANT_TAGS}; tag_stats = {tag: {"tp": 0, "fn": 0} for tag in IMPORTANT_TAGS}
    for entry in manifest["images"]:
        image = Image.open(image_dir / entry["file"]); tensor, transform = letterbox(image)
        pt = decode(pt_runner(tensor), transform); onnx = decode(onnx_runner(tensor), transform)
        gt = np.asarray([obj["bbox_xyxy"] for obj in entry["objects"]], dtype=np.float32).reshape(-1, 4)
        results = {"pytorch": match(pt[0], pt[1], gt, GT_IOU_THRESHOLD), "onnx": match(onnx[0], onnx[1], gt, GT_IOU_THRESHOLD)}
        gt_result = results["onnx"]
        for name, result in results.items():
            totals[name]["tp"] += len(result["matches"]); totals[name]["fp"] += len(result["false_positives"]); totals[name]["fn"] += len(result["false_negatives"])
        enemy_objects += len(gt); positive_frames += int(bool(len(gt)))
        if len(gt):
            positive_frames_with_pt_detection += int(bool(len(pt[0])))
            positive_frames_with_onnx_detection += int(bool(len(onnx[0])))
        matched_gt = {row["ground_truth"] for row in gt_result["matches"]}
        for gt_index, obj in enumerate(entry["objects"]):
            for tag in set(obj["tags"]) & set(IMPORTANT_TAGS):
                tag_counts[tag] += 1
                tag_stats[tag]["tp" if gt_index in matched_gt else "fn"] += 1
        if "negative" in entry.get("tags", []):
            negative_frames += 1
            negative_predictions["pytorch"] += int(bool(len(pt[0]))); negative_predictions["onnx"] += int(bool(len(onnx[0])))
        parity = match(onnx[0], onnx[1], pt[0], PARITY_IOU_THRESHOLD)
        pt_only += len(parity["false_negatives"]); onnx_only += len(parity["false_positives"])
        count_disagreement_frames += int(len(pt[0]) != len(onnx[0]))
        parity_ious.extend(row["iou"] for row in parity["matches"])
        parity_confidence_differences.extend(abs(float(onnx[1][row["prediction"]] - pt[1][row["ground_truth"]])) for row in parity["matches"])
        parity_ok = len(pt[0]) == len(onnx[0]) and not parity["false_positives"] and not parity["false_negatives"] and all(abs(float(onnx[1][m["prediction"]] - pt[1][m["ground_truth"]])) <= PARITY_CONFIDENCE_TOLERANCE for m in parity["matches"])
        if not parity_ok: parity_failures.append(entry["file"])
        stem = Path(entry["file"]).stem.upper()
        if mode == "rehearsal" and stem in REHEARSAL_EXPECTATIONS:
            observed_rehearsal_stems.add(stem)
            minimum, maximum = REHEARSAL_EXPECTATIONS[stem]
            count = len(onnx[0])
            if count < minimum or (maximum is not None and count > maximum):
                rehearsal_expectation_failures.append(f"{stem}: expected {minimum if maximum is None else f'{minimum}..{maximum}'} ONNX detections, observed {count}")
        draw_review(review_dir / f"{Path(entry['file']).stem}.png", image, gt, pt, onnx, gt_result)
        frames.append({"file": entry["file"], "tp": len(gt_result["matches"]), "fp": len(gt_result["false_positives"]), "fn": len(gt_result["false_negatives"]), "parity": parity_ok})
    if mode == "rehearsal":
        for missing in sorted(set(REHEARSAL_EXPECTATIONS) - observed_rehearsal_stems):
            rehearsal_expectation_failures.append(f"{missing}: required rehearsal frame is missing")
    quality = {}
    for name in ("pytorch", "onnx"):
        value = totals[name]; recall = value["tp"] / max(1, value["tp"] + value["fn"]); precision = value["tp"] / max(1, value["tp"] + value["fp"]); negative_rate = negative_predictions[name] / max(1, negative_frames)
        quality[name] = {**value, "recall": recall, "precision": precision, "negative_fp_rate": negative_rate, "status": "PASS" if recall >= QUALITY_GATES["recall"] and precision >= QUALITY_GATES["precision"] and negative_rate <= QUALITY_GATES["negative_fp_rate"] else "FAIL"}
    composition_failures = ([name for name, actual in (("positive_frames", positive_frames), ("negative_frames", negative_frames), ("enemy_objects", enemy_objects)) if actual < RELEASE_MINIMUMS[name]] + [f"tag:{tag}" for tag, count in tag_counts.items() if count < RELEASE_MINIMUMS["important_tag_objects"]])
    tags = {tag: {"objects": tag_counts[tag], "recall": tag_stats[tag]["tp"] / max(1, tag_stats[tag]["tp"] + tag_stats[tag]["fn"]), "gate": tag_counts[tag] >= 5, "status": "INFORMATIONAL" if tag_counts[tag] < 5 else ("PASS" if tag_stats[tag]["tp"] / max(1, tag_stats[tag]["tp"] + tag_stats[tag]["fn"]) >= QUALITY_GATES["recall"] else "FAIL")} for tag in IMPORTANT_TAGS}
    failed_tag_gates = [tag for tag, result in tags.items() if result["gate"] and result["status"] == "FAIL"]
    parity_status = "PASS" if not parity_failures else "FAIL"
    parity_match_count = len(parity_ious)
    rehearsal_coverage_ok = parity_match_count >= 3 and positive_frames_with_onnx_detection > 0 and positive_frames_with_pt_detection > 0 and not rehearsal_expectation_failures
    rehearsal_status = "PASS" if parity_status == "PASS" and rehearsal_coverage_ok else "FAIL"
    release_status = "PASS" if parity_status == quality["onnx"]["status"] == "PASS" and not composition_failures and not failed_tag_gates else "FAIL"
    report = {"mode": mode, "benchmark_scope": "detector_export_path_not_full_king_aim_runtime_filters", "status": rehearsal_status if mode == "rehearsal" else release_status, "frozen_contract": {"confidence_threshold": CONFIDENCE_THRESHOLD, "nms_iou_threshold": NMS_IOU_THRESHOLD, "nms_boundary": "suppress_iou_greater_than_or_equal", "input_size": INPUT_SIZE, "letterbox_mode": "Python detector benchmark; not bit-identical to MathUtil.LetterboxResize", "maximum_detections": MAX_DETECTIONS, "ultralytics_version": ultralytics_version, "pytorch_sha256": sha256(pt_path), "onnx_sha256": sha256(onnx_path), "manifest_sha256": sha256(manifest_path)}, "rehearsal_coverage": {"status": "PASS" if rehearsal_coverage_ok else "FAIL", "parity_match_count": parity_match_count, "required_parity_match_count": 3, "positive_frames_with_pt_detection": positive_frames_with_pt_detection, "positive_frames_with_onnx_detection": positive_frames_with_onnx_detection, "expectation_failures": rehearsal_expectation_failures}, "parity": {"status": parity_status, "parity_iou_threshold": PARITY_IOU_THRESHOLD, "parity_confidence_tolerance": PARITY_CONFIDENCE_TOLERANCE, "observed_minimum_matched_iou": min(parity_ious) if parity_ious else None, "observed_mean_matched_iou": float(np.mean(parity_ious)) if parity_ious else None, "observed_maximum_confidence_difference": max(parity_confidence_differences) if parity_confidence_differences else None, "observed_mean_confidence_difference": float(np.mean(parity_confidence_differences)) if parity_confidence_differences else None, "pt_only_detection_count": pt_only, "onnx_only_detection_count": onnx_only, "count_disagreement_frames": count_disagreement_frames, "failed_frames": parity_failures}, "composition": {"positive_frames": positive_frames, "negative_frames": negative_frames, "enemy_objects": enemy_objects, "failures": composition_failures}, "quality": quality, "tags": tags, "failed_tag_gates": failed_tag_gates, "frames": frames}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen E050 deployment release gate")
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--pt", required=True, type=Path); parser.add_argument("--onnx", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("rehearsal", "release"), default="rehearsal")
    args = parser.parse_args()
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO
    model = YOLO(str(args.pt)).model.eval().cpu(); session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"]); input_name = session.get_inputs()[0].name
    def run_pt(tensor: np.ndarray) -> np.ndarray:
        with torch.no_grad(): result = model(torch.from_numpy(tensor))
        if isinstance(result, (tuple, list)): result = result[0]
        return result.detach().cpu().numpy()
    report = evaluate(args.manifest, args.images, args.pt, args.onnx, args.output_dir, run_pt, lambda tensor: session.run(None, {input_name: tensor})[0], ultralytics.__version__, args.mode)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
