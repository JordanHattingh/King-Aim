"""Strict YOLO four-keypoint annotation and split-leakage auditor."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def audit(root: Path, provenance: Path | None = None) -> list[dict]:
    issues: list[dict] = []
    image_by_key: dict[tuple[str, str], Path] = {}
    label_by_key: dict[tuple[str, str], Path] = {}
    for split in SPLITS:
        for image in (root / "images" / split).rglob("*") if (root / "images" / split).is_dir() else ():
            if image.suffix.lower() in IMAGE_SUFFIXES:
                image_by_key[(split, image.stem)] = image
        for label in (root / "labels" / split).rglob("*.txt") if (root / "labels" / split).is_dir() else ():
            label_by_key[(split, label.stem)] = label
    for key, image in image_by_key.items():
        if key not in label_by_key:
            issues.append({"severity": "error", "code": "missing_label", "file": str(image), "detail": "image has no label file"})
    for key, label in label_by_key.items():
        if key not in image_by_key:
            issues.append({"severity": "error", "code": "missing_image", "file": str(label), "detail": "label has no image"})
        seen: set[tuple[float, ...]] = set()
        for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                values = tuple(float(value) for value in raw.split())
            except ValueError:
                issues.append({"severity": "error", "code": "malformed", "file": str(label), "line": line_number, "detail": "non-numeric value"})
                continue
            if len(values) != 17:
                issues.append({"severity": "error", "code": "field_count", "file": str(label), "line": line_number, "detail": f"expected 17 fields, got {len(values)}"})
                continue
            if values[0] != 0:
                issues.append({"severity": "error", "code": "class_id", "file": str(label), "line": line_number, "detail": "only class 0 human is allowed"})
            cx, cy, width, height = values[1:5]
            if any(value < 0 or value > 1 for value in values[1:5]) or width <= 0 or height <= 0:
                issues.append({"severity": "error", "code": "invalid_box", "file": str(label), "line": line_number, "detail": "box must be normalized with positive size"})
            if width * height < 0.000025:
                issues.append({"severity": "warning", "code": "tiny_box", "file": str(label), "line": line_number, "detail": "box area below 0.0025%"})
            for index in range(4):
                x, y, visibility = values[5 + index * 3:8 + index * 3]
                if visibility not in (0, 1, 2):
                    issues.append({"severity": "error", "code": "visibility", "file": str(label), "line": line_number, "detail": f"keypoint {index} visibility must be 0, 1, or 2"})
                if visibility and not (0 <= x <= 1 and 0 <= y <= 1):
                    issues.append({"severity": "error", "code": "keypoint_bounds", "file": str(label), "line": line_number, "detail": f"keypoint {index} outside normalized image"})
                margin_x, margin_y = width * 0.5, height * 0.5
                if visibility and (abs(x - cx) > margin_x * 1.5 or abs(y - cy) > margin_y * 1.5):
                    issues.append({"severity": "warning", "code": "keypoint_far_from_box", "file": str(label), "line": line_number, "detail": f"keypoint {index} far outside box"})
            if values in seen:
                issues.append({"severity": "error", "code": "duplicate_annotation", "file": str(label), "line": line_number, "detail": "duplicate row"})
            seen.add(values)
    if provenance and provenance.exists():
        sessions: dict[str, set[str]] = defaultdict(set)
        for line_number, line in enumerate(provenance.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("dataset_split"):
                    sessions[str(row["session_id"])].add(str(row["dataset_split"]))
            except (json.JSONDecodeError, KeyError) as exc:
                issues.append({"severity": "error", "code": "provenance", "file": str(provenance), "line": line_number, "detail": str(exc)})
        for session, splits in sessions.items():
            if len(splits) > 1:
                issues.append({"severity": "error", "code": "session_leakage", "file": str(provenance), "detail": f"{session} appears in {sorted(splits)}"})
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit King Aim YOLO pose annotations")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    issues = audit(args.dataset, args.provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "annotation_audit.json").write_text(json.dumps(issues, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "annotation_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("severity", "code", "file", "line", "detail"), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(issues)
    counts = Counter(issue["severity"] for issue in issues)
    summary = f"errors={counts['error']}\nwarnings={counts['warning']}\nstatus={'FAIL' if counts['error'] or (args.fail_on_warning and counts['warning']) else 'PASS'}\n"
    (args.output_dir / "annotation_audit_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 1 if counts["error"] or (args.fail_on_warning and counts["warning"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
