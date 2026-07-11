"""Detect exact and perceptually near-duplicate dataset images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def average_hash(path: Path, size: int = 16) -> int:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((size, size)).getdata())
    mean = sum(pixels) / len(pixels)
    return sum((1 << index) for index, value in enumerate(pixels) if value >= mean)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report exact and near duplicate images")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--distance", type=int, default=8)
    args = parser.parse_args()
    images = sorted(path for path in args.input.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    exact: dict[str, Path] = {}
    unique_hashes: list[tuple[Path, int]] = []
    rows: list[dict] = []
    for image in images:
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in exact:
            rows.append({"path": str(image), "classification": "Exact duplicate", "duplicate_of": str(exact[digest]), "distance": 0})
            continue
        exact[digest] = image
        fingerprint = average_hash(image)
        nearest = min(((fingerprint ^ other_hash).bit_count(), other) for other, other_hash in unique_hashes) if unique_hashes else None
        if nearest and nearest[0] <= args.distance:
            classification = "Near duplicate" if nearest[0] > 0 else "Exact visual duplicate"
            rows.append({"path": str(image), "classification": classification, "duplicate_of": str(nearest[1]), "distance": nearest[0]})
        else:
            rows.append({"path": str(image), "classification": "Unique", "duplicate_of": None, "distance": None})
            unique_hashes.append((image, fingerprint))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(images)} images; flagged {sum(row['classification'] != 'Unique' for row in rows)} duplicates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
