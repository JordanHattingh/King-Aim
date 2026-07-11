"""Import reviewed explicit image URLs described by a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from download_utils import download_image, store_image
from provenance import ProvenanceRecord, append_record, imported_now

REQUIRED = {"url", "source_page", "creator", "license", "license_url", "attribution_text", "session_id"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import explicit reviewed URLs into candidate storage")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=360)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        rows = list(reader)[: args.max_images]
    if args.dry_run:
        print(f"Validated {len(rows)} explicit URL candidates")
        return 0
    imported = 0
    for index, row in enumerate(rows, 1):
        try:
            data = download_image(row["url"])
            destination, digest, perceptual, width, height = store_image(args.output_dir, f"explicit-{index:05d}", data, args.min_width, args.min_height)
            append_record(args.manifest, ProvenanceRecord(
                image_id=digest[:24], local_filename=str(destination.resolve()), source_type="explicit_url",
                source_url=row["url"], source_page=row["source_page"], creator=row["creator"], license=row["license"],
                license_url=row["license_url"], permission_evidence=row.get("permission_evidence") or None,
                attribution_text=row["attribution_text"], imported_at_utc=imported_now(), sha256=digest,
                perceptual_hash=perceptual, width=width, height=height, game_category=row.get("game_category") or None,
                session_id=row["session_id"], accepted=False, rejection_reason="pending manual review", dataset_split=None,
            ))
            imported += 1
        except Exception as exc:
            print(f"warning: {row['url']}: {exc}")
    print(f"Imported {imported}/{len(rows)} explicit candidates")
    return 0 if imported == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
