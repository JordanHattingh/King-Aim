"""Import selected Open Images V7 candidates from the official metadata CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from download_utils import download_image, store_image
from provenance import ProvenanceRecord, append_record, imported_now


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire Open Images candidates using official metadata")
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--image-ids", required=True, type=Path, help="One Open Images ID per line")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=360)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    wanted = {line.strip().split("/")[-1] for line in args.image_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
    imported = 0
    with args.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("ImageID") not in wanted or imported >= args.max_images:
                continue
            url = row.get("OriginalURL") or row.get("Thumbnail300KURL")
            if not url:
                continue
            if args.dry_run:
                print(f"{row['ImageID']} | {row.get('License')} | {row.get('OriginalLandingURL')}"); imported += 1; continue
            try:
                data = download_image(url)
                destination, digest, perceptual, width, height = store_image(args.output_dir, f"openimages-{row['ImageID']}", data, args.min_width, args.min_height)
                append_record(args.manifest, ProvenanceRecord(
                    image_id=digest[:24], local_filename=str(destination.resolve()), source_type="open_images_v7",
                    source_url=url, source_page=row.get("OriginalLandingURL"), creator=row.get("Author") or None,
                    license="CC BY 2.0", license_url=row.get("License") or None, permission_evidence=None,
                    attribution_text=f"{row.get('Title') or row['ImageID']} — {row.get('Author') or 'unknown'}",
                    imported_at_utc=imported_now(), sha256=digest, perceptual_hash=perceptual, width=width, height=height,
                    game_category=None, session_id=f"openimages-{row['ImageID']}", accepted=False,
                    rejection_reason="pending manual review", dataset_split=None,
                )); imported += 1
            except Exception as exc:
                print(f"warning: {row['ImageID']}: {exc}")
    print(f"Collected {imported} Open Images candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
