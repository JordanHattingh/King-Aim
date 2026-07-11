"""Collect Wikimedia Commons candidates through the MediaWiki Action API."""

from __future__ import annotations

import argparse
import html
import json
import urllib.parse
import urllib.request
from pathlib import Path

from download_utils import USER_AGENT, download_image, store_image
from provenance import ProvenanceRecord, append_record, imported_now

API = "https://commons.wikimedia.org/w/api.php"
LICENSE_MAP = {"CC0": "CC0", "Public domain": "Public Domain", "CC BY 4.0": "CC BY 4.0", "CC BY-SA 4.0": "CC BY-SA 4.0", "CC BY 3.0": "CC BY 3.0"}


def metadata_value(metadata: dict, key: str) -> str:
    raw = str((metadata.get(key) or {}).get("value") or "")
    return html.unescape(__import__("re").sub(r"<[^>]+>", " ", raw)).strip()


def search(query: str, limit: int) -> list[dict]:
    params = {
        "action": "query", "format": "json", "formatversion": "2", "generator": "search",
        "gsrsearch": query, "gsrnamespace": "6", "gsrlimit": str(min(limit, 50)), "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|AttributionRequired|UsageTerms",
    }
    request = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response).get("query", {}).get("pages", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire reviewed Wikimedia Commons image candidates")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=25)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=360)
    parser.add_argument("--license-allowlist", nargs="*", default=list(LICENSE_MAP.values()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pages = search(args.query, args.max_images)
    accepted = 0
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        metadata = info.get("extmetadata") or {}
        raw_license = metadata_value(metadata, "LicenseShortName")
        license_name = LICENSE_MAP.get(raw_license, raw_license)
        if license_name not in args.license_allowlist:
            continue
        if args.dry_run:
            print(f"{page.get('title')} | {license_name} | {info.get('descriptionurl')}")
            accepted += 1
            continue
        try:
            data = download_image(info["url"])
            destination, digest, perceptual, width, height = store_image(args.output_dir, page.get("title", "commons"), data, args.min_width, args.min_height)
            creator = metadata_value(metadata, "Artist")
            credit = metadata_value(metadata, "Credit")
            append_record(args.manifest, ProvenanceRecord(
                image_id=digest[:24], local_filename=str(destination.resolve()), source_type="wikimedia_commons",
                source_url=info["url"], source_page=info.get("descriptionurl"), creator=creator or None,
                license=license_name, license_url=metadata_value(metadata, "LicenseUrl") or None,
                permission_evidence=None, attribution_text=credit or creator or page.get("title"), imported_at_utc=imported_now(),
                sha256=digest, perceptual_hash=perceptual, width=width, height=height, game_category=None,
                session_id=f"commons-{page.get('pageid')}", accepted=False, rejection_reason="pending manual review", dataset_split=None,
            )); accepted += 1
        except Exception as exc:
            print(f"warning: {page.get('title')}: {exc}")
    print(f"Collected {accepted} Wikimedia candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
