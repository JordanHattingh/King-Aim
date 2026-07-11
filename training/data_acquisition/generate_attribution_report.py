"""Render accepted provenance entries into a stable Markdown attribution report."""

from __future__ import annotations

import argparse
from pathlib import Path

from provenance import load_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dataset attribution report")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [row for row in load_records(args.manifest) if row.get("accepted") and row.get("attribution_text")]
    lines = ["# Dataset Attribution", "", f"Accepted attributed images: {len(rows)}", ""]
    for row in sorted(rows, key=lambda item: str(item.get("image_id"))):
        lines.extend([
            f"## {row['image_id']}", "", str(row["attribution_text"]), "",
            f"- Source: {row.get('source_page') or row.get('source_url') or 'local'}",
            f"- License: {row.get('license')} ({row.get('license_url') or 'recorded permission'})",
            f"- SHA-256: `{row.get('sha256')}`", "",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote {len(rows)} attributions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
