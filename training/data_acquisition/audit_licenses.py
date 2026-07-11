"""Fail closed when accepted provenance records lack an approved usage basis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from provenance import load_records

DEFAULT_ALLOWLIST = {"CC0", "Public Domain", "CC BY 2.0", "CC BY 3.0", "CC BY 4.0", "CC BY-SA 4.0", "explicit written permission", "self-captured"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit accepted King Aim image provenance")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--allow", action="append", default=[])
    args = parser.parse_args()
    allowlist = DEFAULT_ALLOWLIST | set(args.allow)
    problems: list[dict] = []
    records = load_records(args.manifest)
    for row in records:
        if not row.get("accepted"):
            continue
        license_name = str(row.get("license") or "")
        if license_name not in allowlist:
            problems.append({"image_id": row.get("image_id"), "problem": "license_not_allowed", "license": license_name})
        if license_name == "explicit written permission" and not row.get("permission_evidence"):
            problems.append({"image_id": row.get("image_id"), "problem": "missing_permission_evidence"})
        if license_name.startswith("CC BY") and not row.get("attribution_text"):
            problems.append({"image_id": row.get("image_id"), "problem": "missing_attribution"})
        for field in ("sha256", "session_id", "local_filename"):
            if not row.get(field):
                problems.append({"image_id": row.get("image_id"), "problem": f"missing_{field}"})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"records": len(records), "problems": problems}, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(records)} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
