"""Assign entire source sessions to deterministic train/val/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from provenance import load_records


def split_for(session_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{session_id}".encode()).hexdigest()[:16], 16) / 0xFFFFFFFFFFFFFFFF
    return "train" if value < 0.70 else "val" if value < 0.85 else "test"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create leakage-safe grouped split assignments")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = load_records(args.manifest)
    assignments = {session: split_for(session, args.seed) for session in sorted({str(row["session_id"]) for row in records})}
    output = [{**row, "dataset_split": assignments[str(row["session_id"])]} for row in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({split: sum(value == split for value in assignments.values()) for split in ("train", "val", "test")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
