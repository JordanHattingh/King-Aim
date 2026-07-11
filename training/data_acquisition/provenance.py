"""JSONL provenance records used by every King Aim dataset importer."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ProvenanceRecord:
    image_id: str
    local_filename: str
    source_type: str
    source_url: str | None
    source_page: str | None
    creator: str | None
    license: str
    license_url: str | None
    permission_evidence: str | None
    attribution_text: str | None
    imported_at_utc: str
    sha256: str
    perceptual_hash: str | None
    width: int
    height: int
    game_category: str | None
    session_id: str
    accepted: bool
    rejection_reason: str | None
    dataset_split: str | None


def append_record(path: Path, record: ProvenanceRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def imported_now() -> str:
    return datetime.now(timezone.utc).isoformat()
