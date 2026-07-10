"""
King Aim — GRU training data preparation (feature schema track-motion-8x8-v2).

The split is SESSION-GROUPED. Windows/tracks from one recording session are never
spread between train/validation/test. This prevents map/HUD/capture-condition
leakage that made the old random sequence split over-optimistic.

Usage:
    python prepare_gru_data.py --logs-dir runs/logs --out-dir data/tracks
    python prepare_gru_data.py --logs-dir runs/logs --val-fraction 0.15 --test-fraction 0.15
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from contracts import GRU_FRAME_FIELDS, TEMPORAL_FEATURE_SCHEMA


@dataclass(frozen=True)
class SessionSequences:
    session_id: str
    source_path: Path
    sequences: list[dict]


def _load_sequence_file(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL {path}:{line_number}: {exc}") from exc
        return records

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def load_sessions(logs_dir: Path, min_frames: int) -> list[SessionSequences]:
    sessions: list[SessionSequences] = []
    session_dirs = sorted(logs_dir.glob("session_*"))
    if not session_dirs:
        print(f"[WARN] No session_* folders found under {logs_dir}")
        return sessions

    for session_dir in session_dirs:
        jsonl_path = session_dir / "gru_sequences.jsonl"
        legacy_path = session_dir / "gru_sequences.json"
        path = jsonl_path if jsonl_path.exists() else legacy_path
        if not path.exists():
            continue

        data = _load_sequence_file(path)
        kept = [s for s in data if len(s.get("frames", [])) >= min_frames]
        session_id = path.parent.name
        print(
            f"  {session_id}: {len(data)} tracks -> "
            f"{len(kept)} kept (>={min_frames} frames)"
        )
        if kept:
            sessions.append(SessionSequences(session_id, path, kept))

    return sessions


def split_sessions(
    sessions: list[SessionSequences],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[SessionSequences]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation + test fractions must be < 1.0")
    if len(sessions) < 2 and val_fraction > 0:
        raise ValueError(
            "At least two non-empty recording sessions are required for a grouped "
            "train/validation split. Collect another session; do not random-split windows."
        )
    if len(sessions) < 3 and test_fraction > 0:
        raise ValueError(
            "At least three non-empty recording sessions are required when --test-fraction > 0."
        )

    shuffled = list(sessions)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)

    n_test = math.ceil(count * test_fraction) if test_fraction > 0 else 0
    n_val = math.ceil(count * val_fraction) if val_fraction > 0 else 0

    # Keep at least one train session.
    while n_test + n_val >= count:
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return {"train": train, "val": val, "test": test}


def write_split(out_root: Path, name: str, sessions: list[SessionSequences]) -> dict:
    split_dir = out_root / name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    sequence_count = 0
    manifest_sessions: list[dict] = []
    for session in sessions:
        out_path = split_dir / f"{session.session_id}__gru_sequences.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(session.sequences, handle, separators=(",", ":"))
        sequence_count += len(session.sequences)
        manifest_sessions.append(
            {
                "session_id": session.session_id,
                "source": str(session.source_path),
                "sequence_count": len(session.sequences),
                "output": out_path.name,
            }
        )

    print(f"  {name}: {len(sessions)} sessions / {sequence_count} sequences -> {split_dir}")
    return {
        "name": name,
        "session_count": len(sessions),
        "sequence_count": sequence_count,
        "sessions": manifest_sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare session-grouped GRU training data")
    parser.add_argument("--logs-dir", default="runs/logs")
    parser.add_argument("--out-dir", default="data/tracks")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument(
        "--min-frames",
        type=int,
        default=9,
        help="Minimum frames per track. Eight input frames + one target frame require >=9.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_root = Path(args.out_dir)
    print(f"Loading sessions from {logs_dir} (min_frames={args.min_frames})")
    sessions = load_sessions(logs_dir, args.min_frames)
    if not sessions:
        raise SystemExit("No usable sessions loaded. Enable detection logging and collect real sessions first.")

    split = split_sessions(sessions, args.val_fraction, args.test_fraction, args.seed)
    out_root.mkdir(parents=True, exist_ok=True)
    split_summaries = [write_split(out_root, name, split[name]) for name in ("train", "val", "test")]

    manifest = {
        "schema_version": 2,
        "feature_schema": TEMPORAL_FEATURE_SCHEMA,
        "split_strategy": "session_grouped",
        "seed": args.seed,
        "min_frames": args.min_frames,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "splits": split_summaries,
    }
    with (out_root / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nDone. Session IDs are isolated by split.")
    print("Next: python train_gru.py")


if __name__ == "__main__":
    main()
