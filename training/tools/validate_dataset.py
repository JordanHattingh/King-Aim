#!/usr/bin/env python3
"""
Validate King Aim GRU JSONL track logs without third-party dependencies.

The tool scans recursively for *.jsonl files, accepts the TrackLogger
`gru_sequences.jsonl` schema, and prints a compact dataset-quality summary.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetSummary:
    total_sequences: int
    total_frames: int
    below_gru_minimum: int
    no_occlusion_diversity: int
    delta_cx: tuple[float, ...]
    delta_cy: tuple[float, ...]
    class_counts: Counter[str]


def iter_sequences(dataset_dir: Path) -> Iterable[dict]:
    for path in sorted(dataset_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"warning: {path}:{line_number}: invalid JSON: {exc}", file=sys.stderr)
                    continue

                frames = record.get("frames")
                if isinstance(frames, list):
                    yield record


def summarize_dataset(dataset_dir: Path) -> DatasetSummary:
    total_sequences = 0
    total_frames = 0
    below_gru_minimum = 0
    no_occlusion_diversity = 0
    delta_cx: list[float] = []
    delta_cy: list[float] = []
    class_counts: Counter[str] = Counter()

    for sequence in iter_sequences(dataset_dir):
        frames = sequence.get("frames", [])
        total_sequences += 1
        total_frames += len(frames)
        class_counts[str(sequence.get("class_name", "unknown"))] += 1

        if len(frames) < 8:
            below_gru_minimum += 1

        observed_values = {
            int(float(frame.get("observed", 0)))
            for frame in frames
            if isinstance(frame, dict)
        }
        if len(observed_values) <= 1:
            no_occlusion_diversity += 1

        previous: dict | None = None
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            if previous is not None:
                try:
                    delta_cx.append(float(frame["cx"]) - float(previous["cx"]))
                    delta_cy.append(float(frame["cy"]) - float(previous["cy"]))
                except (KeyError, TypeError, ValueError):
                    pass
            previous = frame

    return DatasetSummary(
        total_sequences=total_sequences,
        total_frames=total_frames,
        below_gru_minimum=below_gru_minimum,
        no_occlusion_diversity=no_occlusion_diversity,
        delta_cx=tuple(delta_cx),
        delta_cy=tuple(delta_cy),
        class_counts=class_counts,
    )


def percentile(values: tuple[float, ...], q: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution_row(name: str, values: tuple[float, ...]) -> tuple[str, str]:
    if not values:
        return name, "n=0 mean=0.000000 p50=0.000000 p95=0.000000"

    return (
        name,
        "n={count} mean={mean:.6f} p50={p50:.6f} p95={p95:.6f}".format(
            count=len(values),
            mean=statistics.fmean(values),
            p50=percentile(values, 0.50),
            p95=percentile(values, 0.95),
        ),
    )


def print_summary(summary: DatasetSummary) -> None:
    rows = [
        ("total sequences", str(summary.total_sequences)),
        ("total frames", str(summary.total_frames)),
        ("sequences below 8 frames", str(summary.below_gru_minimum)),
        ("sequences with no occlusion diversity", str(summary.no_occlusion_diversity)),
        distribution_row("delta cx", summary.delta_cx),
        distribution_row("delta cy", summary.delta_cy),
    ]

    class_text = ", ".join(
        f"{label}={count}"
        for label, count in sorted(summary.class_counts.items())
    ) or "(none)"
    rows.append(("class labels", class_text))

    width = max(len(name) for name, _ in rows)
    print("King Aim dataset quality summary")
    print("-" * (width + 42))
    for name, value in rows:
        print(f"{name:<{width}} : {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate King Aim GRU JSONL track logs")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.is_dir():
        print(f"error: dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1

    summary = summarize_dataset(dataset_dir)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
