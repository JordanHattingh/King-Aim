"""
King Aim — contextual calibration MLP training.

Feature schema: detection-context-v2
The split is grouped by session_id. Random detection-row splitting is rejected by
default because neighbouring detections from one recording leak scene/model state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

from contracts import CALIBRATION_FEATURE_SCHEMA, CALIBRATION_LOG_FIELDS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class CalibrationMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CalibrationDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        probability = float(np.clip(record["raw_conf"], 1e-6, 1 - 1e-6))
        features = [
            math.log(probability / (1 - probability)),
            math.log(max(float(record["w_norm"]) * float(record["h_norm"]), 1e-10)),
            math.log(max(float(record["h_norm"]) / max(float(record["w_norm"]), 1e-5), 1e-5)),
            math.sqrt(
                (float(record["cx_norm"]) - 0.5) ** 2
                + (float(record["cy_norm"]) - 0.5) ** 2
            )
            / 0.70710678,
            min(float(record["frame_age_ms"]), 500.0) / 100.0,
            float(record.get("pose_quality", 0.0)),
        ]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(float(record["label"]), dtype=torch.float32),
        )


def split_by_session(records: list[dict], val_fraction: float, seed: int, allow_random_split: bool) -> tuple[list[dict], list[dict]]:
    sessions: dict[str, list[dict]] = {}
    for record in records:
        session_id = str(record.get("session_id", ""))
        if not session_id:
            if not allow_random_split:
                raise ValueError(
                    "Calibration records are missing session_id. Re-label TrackLogger v2 data, "
                    "or pass --allow-random-split only for bootstrap experiments."
                )
            session_id = "__missing__"
        sessions.setdefault(session_id, []).append(record)

    if len(sessions) < 2:
        if not allow_random_split:
            raise ValueError("At least two sessions are required for grouped train/validation calibration")
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        return shuffled[n_val:], shuffled[:n_val]

    session_ids = list(sessions)
    random.Random(seed).shuffle(session_ids)
    n_val_sessions = max(1, int(round(len(session_ids) * val_fraction)))
    n_val_sessions = min(n_val_sessions, len(session_ids) - 1)
    val_ids = set(session_ids[:n_val_sessions])
    train_records = [record for sid, values in sessions.items() if sid not in val_ids for record in values]
    val_records = [record for sid, values in sessions.items() if sid in val_ids for record in values]
    return train_records, val_records


def brier_score(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return float(torch.mean((predictions - labels) ** 2).item())


def expected_calibration_error(predictions: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    ece = torch.tensor(0.0)
    boundaries = torch.linspace(0, 1, bins + 1)
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (predictions > low) & (predictions <= high)
        if mask.any():
            ece += mask.float().mean() * torch.abs(labels[mask].mean() - predictions[mask].mean())
    return float(ece.item())


def train(
    data_path: str = "data/calibration_data.json",
    out_dir: str = "runs/calibration",
    val_fraction: float = 0.15,
    seed: int = 42,
    allow_random_split: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(data_path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not records:
        raise ValueError("Calibration dataset is empty")
    if any(record.get("label") is None for record in records):
        raise ValueError("Calibration dataset contains unlabeled records")

    train_records, val_records = split_by_session(records, val_fraction, seed, allow_random_split)
    print(
        f"Loaded {len(records)} records; train={len(train_records)} val={len(val_records)} "
        f"sessions={len(set(str(r.get('session_id', '')) for r in records))}"
    )

    train_dl = DataLoader(CalibrationDataset(train_records), batch_size=1024, shuffle=True)
    val_dl = DataLoader(CalibrationDataset(val_records), batch_size=1024, shuffle=False)

    model = CalibrationMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    best_val = float("inf")
    for epoch in range(1, 101):
        model.train()
        for features, labels in train_dl:
            features = features.to(device)
            labels = labels.to(device)
            prediction = model(features)
            loss = F.binary_cross_entropy(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        predictions: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        with torch.no_grad():
            for features, labels in val_dl:
                features = features.to(device)
                labels = labels.to(device)
                prediction = model(features)
                val_loss += F.binary_cross_entropy(prediction, labels).item()
                predictions.append(prediction.cpu())
                labels_all.append(labels.cpu())
        val_loss /= max(len(val_dl), 1)
        pred_tensor = torch.cat(predictions)
        label_tensor = torch.cat(labels_all)
        brier = brier_score(pred_tensor, label_tensor)
        ece = expected_calibration_error(pred_tensor, label_tensor)
        print(f"Epoch {epoch:3d} val_bce={val_loss:.6f} brier={brier:.6f} ece={ece:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), Path(out_dir) / "best.pt")

    model.load_state_dict(torch.load(Path(out_dir) / "best.pt", map_location="cpu"))
    model.eval().cpu()
    sample = torch.randn(1, 6)
    torch.onnx.export(
        model,
        (sample,),
        Path(out_dir) / "calibration_mlp.onnx",
        input_names=["features"],
        output_names=["calibrated_conf"],
        opset_version=18,
    )
    with (Path(out_dir) / "feature_schema.json").open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": 2, "feature_schema": CALIBRATION_FEATURE_SCHEMA}, handle, indent=2)
    print(f"Exported: {Path(out_dir) / 'calibration_mlp.onnx'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train King Aim contextual confidence calibrator")
    parser.add_argument("--data", default="data/calibration_data.json")
    parser.add_argument("--out-dir", default="runs/calibration")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-random-split",
        action="store_true",
        help="Bootstrap only. Allows row-level random split when session IDs are unavailable.",
    )
    args = parser.parse_args()
    train(args.data, args.out_dir, args.val_fraction, args.seed, args.allow_random_split)


if __name__ == "__main__":
    main()
