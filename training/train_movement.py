"""
King Aim — controlled accessibility/TestArena pointing movement MLP training.

Feature schema: pointing-velocity-v1
Architecture: 8 -> 64 -> 64 -> 32 -> 2 (Tanh).

This trainer expects generic pointing-task recordings with session_id. Validation
is grouped by session. It does NOT train from live detected-enemy steering data.
The old adjacent-batch "acceleration" loss was invalid because batches are shuffled;
it has been removed until a true sequential-window dataset is implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from contracts import MOVEMENT_FEATURE_FIELDS, MOVEMENT_FEATURE_SCHEMA, MOVEMENT_TARGET_FIELDS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MovementMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 2), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MovementDataset(Dataset):
    def __init__(self, records: list[dict], augment: bool = False) -> None:
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        features = [
            float(record["dx"]),
            float(record["dy"]),
            float(record["distance"]),
            float(record["speed_pix_per_ms"]),
            float(record["target_size"]),
            float(record["dt_sec"]),
            float(record["prev_vx"]),
            float(record["prev_vy"]),
        ]
        target = [float(record["human_vx"]), float(record["human_vy"])]

        if self.augment:
            if np.random.rand() < 0.5:
                features[0] *= -1
                features[6] *= -1
                target[0] *= -1
            if np.random.rand() < 0.5:
                features[1] *= -1
                features[7] *= -1
                target[1] *= -1
            for feature_index in range(5):
                features[feature_index] += (
                    np.random.randn() * max(abs(features[feature_index]), 1e-3) * 0.02
                )

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def movement_loss(prediction: torch.Tensor, human: torch.Tensor) -> torch.Tensor:
    velocity_loss = F.smooth_l1_loss(prediction, human)
    speed_mask = torch.linalg.vector_norm(human, dim=-1) > 1e-4
    if speed_mask.any():
        cosine = F.cosine_similarity(prediction[speed_mask], human[speed_mask], dim=-1, eps=1e-6)
        direction_loss = (1.0 - cosine).mean()
    else:
        direction_loss = prediction.new_tensor(0.0)
    return velocity_loss + 0.15 * direction_loss


def split_by_session(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    sessions: dict[str, list[dict]] = {}
    for record in records:
        session_id = str(record.get("session_id", ""))
        if not session_id:
            raise ValueError(
                "Movement records require session_id. Re-record with record_movement.py v2; "
                "random row-level splitting is not permitted."
            )
        sessions.setdefault(session_id, []).append(record)
    if len(sessions) < 2:
        raise ValueError("At least two independent pointing sessions are required")

    session_ids = list(sessions)
    random.Random(seed).shuffle(session_ids)
    n_val = max(1, int(round(len(session_ids) * val_fraction)))
    n_val = min(n_val, len(session_ids) - 1)
    val_ids = set(session_ids[:n_val])
    train_records = [r for sid, values in sessions.items() if sid not in val_ids for r in values]
    val_records = [r for sid, values in sessions.items() if sid in val_ids for r in values]
    return train_records, val_records


def train(
    data_path: str = "data/movement_data.json",
    out_dir: str = "runs/movement",
    val_fraction: float = 0.15,
    seed: int = 42,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(data_path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not records:
        raise ValueError("Movement dataset is empty")
    if any(record.get("source") != "testarena_pointing" for record in records):
        raise ValueError(
            "Movement dataset contains an unsupported source. Use controlled TestArena/accessibility "
            "pointing records with source='testarena_pointing'."
        )

    train_records, val_records = split_by_session(records, val_fraction, seed)
    print(f"Loaded {len(records)} samples; train={len(train_records)} val={len(val_records)}")

    train_dl = DataLoader(MovementDataset(train_records, augment=True), batch_size=256, shuffle=True)
    val_dl = DataLoader(MovementDataset(val_records), batch_size=256, shuffle=False)

    model = MovementMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    best_val = float("inf")
    for epoch in range(1, 101):
        model.train()
        for features, target in train_dl:
            features = features.to(device)
            target = target.to(device)
            loss = movement_loss(model(features), target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, target in val_dl:
                features = features.to(device)
                target = target.to(device)
                val_loss += movement_loss(model(features), target).item()
        val_loss /= max(len(val_dl), 1)
        print(f"Epoch {epoch:3d} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), Path(out_dir) / "best.pt")

    model.load_state_dict(torch.load(Path(out_dir) / "best.pt", map_location="cpu"))
    model.eval().cpu()
    sample = torch.randn(1, 8)
    torch.onnx.export(
        model,
        (sample,),
        Path(out_dir) / "movement_mlp.onnx",
        input_names=["features"],
        output_names=["velocity"],
        opset_version=18,
    )
    with (Path(out_dir) / "feature_schema.json").open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "feature_schema": MOVEMENT_FEATURE_SCHEMA}, handle, indent=2)
    print(f"Exported: {Path(out_dir) / 'movement_mlp.onnx'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train controlled TestArena pointing MLP")
    parser.add_argument("--data", default="data/movement_data.json")
    parser.add_argument("--out-dir", default="runs/movement")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.out_dir, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
