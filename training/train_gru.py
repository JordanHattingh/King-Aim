"""
King Aim — GRU-64 temporal predictor training.

Feature schema: track-motion-8x8-v2
Input window: 8 timestamped track observations.
Features per observation:
    cx_norm, cy_norm, log_w, log_h, confidence,
    observed_mask, dt_sec, age_sec_since_last_real_detection
Output:
    delta_cx_next, delta_cy_next, vx_norm_per_sec, vy_norm_per_sec

The data directory must already be SESSION-GROUPED by prepare_gru_data.py.
Normalization is computed from TRAIN only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from contracts import GRU_FRAME_FIELDS, GRU_NORM_FIELDS, TEMPORAL_FEATURE_SCHEMA

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TrajectoryGRU(nn.Module):
    """GRU-64: one recurrent layer plus a 64->32->4 head."""

    def __init__(self, input_size: int = 8, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])


class TrackSequenceDataset(Dataset):
    SEQ_LEN = 8

    def __init__(self, data_dir: str | Path, norm_constants: dict, augment: bool = False) -> None:
        self.norm = norm_constants
        self.augment = augment
        self.windows: list[list[dict]] = []

        for path in sorted(Path(data_dir).glob("*gru_sequences*.json")):
            with path.open(encoding="utf-8") as handle:
                tracks = json.load(handle)
            for track in tracks:
                frames = track.get("frames", [])
                for index in range(len(frames) - self.SEQ_LEN):
                    window = frames[index : index + self.SEQ_LEN + 1]
                    # Prediction targets must be real observations. Training against a carried-forward
                    # missing sample teaches the network that 'no measurement' is ground truth motion.
                    if float(window[-1].get("observed", 0)) <= 0:
                        continue
                    self.windows.append(window)

        if not self.windows:
            raise ValueError(f"No valid 8+1 GRU windows found in {data_dir}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = self.windows[idx]
        seq_raw = [dict(frame) for frame in window[: self.SEQ_LEN]]
        next_frame = window[self.SEQ_LEN]

        if self.augment:
            seq_raw = self._augment_sequence(seq_raw)

        sequence = self._encode(seq_raw)
        target, next_dt = self._make_target(window[self.SEQ_LEN - 1], next_frame)
        return sequence, target, next_dt

    @staticmethod
    def _augment_sequence(frames: list[dict]) -> list[dict]:
        """Simulate missing measurements without leaking future geometry into dropped frames."""
        output: list[dict] = []
        previous: dict | None = None
        artificial_age = 0.0

        for source in frames:
            frame = dict(source)
            drop = previous is not None and np.random.rand() < 0.20
            dt = float(np.clip(frame.get("dt", 0.0), 0.0, 0.10))

            if drop:
                frame["cx"] = previous["cx"]
                frame["cy"] = previous["cy"]
                frame["w"] = previous["w"]
                frame["h"] = previous["h"]
                frame["conf"] = 0.0
                frame["observed"] = 0
                artificial_age = min(artificial_age + dt, 0.25)
                frame["age"] = artificial_age
            else:
                if float(frame.get("observed", 0)) > 0:
                    artificial_age = 0.0
                    frame["age"] = 0.0
                else:
                    artificial_age = min(artificial_age + dt, 0.25)
                    frame["age"] = max(float(frame.get("age", 0.0)), artificial_age)

                # Small detector-like center jitter on observed samples only.
                if float(frame.get("observed", 0)) > 0:
                    frame["cx"] = float(frame["cx"]) + float(np.random.randn() * 0.0025)
                    frame["cy"] = float(frame["cy"]) + float(np.random.randn() * 0.0025)

            output.append(frame)
            previous = frame

        return output

    def _encode(self, frames: list[dict]) -> torch.Tensor:
        output: list[list[float]] = []
        for frame in frames:
            output.append(
                [
                    (float(frame["cx"]) - 0.5) * 2.0,
                    (float(frame["cy"]) - 0.5) * 2.0,
                    (math.log(max(float(frame["w"]), 1e-5)) - self.norm["log_w_mean"])
                    / self.norm["log_w_std"],
                    (math.log(max(float(frame["h"]), 1e-5)) - self.norm["log_h_mean"])
                    / self.norm["log_h_std"],
                    float(frame["conf"]),
                    float(frame["observed"]),
                    (min(float(frame["dt"]), 0.10) - self.norm["dt_mean"]) / self.norm["dt_std"],
                    (min(float(frame["age"]), 0.25) - self.norm["age_mean"]) / self.norm["age_std"],
                ]
            )
        return torch.tensor(output, dtype=torch.float32)

    @staticmethod
    def _make_target(previous: dict, nxt: dict) -> tuple[torch.Tensor, torch.Tensor]:
        dt = max(float(nxt["dt"]), 1e-4)
        dcx = (float(nxt["cx"]) - float(previous["cx"])) * 2.0
        dcy = (float(nxt["cy"]) - float(previous["cy"])) * 2.0
        vx = dcx / dt
        vy = dcy / dt
        return (
            torch.tensor([dcx, dcy, vx, vy], dtype=torch.float32),
            torch.tensor(dt, dtype=torch.float32),
        )


def temporal_loss(pred: torch.Tensor, target: torch.Tensor, next_dt: torch.Tensor) -> torch.Tensor:
    pos_loss = F.smooth_l1_loss(pred[:, :2], target[:, :2])
    vel_loss = F.smooth_l1_loss(pred[:, 2:], target[:, 2:])
    expected_delta = pred[:, 2:] * next_dt.unsqueeze(-1)
    consistency_loss = F.smooth_l1_loss(pred[:, :2], expected_delta)
    return pos_loss + 0.35 * vel_loss + 0.10 * consistency_loss


def compute_norm_constants(train_dir: str | Path) -> dict[str, float]:
    """Derive feature normalization from TRAIN ONLY."""
    log_ws: list[float] = []
    log_hs: list[float] = []
    dts: list[float] = []
    ages: list[float] = []

    for path in sorted(Path(train_dir).glob("*gru_sequences*.json")):
        with path.open(encoding="utf-8") as handle:
            tracks = json.load(handle)
        for track in tracks:
            for frame in track.get("frames", []):
                if float(frame.get("observed", 0)) > 0:
                    log_ws.append(math.log(max(float(frame["w"]), 1e-5)))
                    log_hs.append(math.log(max(float(frame["h"]), 1e-5)))
                dts.append(min(float(frame["dt"]), 0.1))
                ages.append(min(float(frame["age"]), 0.25))

    if not log_ws or not dts:
        raise ValueError("Training split does not contain enough observed GRU samples for normalization")

    return {
        "log_w_mean": float(np.mean(log_ws)),
        "log_w_std": float(max(np.std(log_ws), 0.01)),
        "log_h_mean": float(np.mean(log_hs)),
        "log_h_std": float(max(np.std(log_hs), 0.01)),
        "dt_mean": float(np.mean(dts)),
        "dt_std": float(max(np.std(dts), 0.001)),
        "age_mean": float(np.mean(ages)),
        "age_std": float(max(np.std(ages), 0.001)),
    }


def train(data_dir: str = "data/tracks", out_dir: str = "runs/gru") -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    norm = compute_norm_constants(train_dir)
    print("Norm constants:", json.dumps(norm, indent=2))
    with open(Path(out_dir) / "norm_constants.json", "w", encoding="utf-8") as handle:
        json.dump(norm, handle, indent=2)

    train_ds = TrackSequenceDataset(train_dir, norm, augment=True)
    val_ds = TrackSequenceDataset(val_dir, norm, augment=False)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = TrajectoryGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    best_val = float("inf")
    for epoch in range(1, 101):
        model.train()
        for sequence, target, next_dt in train_dl:
            sequence = sequence.to(device)
            target = target.to(device)
            next_dt = next_dt.to(device)
            loss = temporal_loss(model(sequence), target, next_dt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequence, target, next_dt in val_dl:
                sequence = sequence.to(device)
                target = target.to(device)
                next_dt = next_dt.to(device)
                val_loss += temporal_loss(model(sequence), target, next_dt).item()
        val_loss /= max(len(val_dl), 1)

        print(f"Epoch {epoch:3d}  val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), Path(out_dir) / "best.pt")

    model.load_state_dict(torch.load(Path(out_dir) / "best.pt", map_location="cpu"))
    model.eval().cpu()
    sample = torch.randn(1, 8, 8)
    torch.onnx.export(
        model,
        (sample,),
        Path(out_dir) / "trajectory_gru.onnx",
        input_names=["sequence"],
        output_names=["motion"],
        opset_version=18,
        dynamic_axes=None,
    )
    print(f"Exported: {Path(out_dir) / 'trajectory_gru.onnx'}")
    print(f"Feature schema: {TEMPORAL_FEATURE_SCHEMA}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the King Aim GRU-64 temporal predictor")
    parser.add_argument("--data-dir", default="data/tracks")
    parser.add_argument("--out-dir", default="runs/gru")
    args = parser.parse_args()
    train(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
