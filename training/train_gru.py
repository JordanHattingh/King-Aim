"""
King Aim — Phase 2: GRU-64 Temporal Predictor training.

Collect training data by running King Aim with detection logging enabled.
Each logged sequence = one player track with timestamps, boxes, confidence.

Input features per frame (8 total):
    cx_norm, cy_norm, log_w, log_h, confidence, observed_mask, dt_sec, age_sec

Output (4 total):
    delta_cx_next, delta_cy_next, vx_norm, vy_norm

Run AFTER the pose model is deployed and logging real detections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json, os, math
from pathlib import Path


# ── Architecture ──────────────────────────────────────────────────────────────

class TrajectoryGRU(nn.Module):
    """GRU-64: 1-layer GRU + 64→32→4 head. ~16k parameters."""
    def __init__(self, input_size=8, hidden_size=64):
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

    def forward(self, x):                    # x: [B, 8, 8]
        out, _ = self.gru(x)                 # out: [B, 8, 64]
        return self.head(out[:, -1, :])      # [B, 4]


# ── Dataset ───────────────────────────────────────────────────────────────────

class TrackSequenceDataset(Dataset):
    """
    Expects a directory of .json files, one per recorded gameplay session.
    Each file is a list of track sequences:
      [
        {
          "frames": [
            {"cx": 0.5, "cy": 0.4, "w": 0.05, "h": 0.12,
             "conf": 0.9, "observed": 1, "dt": 0.0167, "age": 0.0},
            ...  (at least 9 frames for 1 window)
          ]
        },
        ...
      ]
    """
    SEQ_LEN = 8

    def __init__(self, data_dir, norm_constants, augment=False):
        self.norm = norm_constants
        self.augment = augment
        self.windows = []
        # Only load GRU sequence files; calibration_samples.json has no "frames" key.
        for path in Path(data_dir).glob("gru_sequences*.json"):
            with open(path) as f:
                sessions = json.load(f)
            for track in sessions:
                frames = track["frames"]
                for i in range(len(frames) - self.SEQ_LEN):
                    self.windows.append(frames[i:i + self.SEQ_LEN + 1])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        seq_raw = window[:self.SEQ_LEN]
        next_f  = window[self.SEQ_LEN]

        seq = self._encode(seq_raw)
        target = self._make_target(seq_raw[-1], next_f)
        return seq, target

    def _encode(self, frames):
        out = []
        for f in frames:
            obs = [
                (f["cx"] - 0.5) * 2.0,
                (f["cy"] - 0.5) * 2.0,
                (math.log(max(f["w"], 1e-5)) - self.norm["log_w_mean"]) / self.norm["log_w_std"],
                (math.log(max(f["h"], 1e-5)) - self.norm["log_h_mean"]) / self.norm["log_h_std"],
                f["conf"],
                float(f["observed"]),
                (min(f["dt"], 0.10) - self.norm["dt_mean"])  / self.norm["dt_std"],
                (min(f["age"], 0.25) - self.norm["age_mean"]) / self.norm["age_std"],
            ]
            if self.augment:
                # Drop random frames (simulate missed detections)
                if np.random.rand() < 0.20:
                    obs[4] = 0.0; obs[5] = 0.0
                # Jitter position
                obs[0] += np.random.randn() * 0.005
                obs[1] += np.random.randn() * 0.005
            out.append(obs)
        return torch.tensor(out, dtype=torch.float32)

    def _make_target(self, prev, nxt):
        dt = max(nxt["dt"], 1e-4)
        dcx = (nxt["cx"] - prev["cx"]) * 2.0      # normalize to [-2,+2] range
        dcy = (nxt["cy"] - prev["cy"]) * 2.0
        vx  = dcx / dt
        vy  = dcy / dt
        return torch.tensor([dcx, dcy, vx, vy], dtype=torch.float32)


# ── Loss ──────────────────────────────────────────────────────────────────────

def temporal_loss(pred, target):
    pos_loss = F.smooth_l1_loss(pred[:, :2], target[:, :2])
    vel_loss = F.smooth_l1_loss(pred[:, 2:], target[:, 2:])
    # Consistency: position delta should match velocity * dt (dt≈0.0167 at 60fps)
    expected_delta = pred[:, 2:] * 0.0167
    con_loss = F.smooth_l1_loss(pred[:, :2], expected_delta)
    return pos_loss + 0.35 * vel_loss + 0.10 * con_loss


# ── Training ──────────────────────────────────────────────────────────────────

def compute_norm_constants(data_dir):
    """Derive normalization stats from the dataset."""
    log_ws, log_hs, dts, ages = [], [], [], []
    for path in Path(data_dir).glob("gru_sequences*.json"):
        with open(path) as f:
            sessions = json.load(f)
        for track in sessions:
            for fr in track["frames"]:
                if fr["observed"]:
                    log_ws.append(math.log(max(fr["w"], 1e-5)))
                    log_hs.append(math.log(max(fr["h"], 1e-5)))
                dts.append(min(fr["dt"], 0.1))
                ages.append(min(fr["age"], 0.25))
    return {
        "log_w_mean": float(np.mean(log_ws)), "log_w_std": float(max(np.std(log_ws), 0.01)),
        "log_h_mean": float(np.mean(log_hs)), "log_h_std": float(max(np.std(log_hs), 0.01)),
        "dt_mean":    float(np.mean(dts)),    "dt_std":    float(max(np.std(dts),    0.001)),
        "age_mean":   float(np.mean(ages)),   "age_std":   float(max(np.std(ages),   0.001)),
    }


def train(data_dir="data/tracks", out_dir="runs/gru"):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm = compute_norm_constants(data_dir)
    print("Norm constants:", json.dumps(norm, indent=2))
    with open(f"{out_dir}/norm_constants.json", "w") as f:
        json.dump(norm, f, indent=2)

    train_ds = TrackSequenceDataset(f"{data_dir}/train", norm, augment=True)
    val_ds   = TrackSequenceDataset(f"{data_dir}/val",   norm, augment=False)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2)

    model = TrajectoryGRU().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)

    best_val = float("inf")
    for epoch in range(1, 101):
        model.train()
        for seq, tgt in train_dl:
            seq, tgt = seq.to(device), tgt.to(device)
            loss = temporal_loss(model(seq), tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, tgt in val_dl:
                seq, tgt = seq.to(device), tgt.to(device)
                val_loss += temporal_loss(model(seq), tgt).item()
        val_loss /= max(len(val_dl), 1)

        print(f"Epoch {epoch:3d}  val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{out_dir}/best.pt")

    # Export to ONNX
    model.load_state_dict(torch.load(f"{out_dir}/best.pt"))
    model.eval().cpu()
    sample = torch.randn(1, 8, 8)
    torch.onnx.export(
        model, (sample,), f"{out_dir}/trajectory_gru.onnx",
        input_names=["sequence"], output_names=["motion"],
        opset_version=18,
        dynamic_axes=None,
    )
    print(f"Exported: {out_dir}/trajectory_gru.onnx")
    print(f"Copy norm_constants.json to your model bundle and update GruNorm in manifest.json")


if __name__ == "__main__":
    train()
