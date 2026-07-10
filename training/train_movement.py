"""
King Aim — Phase 4: Movement MLP training.
Architecture: 8 → 64 (SiLU) → 64 (SiLU) → 32 (SiLU) → 2 (Tanh). ~8k params.

Outputs velocity in [-1, +1]. Host integrates: delta = velocity * MaxVelocity * dt.

Collect training data from the TestArena pointing recorder:
  - raw mouse deltas at full polling rate (Windows Raw Input)
  - target position, size, current cursor position
  - timestamps in microseconds

Data format (movement_data.json):
  [
    {
      "dx": 45.2, "dy": -12.1, "distance": 46.8,
      "speed_pix_per_ms": 3.2, "target_size": 38.0,
      "dt_sec": 0.0042, "prev_vx": 0.65, "prev_vy": -0.18,
      "human_vx": 0.71, "human_vy": -0.22   ← normalized to [-1,+1]
    },
    ...
  ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np, json, os, math


# ── Architecture ──────────────────────────────────────────────────────────────

class MovementMLP(nn.Module):
    """8 → 64 → 64 → 32 → 2 (Tanh). Outputs normalized velocity."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,  64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 2),  nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MovementDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        feats = [
            float(r["dx"]),
            float(r["dy"]),
            float(r["distance"]),
            float(r["speed_pix_per_ms"]),
            float(r["target_size"]),
            float(r["dt_sec"]),
            float(r["prev_vx"]),
            float(r["prev_vy"]),
        ]
        target = [float(r["human_vx"]), float(r["human_vy"])]

        if self.augment:
            # Horizontal/vertical mirror (consistent flip)
            if np.random.rand() < 0.5:
                feats[0] *= -1; feats[6] *= -1; target[0] *= -1
            if np.random.rand() < 0.5:
                feats[1] *= -1; feats[7] *= -1; target[1] *= -1
            # Small input noise
            for i in range(5):
                feats[i] += np.random.randn() * abs(feats[i]) * 0.02

        return (
            torch.tensor(feats,   dtype=torch.float32),
            torch.tensor(target,  dtype=torch.float32),
        )


# ── Loss ──────────────────────────────────────────────────────────────────────

def movement_loss(pred, human):
    vel_loss  = F.smooth_l1_loss(pred, human)
    cos       = F.cosine_similarity(pred, human, dim=-1, eps=1e-6)
    dir_loss  = (1.0 - cos).mean()
    # Acceleration smoothness (needs sequential batch — approximate with adjacent pairs)
    accel = pred[1:] - pred[:-1]
    acc_loss = (accel ** 2).mean()
    return vel_loss + 0.15 * dir_loss + 0.05 * acc_loss


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path="data/movement_data.json", out_dir="runs/movement"):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(data_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} movement samples")

    ds = MovementDataset(records)
    n_val = max(int(len(ds) * 0.15), 1)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(MovementDataset(
        [records[i] for i in train_ds.indices], augment=True),
        batch_size=256, shuffle=True)
    val_dl   = DataLoader(MovementDataset(
        [records[i] for i in val_ds.indices]),
        batch_size=256)

    model = MovementMLP().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)

    best_val = float("inf")
    for epoch in range(1, 101):
        model.train()
        for feats, target in train_dl:
            feats, target = feats.to(device), target.to(device)
            loss = movement_loss(model(feats), target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, target in val_dl:
                feats, target = feats.to(device), target.to(device)
                val_loss += movement_loss(model(feats), target).item()
        val_loss /= max(len(val_dl), 1)
        print(f"Epoch {epoch:3d}  val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{out_dir}/best.pt")

    model.load_state_dict(torch.load(f"{out_dir}/best.pt"))
    model.eval().cpu()
    sample = torch.randn(1, 8)
    torch.onnx.export(
        model, (sample,), f"{out_dir}/movement_mlp.onnx",
        input_names=["features"], output_names=["velocity"],
        opset_version=18,
    )
    print(f"Exported: {out_dir}/movement_mlp.onnx")


if __name__ == "__main__":
    train()
