"""
King Aim — Phase 3: Calibration MLP training.
Architecture: 6 → 16 (SiLU) → 8 (SiLU) → 1 (Sigmoid). ~300 params.

Collect calibration data by running King Aim with low confidence threshold (0.05)
and logging every detection alongside its ground-truth label (TP=1 / FP=0).

Data format (calibration_data.json):
  [
    {
      "raw_conf": 0.72, "w_norm": 0.04, "h_norm": 0.11,
      "cx_norm": 0.51, "cy_norm": 0.48, "frame_age_ms": 16.7,
      "pose_quality": 0.88, "label": 1
    },
    ...
  ]
label=1 means correct detection (IoU>=0.5 with ground truth human).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np, json, math, os


# ── Architecture ──────────────────────────────────────────────────────────────

class CalibrationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16), nn.SiLU(),
            nn.Linear(16, 8), nn.SiLU(),
            nn.Linear(8, 1),  nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CalibrationDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        p = float(np.clip(r["raw_conf"], 1e-6, 1 - 1e-6))
        features = [
            math.log(p / (1 - p)),
            math.log(max(r["w_norm"] * r["h_norm"], 1e-10)),
            math.log(max(r["h_norm"] / max(r["w_norm"], 1e-5), 1e-5)),
            math.sqrt((r["cx_norm"] - 0.5)**2 + (r["cy_norm"] - 0.5)**2) / 0.7071,
            min(r["frame_age_ms"], 500.0) / 100.0,
            float(r.get("pose_quality", 0.0)),
        ]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(float(r["label"]),  dtype=torch.float32),
        )


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path="data/calibration_data.json", out_dir="runs/calibration"):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(data_path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records  "
          f"pos={sum(r['label'] for r in records)}  "
          f"neg={sum(1-r['label'] for r in records)}")

    ds = CalibrationDataset(records)
    n_val = max(int(len(ds) * 0.15), 1)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=1024)

    model = CalibrationMLP().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)

    best_val = float("inf")
    for epoch in range(1, 101):
        model.train()
        for feats, labels in train_dl:
            feats, labels = feats.to(device), labels.to(device)
            pred = model(feats)
            loss = F.binary_cross_entropy(pred, labels)
            opt.zero_grad(); loss.backward()
            opt.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, labels in val_dl:
                feats, labels = feats.to(device), labels.to(device)
                val_loss += F.binary_cross_entropy(model(feats), labels).item()
        val_loss /= max(len(val_dl), 1)
        print(f"Epoch {epoch:3d}  val_bce={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{out_dir}/best.pt")

    model.load_state_dict(torch.load(f"{out_dir}/best.pt"))
    model.eval().cpu()
    sample = torch.randn(1, 6)
    torch.onnx.export(
        model, (sample,), f"{out_dir}/calibration_mlp.onnx",
        input_names=["features"], output_names=["calibrated_conf"],
        opset_version=18,
    )
    print(f"Exported: {out_dir}/calibration_mlp.onnx")


if __name__ == "__main__":
    train()
