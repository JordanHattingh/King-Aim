"""
King Aim — Phase 1: YOLO11-Pose training script.
Run this on the gaming PC once you have keypoint annotations.

Dataset YAML format (kingaim_pose.yaml):
    path: C:/KingAimTraining/pose
    train: images/train
    val:   images/val
    kpt_shape: [4, 3]          # 4 keypoints, (x, y, visibility)
    flip_idx: [0, 1, 2, 3]    # midline keypoints — flip maps to self
    names:
      0: human
    kpt_names:
      0: [head, neck, chest, hip]

Keypoint annotation format (YOLO pose .txt):
    class cx cy w h  kx0 ky0 kv0  kx1 ky1 kv1  kx2 ky2 kv2  kx3 ky3 kv3
    All values normalized 0..1. visibility: 0=unknown, 1=occluded, 2=visible.
"""

from ultralytics import YOLO
import torch

# ── Config ────────────────────────────────────────────────────────────────────
DATA_YAML   = "C:/KingAimTraining/pose/kingaim_pose.yaml"
IMGSZ       = 512
EPOCHS      = 200
BATCH       = 6          # GTX 1650 4GB safe batch
DEVICE      = 0          # GPU 0
WORKERS     = 4
SEED        = 42

# Train both sizes — compare before picking production model
for model_name in ["yolo11n-pose.pt", "yolo11s-pose.pt"]:
    print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")
    model = YOLO(model_name)

    results = model.train(
        data       = DATA_YAML,
        imgsz      = IMGSZ,
        epochs     = EPOCHS,
        batch      = BATCH,
        device     = DEVICE,
        workers    = WORKERS,
        seed       = SEED,
        optimizer  = "AdamW",
        lr0        = 5e-4,
        lrf        = 0.01,         # final lr = lr0 * lrf
        weight_decay = 5e-4,
        warmup_epochs = 3,
        cos_lr     = True,
        close_mosaic = 30,         # disable mosaic last 30 epochs
        amp        = True,         # automatic mixed precision
        # Augmentation — tuned for FPS games
        hsv_h      = 0.015,
        hsv_s      = 0.5,
        hsv_v      = 0.3,
        degrees    = 5.0,          # small rotation only
        translate  = 0.1,
        scale      = 0.5,
        fliplr     = 0.5,
        flipud     = 0.0,          # FPS never upside down
        mosaic     = 0.8,
        mixup      = 0.1,
        copy_paste = 0.05,
        erasing    = 0.3,          # occlusion simulation
        # Pose-specific
        pose       = 12.0,         # keypoint loss weight
        kobj       = 1.0,          # keypoint objectness loss weight
        project    = f"runs/pose",
        name       = model_name.replace(".pt", ""),
        exist_ok   = True,
    )

    # Export best weights to ONNX FP32 (validate first)
    best = YOLO(f"runs/pose/{model_name.replace('.pt','')}/weights/best.pt")
    best.export(
        format   = "onnx",
        imgsz    = IMGSZ,
        opset    = 18,
        simplify = True,
        dynamic  = False,
        half     = False,           # FP32 first — validate, then convert to FP16
    )
    print(f"Exported FP32 ONNX: runs/pose/{model_name.replace('.pt','')}/weights/best.onnx")

print("\nDone. Next: validate FP32 ONNX, then convert to FP16 with onnxconverter-common.")
