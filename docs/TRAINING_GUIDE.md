# Training Guide

Preserve the active YOLOv8 run. At epoch 50, copy its checkpoint and reports with `freeze_yolov8_baseline.py`, then export static FP32 ONNX with `export_yolov8_baseline.py`.

Before candidate training, run the duplicate, provenance, annotation, and grouped-split gates. Start with the 1,000-positive and 300-negative pilot. The first frozen matrix is `yolo26s-pose.pt` (primary), `yolo26n-pose.pt` (low-end), and `yolo11s-pose.pt` (control). Train one model per command against identical splits, augmentation, seed, epoch count, and batch policy. The trainer records its environment, dataset identity, candidate role, configuration, checkpoints, and observed ONNX input/output shapes.

Export uses static FP32 ONNX, batch 1, 512x512, opset 18, and `end2end=False` so the first migration retains the one-to-many decoder contract. The export contract is inspected and written to `onnx_export_contract.json`; output shape is never assumed. Native YOLO26 end-to-end output remains a separate later experiment.

Do not upgrade Ultralytics during a comparison series. Do not train final companion networks until the selected perception model stabilizes their input distribution.
