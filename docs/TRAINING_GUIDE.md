# Training Guide

Preserve the active YOLOv8 run. At epoch 50, copy its checkpoint and reports with `freeze_yolov8_baseline.py`, then export static FP32 ONNX with `export_yolov8_baseline.py`.

Before YOLO11 training, run the duplicate, provenance, license, annotation, and grouped-split gates. Start with the 1,000-positive and 300-negative pilot. Train `yolo11n-pose.pt` first, one run per command, then train `yolo11s-pose.pt` against the identical frozen splits. The trainer records its environment, dataset identity, seed, configuration, checkpoints, and export.

Do not upgrade Ultralytics during a comparison series. Do not train final companion networks until the selected perception model stabilizes their input distribution.
