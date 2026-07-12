# Validation Guide

Source gates are Release build with zero warnings and errors, all .NET tests, all Python tests, Python syntax, and diff integrity. Dataset gates are provenance completeness, approved usage basis, exact and near-duplicate review, four-keypoint annotation audit, and session leakage audit.

Model selection keeps YOLOv8 epoch 50 as the frozen baseline and compares YOLO26s-Pose (primary), YOLO26n-Pose (low-end), and YOLO11s-Pose (control) on identical locked splits. Measure small-target recall, DirectML P50/P95/P99 latency, box recall/precision, keypoint accuracy, occlusion recall, negative false-positive rate, and runtime stability. Pose exports additionally require raw PyTorch versus ONNX parity and DirectML/C# decoder verification on the target Windows GPU. A candidate that fails DirectML execution, parity, small-target recall, negative-FP limits, C# decoding, or installer integrity cannot win by weighted score.

No GRU, calibration, or movement model becomes production-critical until it beats its simpler baseline on unseen grouped sessions.

TestArena's **Run all reports** command records every scenario for ten seconds with the live pipeline. JSON and CSV reports are written under `%LOCALAPPDATA%\KingAim\TestArenaReports`. The report contract includes detections, false positives, misses, identity switches, track losses, reacquisition, extrapolated-box error, GRU error, observation age, inference percentiles, frame age, and capture FPS.
