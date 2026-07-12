# King Aim

King Aim is a Windows accessibility platform for people who need additional visual, haptic, audio, or controlled-pointing support in games. It is derived from Aimmy and retains the upstream attribution and source-available licensing terms in this repository.

## Current release

Version `2.5.0` is the stable DA1 foundation. It includes calibrated detector confidence, global Hungarian association, timestamped Kalman tracking, GRU-64 motion hints, pose-aware association, short-occlusion prediction, typed coordinate contracts, and immutable `AccessibilityObservation` snapshots.

## Neural architecture

```text
screen capture
  -> YOLO detector / YOLO26-Pose candidate
  -> confidence calibration
  -> Hungarian association (IoU + distance + Kalman + optional GRU/pose)
  -> TrackManager
  -> AccessibilityObservation
  -> visual / haptic / audio / optional controlled pointing
```

Pose models use one `human` class and four ordered keypoints: `head`, `neck`, `upper_chest`, `hip`. Missing or incompatible model schemas fail closed.

## Build and test

```powershell
dotnet restore .\Aimmy2.sln
dotnet build .\Aimmy2.sln -c Release --no-restore
dotnet test .\Aimmy2.sln -c Release --no-build
python -m unittest discover -s training/tests -p "test_*.py" -v
```

The application targets .NET 8 WPF and ONNX Runtime DirectML. GPU model parity remains a local hardware gate; normal builds and contract tests run in Windows CI.

## Training foundation

Initialize the external workspace:

```powershell
python training\initialize_training_workspace.py --root C:\KingAimTraining
```

Freeze YOLOv8 epoch 50 without modifying the active run:

```powershell
python training\freeze_yolov8_baseline.py --run C:\path\to\run --output C:\KingAimTraining\baseline\yolov8-e050 --dataset C:\path\to\dataset
```

Train one frozen-matrix pose candidate per command:

```powershell
python training\train_pose.py --data C:\KingAimTraining\pose\kingaim_pose.yaml --model yolo26s-pose.pt --imgsz 512 --epochs 200 --batch 6 --device 0 --save-period 10 --project C:\KingAimTraining\runs\pose --name kingaim-yolo26s-pose-v1
```

Every accepted image belongs to one source session and one dataset split. Run provenance, duplicate, license, and annotation audits before training.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Pose annotation handbook](docs/POSE_ANNOTATION_HANDBOOK.md)
- [Dataset provenance](docs/DATASET_PROVENANCE.md)
- [Training guide](docs/TRAINING_GUIDE.md)
- [Validation guide](docs/VALIDATION_GUIDE.md)
- [Model bundle specification](docs/MODEL_BUNDLE_SPEC.md)
- [Accessibility outputs](docs/ACCESSIBILITY_OUTPUTS.md)

## Current limits

- YOLOv8 E050 remains the comparison baseline until YOLO26s-Pose, YOLO26n-Pose, or the YOLO11s-Pose control wins every locked deployment gate.
- Final GRU, calibration, and movement models are not activated until they beat simpler baselines on unseen grouped sessions.
- DirectML performance and parity require the target Windows GPU.
- Controlled pointing is optional, separately enabled, bounded, stale-observation rejecting, and manually overridable.

## Privacy and data

Training media and generated datasets live outside the repository by default. Source media, permissions, provenance, session identity, hashes, and split assignments remain together. No frame from one match or creator clip may cross train, validation, and test splits.

See [LICENSE](LICENSE) and [SourceAvailable.md](SourceAvailable.md) for repository terms.
