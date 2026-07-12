# King Aim installer

The installer accepts one explicitly selected, known-good ONNX model bundle. It does not infer the newest model and never packages PyTorch checkpoints.

Required stable bundle files:

- `model.onnx`
- `manifest.json`

The manifest must declare `id`, `version`, `architecture`, `task`, `output_schema`, and `is_pose_model`. `task` must be `detect` or `pose`, and pose identity must agree with `is_pose_model`.

Build from PowerShell:

```powershell
.\installer\Build-Installer.ps1 `
  -StableModelDirectory C:\KingAimTraining\release\known-good-model
```

The build validates the bundle contract, computes SHA-256 values, emits `checksums.json`, and passes an isolated `stable-detector` payload to Inno Setup. Application binaries remain versioned separately from the bundled model metadata. User-added models and `bin\configs` are excluded from installer replacement during upgrades.

E050 and YOLO11 candidates must not be supplied to this command until they pass the complete deployment gate. The stable bundle argument is the explicit promotion boundary and rollback selection point.

`-SkipRedist` exists only for CI fixture compilation because the ViGEmBus redistributable is intentionally not stored in Git. Production builds must omit that switch and provide `installer\redist\ViGEmBus_1.22.0_x64_x86_arm64.exe`.
