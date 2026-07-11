# Model Bundle Specification

A production pose folder contains `model.onnx`, `manifest.json`, `checksums.sha256`, and `MODEL_CARD.md`. Validated companion files may add `temporal.onnx`, `calibration.onnx`, `movement.onnx`, and `norm_constants.json`.

The schema-v2 manifest declares static input dimensions, `yolo-pose-kpt-v1`, four ordered keypoints, visibility activation state, and all companion feature schemas. Absent companions are allowed; declared but missing or incompatible companions are rejected.

Build bundles with `training/build_model_bundle.py`. Never edit checksums after assembly.
