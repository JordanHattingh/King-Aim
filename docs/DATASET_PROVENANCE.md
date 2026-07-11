# Dataset Provenance

Every image record carries local path, source type, source page and URL where applicable, creator, license or permission evidence, attribution, import time, SHA-256, perceptual hash, dimensions, game category, source-session ID, acceptance state, rejection reason, and split.

All frames from one match, video, session, or creator clip stay in one split. Exact SHA-256 duplicates and perceptual near duplicates are reviewed before splitting. Accepted records must pass `audit_licenses.py`; missing provenance fails the data gate.

The canonical external layout is created by `training/initialize_training_workspace.py`. Original media, extracted frames, labels, reports, exports, and model bundles never share a directory.
