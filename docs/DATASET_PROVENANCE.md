# Dataset Provenance

Every image record carries local path, source type, source page and URL where applicable, creator, license or permission evidence, attribution, import time, SHA-256, perceptual hash, dimensions, game category, source-session ID, acceptance state, rejection reason, and split.

All frames from one match, video, session, or creator clip stay in one split. Exact SHA-256 duplicates and perceptual near duplicates are reviewed before splitting. Accepted records must pass `audit_licenses.py`; missing provenance fails the data gate.

The canonical external layout is created by `training/initialize_training_workspace.py`. Original media, extracted frames, labels, reports, exports, and model bundles never share a directory.

Collectors live in `training/data_acquisition`. `acquire_wikimedia.py` uses the MediaWiki Action API, `acquire_open_images.py` consumes the official Open Images metadata CSV plus an explicit ID list, and `import_explicit_urls.py` requires a reviewed CSV. All downloaded records start as rejected from training with `pending manual review`; an operator must explicitly accept them before grouped splitting. `generate_attribution_report.py` renders accepted attributed entries.
