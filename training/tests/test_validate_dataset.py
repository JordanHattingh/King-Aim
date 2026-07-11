from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "validate_dataset.py"
SPEC = importlib.util.spec_from_file_location("validate_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validate_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate_dataset
SPEC.loader.exec_module(validate_dataset)


class ValidateDatasetTests(unittest.TestCase):
    def test_summary_counts_sequences_motion_and_occlusion_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "gru_sequences.jsonl"
            records = [
                {
                    "track_id": 1,
                    "class_name": "enemy",
                    "frames": [
                        {
                            "cx": 0.10 + index * 0.01,
                            "cy": 0.20 + index * 0.02,
                            "w": 0.1,
                            "h": 0.2,
                            "conf": 0.9,
                            "observed": 1 if index != 4 else 0,
                            "dt": 0.016,
                            "age": 0.0 if index != 4 else 0.016,
                        }
                        for index in range(9)
                    ],
                },
                {
                    "track_id": 2,
                    "class_name": "friendly",
                    "frames": [
                        {
                            "cx": 0.5,
                            "cy": 0.5,
                            "w": 0.1,
                            "h": 0.2,
                            "conf": 0.8,
                            "observed": 1,
                            "dt": 0.016,
                            "age": 0.0,
                        }
                        for _ in range(4)
                    ],
                },
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            summary = validate_dataset.summarize_dataset(root)

            self.assertEqual(2, summary.total_sequences)
            self.assertEqual(13, summary.total_frames)
            self.assertEqual(1, summary.below_gru_minimum)
            self.assertEqual(1, summary.no_occlusion_diversity)
            self.assertEqual({"enemy": 1, "friendly": 1}, dict(summary.class_counts))
            self.assertEqual(11, len(summary.delta_cx))
            self.assertAlmostEqual(0.01, validate_dataset.percentile(summary.delta_cx, 0.95), places=6)

    def test_main_returns_one_for_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            self.assertEqual(
                1,
                validate_dataset.main(["--dataset-dir", str(missing)]),
            )


if __name__ == "__main__":
    unittest.main()
