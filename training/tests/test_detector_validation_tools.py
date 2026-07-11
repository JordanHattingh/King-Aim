from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from export_yolov8_baseline import baseline_id, build_manifest
from validate_detector_onnx import summarize_case, synthetic_inputs


class DetectorValidationToolTests(unittest.TestCase):
    def test_epoch_aware_baseline_id(self) -> None:
        self.assertEqual("kingaim-yolov8-baseline-e040", baseline_id(40))
        self.assertEqual("kingaim-yolov8-baseline-e050", baseline_id(50))
        with self.assertRaises(ValueError):
            baseline_id(0)

    def test_manifest_uses_requested_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch40.pt"
            model = root / "kingaim-yolov8-baseline-e040-fp32.onnx"
            checkpoint.write_bytes(b"checkpoint")
            model.write_bytes(b"onnx")
            manifest = build_manifest(checkpoint, model, 40, 512, {0: "enemy"})
            self.assertEqual("kingaim-yolov8-baseline-e040", manifest["id"])
            self.assertEqual(40, manifest["baseline_epoch"])
            self.assertEqual("rehearsal", manifest["release_status"])
            self.assertEqual({"0": "enemy"}, manifest["class_names"])

    def test_synthetic_inputs_are_deterministic(self) -> None:
        first = synthetic_inputs(32, 42)
        second = synthetic_inputs(32, 42)
        self.assertEqual([name for name, _ in first], [name for name, _ in second])
        for (_, left), (_, right) in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_detector_summary_accepts_matching_contract(self) -> None:
        pytorch = np.zeros((1, 5, 10), dtype=np.float32)
        onnx = pytorch.copy()
        onnx[0, 0, 0] = 1e-5
        report = summarize_case("sample", pytorch, onnx, 1, (0.25,))
        self.assertEqual([1, 5, 10], report["shape"])
        self.assertAlmostEqual(1e-5, report["max_abs_error"], places=9)
        self.assertTrue(report["candidate_counts"]["0.25"]["match"])

    def test_detector_summary_rejects_pose_shape(self) -> None:
        value = np.zeros((1, 17, 10), dtype=np.float32)
        with self.assertRaises(ValueError):
            summarize_case("pose", value, value.copy(), 1, (0.25,))


if __name__ == "__main__":
    unittest.main()
