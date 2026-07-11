from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from evaluate_e050_release import decode, evaluate, letterbox, load_manifest, match


class E050ReleaseTests(unittest.TestCase):
    def test_letterbox_reversal_restores_original_box(self) -> None:
        image = Image.new("RGB", (1920, 1080))
        _, transform = letterbox(image)
        original = np.array([812, 244, 901, 506], dtype=np.float32)
        scaled = original.copy()
        scaled[[0, 2]] = scaled[[0, 2]] * transform.ratio + transform.pad_x
        scaled[[1, 3]] = scaled[[1, 3]] * transform.ratio + transform.pad_y
        cx, cy = (scaled[0] + scaled[2]) / 2, (scaled[1] + scaled[3]) / 2
        raw = np.zeros((1, 5, 1), dtype=np.float32)
        raw[0, :, 0] = [cx, cy, scaled[2] - scaled[0], scaled[3] - scaled[1], 0.9]
        boxes, _ = decode(raw, transform)
        np.testing.assert_allclose(boxes[0], original, atol=1e-3)

    def test_duplicate_prediction_becomes_false_positive(self) -> None:
        gt = np.array([[10, 10, 30, 30]], dtype=np.float32)
        predictions = np.array([[10, 10, 30, 30], [11, 11, 29, 29]], dtype=np.float32)
        result = match(predictions, np.array([0.9, 0.8]), gt, 0.5)
        self.assertEqual(1, len(result["matches"]))
        self.assertEqual([1], result["false_positives"])
        self.assertEqual([], result["false_negatives"])

    def test_manifest_rejects_changed_hash_and_unreviewed_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 10)).save(root / "frame.png")
            manifest = {"images": [{"file": "frame.png", "sha256": "0" * 64, "width": 20, "height": 10,
                                    "reviewed": False, "tags": ["negative"], "objects": []}]}
            path = root / "manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed image hash"):
                load_manifest(path, root)

    def test_manifest_accepts_authoritative_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image_path = root / "frame.png"
            Image.new("RGB", (20, 10)).save(image_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            manifest = {"images": [{"file": "frame.png", "sha256": digest, "width": 20, "height": 10,
                                    "reviewed": True, "tags": ["positive"], "objects": [{"class": "enemy", "bbox_xyxy": [1, 1, 9, 9], "tags": ["small"], "occlusion": "none", "size": "small"}]}]}
            path = root / "manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(manifest, load_manifest(path, root))

    def test_manifest_rejects_empty_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"images": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                load_manifest(path, Path(directory))

    def test_manifest_rejects_nonfinite_boolean_and_bad_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image_path = root / "frame.png"
            Image.new("RGB", (20, 10)).save(image_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            manifest = {"images": [{"file": "frame.png", "sha256": digest, "width": 20, "height": 10,
                                    "reviewed": True, "tags": None, "objects": [{"class": "enemy", "bbox_xyxy": [True, 1, float("nan"), 9]}]}]}
            path = root / "manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                load_manifest(path, root)
            self.assertIn("tags must be", str(raised.exception))
            self.assertIn("four numbers", str(raised.exception))

    def test_manifest_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "manifest.json"
            path.write_text(json.dumps({"images": [{"file": "../frame.png"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "basename"):
                load_manifest(path, root)

    def test_nms_suppresses_iou_equal_to_threshold(self) -> None:
        image = Image.new("RGB", (512, 512))
        _, transform = letterbox(image)
        raw = np.zeros((1, 5, 2), dtype=np.float32)
        raw[0, :, 0] = [100, 100, 40, 40, 0.9]
        raw[0, :, 1] = [120, 100, 40, 40, 0.8]
        with patch("evaluate_e050_release.box_iou", return_value=np.array([[0.45]], dtype=np.float32)):
            boxes, _ = decode(raw, transform)
        self.assertEqual(1, len(boxes))

    def test_rehearsal_rejects_zero_detection_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image_path = root / "PV001.png"
            Image.new("RGB", (512, 512)).save(image_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            manifest = {"images": [{"file": "PV001.png", "sha256": digest, "width": 512, "height": 512,
                                    "reviewed": True, "tags": ["positive"], "objects": [{"class": "enemy", "bbox_xyxy": [100, 100, 200, 250], "tags": ["small"]}]}]}
            manifest_path = root / "manifest.json"; manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            pt_path = root / "model.pt"; pt_path.write_bytes(b"pt")
            onnx_path = root / "model.onnx"; onnx_path.write_bytes(b"onnx")
            empty = lambda _: np.zeros((1, 5, 5376), dtype=np.float32)
            report = evaluate(manifest_path, root, pt_path, onnx_path, root / "output", empty, empty, "test", "rehearsal")
            self.assertEqual("FAIL", report["status"])
            self.assertEqual(0, report["rehearsal_coverage"]["parity_match_count"])
            self.assertIn("PV001", report["rehearsal_coverage"]["expectation_failures"][0])


if __name__ == "__main__":
    unittest.main()
