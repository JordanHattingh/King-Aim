from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TRAINING / "data_acquisition"))

from foundation import atomic_json, hash_tree, sha256_file
from data_acquisition.group_dataset_sources import split_for
from tools.audit_pose_annotations import audit
from contracts import yolo_keypoint_visibility_rows, yolo_output_channel_count


class FoundationToolTests(unittest.TestCase):
    @staticmethod
    def write_pose_yaml(root: Path) -> None:
        (root / "kingaim_pose.yaml").write_text(
            "kpt_shape: [4, 3]\nnames:\n  0: enemy\nkpt_names:\n  0: [head, neck, upper_chest, hip]\n",
            encoding="utf-8",
        )

    def test_hash_tree_is_stable_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("one", encoding="utf-8")
            first = hash_tree(root)
            self.assertEqual(first, hash_tree(root))
            (root / "a.txt").write_text("two", encoding="utf-8")
            self.assertNotEqual(first, hash_tree(root))

    def test_atomic_json_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_json(path, {"schema": 1})
            self.assertEqual({"schema": 1}, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(64, len(sha256_file(path)))

    def test_group_split_is_deterministic(self) -> None:
        self.assertEqual(split_for("match-1", 42), split_for("match-1", 42))
        self.assertIn(split_for("match-2", 42), {"train", "val", "test"})

    def test_pose_audit_accepts_valid_four_keypoint_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_pose_yaml(root)
            (root / "images/train").mkdir(parents=True)
            (root / "labels/train").mkdir(parents=True)
            (root / "images/train/sample.jpg").write_bytes(b"image")
            row = "0 0.5 0.5 0.4 0.8 0.5 0.2 2 0.5 0.3 2 0.5 0.45 2 0.5 0.7 2\n"
            (root / "labels/train/sample.txt").write_text(row, encoding="utf-8")
            self.assertEqual([], audit(root))

    def test_pose_audit_rejects_wrong_field_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_pose_yaml(root)
            (root / "images/train").mkdir(parents=True)
            (root / "labels/train").mkdir(parents=True)
            (root / "images/train/sample.jpg").write_bytes(b"image")
            (root / "labels/train/sample.txt").write_text("0 0.5 0.5\n", encoding="utf-8")
            self.assertEqual("field_count", audit(root)[0]["code"])

    def test_pose_audit_identifies_detector_box_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_pose_yaml(root)
            (root / "images/train").mkdir(parents=True)
            (root / "labels/train").mkdir(parents=True)
            (root / "images/train/sample.jpg").write_bytes(b"image")
            (root / "labels/train/sample.txt").write_text("0 0.5 0.5 0.4 0.8\n", encoding="utf-8")
            self.assertEqual("detector_annotation_not_pose", audit(root)[0]["code"])

    def test_parity_contract_supports_detector_without_visibility_rows(self) -> None:
        self.assertEqual(5, yolo_output_channel_count("detect", class_count=1, keypoint_count=0))
        self.assertEqual([], yolo_keypoint_visibility_rows(class_count=1, keypoint_count=0))

    def test_parity_contract_preserves_four_keypoint_pose_shape(self) -> None:
        self.assertEqual(17, yolo_output_channel_count("pose", class_count=1, keypoint_count=4))
        self.assertEqual([7, 10, 13, 16], yolo_keypoint_visibility_rows(class_count=1, keypoint_count=4))

    def test_baseline_exporter_uses_requested_epoch_in_manifest_id(self) -> None:
        exporter_path = TRAINING / "export_yolov8_baseline.py"
        spec = importlib.util.spec_from_file_location(
            "export_yolov8_baseline_under_test",
            exporter_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual("kingaim-yolov8-baseline-e040", module.baseline_id(40))
        self.assertEqual("kingaim-yolov8-baseline-e050", module.baseline_id(50))

        with self.assertRaises(ValueError):
            module.baseline_id(0)


if __name__ == "__main__":
    unittest.main()
