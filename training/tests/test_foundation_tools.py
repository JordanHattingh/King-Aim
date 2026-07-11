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


class FoundationToolTests(unittest.TestCase):
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
            (root / "images/train").mkdir(parents=True)
            (root / "labels/train").mkdir(parents=True)
            (root / "images/train/sample.jpg").write_bytes(b"image")
            row = "0 0.5 0.5 0.4 0.8 0.5 0.2 2 0.5 0.3 2 0.5 0.45 2 0.5 0.7 2\n"
            (root / "labels/train/sample.txt").write_text(row, encoding="utf-8")
            self.assertEqual([], audit(root))

    def test_pose_audit_rejects_wrong_field_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images/train").mkdir(parents=True)
            (root / "labels/train").mkdir(parents=True)
            (root / "images/train/sample.jpg").write_bytes(b"image")
            (root / "labels/train/sample.txt").write_text("0 0.5 0.5\n", encoding="utf-8")
            self.assertEqual("field_count", audit(root)[0]["code"])


if __name__ == "__main__":
    unittest.main()
