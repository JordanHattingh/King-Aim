from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import label_calibration
import prepare_gru_data
import train_calibration
import train_gru
import train_movement
import contracts
import update_manifest


class TrainingContractTests(unittest.TestCase):
    def test_prepare_gru_reads_jsonl_and_keeps_session_grouped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = []
            for index in range(3):
                session_dir = root / f"session_{index}"
                session_dir.mkdir()
                frames = [
                    {
                        "cx": 0.1 + i * 0.01,
                        "cy": 0.2,
                        "w": 0.1,
                        "h": 0.2,
                        "conf": 0.9,
                        "observed": 1,
                        "dt": 0.016,
                        "age": 0.0,
                    }
                    for i in range(9)
                ]
                record = {"track_id": index + 1, "class_name": "enemy", "frames": frames}
                (session_dir / "gru_sequences.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
                sessions.append(session_dir.name)

            loaded = prepare_gru_data.load_sessions(root, min_frames=9)
            self.assertEqual(3, len(loaded))
            split = prepare_gru_data.split_sessions(loaded, val_fraction=0.2, test_fraction=0.2, seed=42)
            split_ids = {name: {session.session_id for session in values} for name, values in split.items()}
            self.assertTrue(split_ids["train"].isdisjoint(split_ids["val"]))
            self.assertTrue(split_ids["train"].isdisjoint(split_ids["test"]))
            self.assertTrue(split_ids["val"].isdisjoint(split_ids["test"]))
            self.assertEqual(set(sessions), set().union(*split_ids.values()))

    def test_gru_consistency_loss_uses_actual_next_dt(self) -> None:
        prediction = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
        target = torch.zeros((1, 4), dtype=torch.float32)
        short_dt = train_gru.temporal_loss(prediction, target, torch.tensor([0.01]))
        long_dt = train_gru.temporal_loss(prediction, target, torch.tensor([0.05]))
        self.assertNotEqual(float(short_dt), float(long_dt))
        self.assertGreater(float(long_dt), float(short_dt))

    def test_calibration_ground_truth_matching_is_one_to_one(self) -> None:
        samples = [
            {
                "session_id": "session_a",
                "frame_id": 1,
                "detection_index": 0,
                "cx_norm": 0.5,
                "cy_norm": 0.5,
                "w_norm": 0.2,
                "h_norm": 0.4,
                "raw_conf": 0.9,
            },
            {
                "session_id": "session_a",
                "frame_id": 1,
                "detection_index": 1,
                "cx_norm": 0.5,
                "cy_norm": 0.5,
                "w_norm": 0.2,
                "h_norm": 0.4,
                "raw_conf": 0.8,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            gt = Path(tmp) / "session_a"
            gt.mkdir()
            (gt / "frame_1.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            labeled = label_calibration.label_with_gt(samples, Path(tmp), iou_threshold=0.5)
        self.assertEqual(2, len(labeled))
        self.assertEqual(1, sum(int(record["label"]) for record in labeled))

    def test_calibration_split_is_grouped_by_session(self) -> None:
        records = [
            {"session_id": session_id, "label": i % 2}
            for session_id in ("a", "b", "c", "d")
            for i in range(5)
        ]
        train, val = train_calibration.split_by_session(
            records,
            val_fraction=0.25,
            seed=42,
            allow_random_split=False,
        )
        train_sessions = {record["session_id"] for record in train}
        val_sessions = {record["session_id"] for record in val}
        self.assertTrue(train_sessions.isdisjoint(val_sessions))

    def test_movement_split_is_grouped_by_session(self) -> None:
        records = [
            {"session_id": session_id}
            for session_id in ("a", "b", "c")
            for _ in range(5)
        ]
        train, val = train_movement.split_by_session(records, val_fraction=0.34, seed=42)
        train_sessions = {record["session_id"] for record in train}
        val_sessions = {record["session_id"] for record in val}
        self.assertTrue(train_sessions.isdisjoint(val_sessions))

    def test_gru_frame_fields_and_encoding_contract_are_exact(self) -> None:
        frame = {
            "cx": 0.75, "cy": 0.25, "w": 0.2, "h": 0.4,
            "conf": 0.8, "observed": 1, "dt": 0.02, "age": 0.03,
        }
        self.assertEqual(set(contracts.GRU_FRAME_FIELDS), set(frame))
        norm = {
            "log_w_mean": 0.0, "log_w_std": 1.0,
            "log_h_mean": 0.0, "log_h_std": 1.0,
            "dt_mean": 0.0, "dt_std": 1.0,
            "age_mean": 0.0, "age_std": 1.0,
        }
        dataset = object.__new__(train_gru.TrackSequenceDataset)
        dataset.norm = norm
        encoded = dataset._encode([frame])[0].tolist()
        self.assertAlmostEqual(0.5, encoded[0], places=6)
        self.assertAlmostEqual(-0.5, encoded[1], places=6)
        self.assertAlmostEqual(float(torch.log(torch.tensor(0.2))), encoded[2], places=6)
        self.assertAlmostEqual(float(torch.log(torch.tensor(0.4))), encoded[3], places=6)
        self.assertAlmostEqual(0.8, encoded[4], places=6)
        self.assertAlmostEqual(1.0, encoded[5], places=6)
        self.assertAlmostEqual(0.02, encoded[6], places=6)
        self.assertAlmostEqual(0.03, encoded[7], places=6)

    def test_gru_delta_target_times_two_contract(self) -> None:
        previous = {"cx": 0.25, "cy": 0.40}
        nxt = {"cx": 0.35, "cy": 0.30, "dt": 0.05}
        target, next_dt = train_gru.TrackSequenceDataset._make_target(previous, nxt)
        self.assertAlmostEqual(0.20, float(target[0]), places=6)
        self.assertAlmostEqual(-0.20, float(target[1]), places=6)
        self.assertAlmostEqual(4.0, float(target[2]), places=6)
        self.assertAlmostEqual(-4.0, float(target[3]), places=6)
        self.assertAlmostEqual(0.05, float(next_dt), places=6)

    def test_calibration_six_feature_contract_is_exact(self) -> None:
        record = {
            "raw_conf": 0.8, "w_norm": 0.2, "h_norm": 0.4,
            "cx_norm": 0.75, "cy_norm": 0.25, "frame_age_ms": 125.0,
            "pose_quality": 0.0, "label": 1,
        }
        dataset = train_calibration.CalibrationDataset([record])
        features, label = dataset[0]
        self.assertEqual(6, len(features))
        self.assertAlmostEqual(float(torch.log(torch.tensor(4.0))), float(features[0]), places=6)
        self.assertAlmostEqual(float(torch.log(torch.tensor(0.08))), float(features[1]), places=6)
        self.assertAlmostEqual(float(torch.log(torch.tensor(2.0))), float(features[2]), places=6)
        self.assertAlmostEqual((0.25**2 + 0.25**2) ** 0.5 / 0.70710678, float(features[3]), places=6)
        self.assertAlmostEqual(1.25, float(features[4]), places=6)
        self.assertAlmostEqual(0.0, float(features[5]), places=6)
        self.assertEqual(1.0, float(label))

    def test_movement_eight_feature_order_is_exact(self) -> None:
        record = {
            "dx": 3.0, "dy": 4.0, "distance": 5.0,
            "speed_pix_per_ms": 1.5, "target_size": 30.0,
            "dt_sec": 0.02, "prev_vx": -0.2, "prev_vy": 0.3,
            "human_vx": 0.4, "human_vy": -0.5,
        }
        features, target = train_movement.MovementDataset([record], augment=False)[0]
        self.assertEqual(list(contracts.MOVEMENT_FEATURE_FIELDS), [
            "dx", "dy", "distance", "speed_pix_per_ms", "target_size",
            "dt_sec", "prev_vx", "prev_vy",
        ])
        expected_features = [3.0, 4.0, 5.0, 1.5, 30.0, 0.02, -0.2, 0.3]
        for expected, actual in zip(expected_features, features.tolist()):
            self.assertAlmostEqual(expected, actual, places=6)
        for expected, actual in zip([0.4, -0.5], target.tolist()):
            self.assertAlmostEqual(expected, actual, places=6)

    def test_schema_ids_are_shared_across_training_tools(self) -> None:
        self.assertEqual("track-motion-8x8-v2", contracts.TEMPORAL_FEATURE_SCHEMA)
        self.assertEqual("detection-context-v2", contracts.CALIBRATION_FEATURE_SCHEMA)
        self.assertEqual("pointing-velocity-v1", contracts.MOVEMENT_FEATURE_SCHEMA)
        self.assertEqual(8, len(contracts.GRU_NORM_FIELDS))


if __name__ == "__main__":
    unittest.main()
