from __future__ import annotations

import sys
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

import json

from train_pose import DEFAULT_EXPERIMENT_CONTRACT, POSE_CANDIDATES, candidate_role


class PoseCandidateMatrixTests(unittest.TestCase):
    def test_frozen_matrix_has_primary_low_end_and_control(self) -> None:
        self.assertEqual({
            "yolo26s-pose.pt": "primary",
            "yolo26n-pose.pt": "low-end",
            "yolo11s-pose.pt": "control",
        }, POSE_CANDIDATES)

    def test_candidate_role_accepts_path_and_case(self) -> None:
        self.assertEqual("primary", candidate_role(r"C:\models\YOLO26S-POSE.PT"))

    def test_unfrozen_candidate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen candidates"):
            candidate_role("yolo26m-pose.pt")

    def test_resume_checkpoint_remains_supported(self) -> None:
        self.assertEqual("resume", candidate_role("runs/pose/last.pt", resume=True))

    def test_repository_contract_matches_candidate_matrix_and_pose_schema(self) -> None:
        contract = json.loads(DEFAULT_EXPERIMENT_CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(POSE_CANDIDATES, contract["candidate_roles"])
        self.assertEqual(["head", "neck", "upper_chest", "hip"], contract["keypoint_names"])
        self.assertEqual({"0": "not_present_or_outside_frame", "1": "occluded_but_inferable", "2": "visible"}, contract["visibility_convention"])
        self.assertIsNone(contract["dataset_revision_sha256"])
        self.assertIsNone(contract["split_manifest_sha256"])
        self.assertIsNone(contract["directml_hardware_id"])


if __name__ == "__main__":
    unittest.main()
