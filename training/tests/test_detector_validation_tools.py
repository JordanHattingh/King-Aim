from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))

from export_yolov8_baseline import baseline_id, build_manifest
from validate_detector_onnx import (
    _box_iou_batch,
    _nms,
    decode_detections,
    semantic_parity_case,
    summarize_case,
    synthetic_inputs,
)


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

    # ------------------------------------------------------------------
    # NMS and decode
    # ------------------------------------------------------------------

    def test_nms_suppresses_overlapping_box(self) -> None:
        # Two heavily overlapping boxes — NMS should keep only the highest-score one.
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, iou_thresh=0.5)
        self.assertEqual([0], keep.tolist())

    def test_nms_keeps_non_overlapping_boxes(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, iou_thresh=0.5)
        self.assertEqual(sorted(keep.tolist()), [0, 1])

    def test_decode_detections_filters_by_threshold(self) -> None:
        # Build a [1, 5, 4] tensor: 2 high-confidence, 2 low-confidence anchors.
        raw = np.zeros((1, 5, 4), dtype=np.float32)
        # cx, cy, w, h in pixel space
        raw[0, :4, 0] = [50, 50, 20, 20]   # conf=0.9
        raw[0, :4, 1] = [150, 150, 20, 20]  # conf=0.8
        raw[0, :4, 2] = [250, 250, 20, 20]  # conf=0.09 (below 0.10)
        raw[0, :4, 3] = [350, 350, 20, 20]  # conf=0.05 (below 0.10)
        raw[0, 4, :] = [0.9, 0.8, 0.09, 0.05]
        boxes, scores = decode_detections(raw, conf_thresh=0.10)
        self.assertEqual(2, len(scores))
        self.assertGreater(scores[0], scores[1])  # sorted descending

    def test_decode_detections_empty_below_threshold(self) -> None:
        raw = np.zeros((1, 5, 3), dtype=np.float32)
        raw[0, 4, :] = [0.01, 0.02, 0.03]
        boxes, scores = decode_detections(raw, conf_thresh=0.10)
        self.assertEqual(0, len(scores))
        self.assertEqual((0, 4), boxes.shape)

    def test_box_iou_identical_boxes(self) -> None:
        box = np.array([[10, 10, 30, 30]], dtype=np.float32)
        iou = _box_iou_batch(box, box)
        self.assertAlmostEqual(1.0, iou[0, 0], places=6)

    def test_box_iou_non_overlapping(self) -> None:
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        iou = _box_iou_batch(a, b)
        self.assertAlmostEqual(0.0, iou[0, 0], places=6)

    # ------------------------------------------------------------------
    # Semantic parity
    # ------------------------------------------------------------------

    def _make_raw(
        self,
        n_anchors: int = 20,
        detections: list[tuple[float, float, float, float, float]] | None = None,
    ) -> np.ndarray:
        """Build a [1,5,n_anchors] tensor with optional high-confidence detections."""
        raw = np.zeros((1, 5, n_anchors), dtype=np.float32)
        for i, (cx, cy, w, h, conf) in enumerate(detections or []):
            raw[0, 0, i] = cx
            raw[0, 1, i] = cy
            raw[0, 2, i] = w
            raw[0, 3, i] = h
            raw[0, 4, i] = conf
        return raw

    def test_semantic_parity_identical_outputs_pass(self) -> None:
        raw = self._make_raw(detections=[(100, 100, 40, 60, 0.9)])
        result = semantic_parity_case("test", raw, raw.copy(), conf_thresholds=(0.25,))
        self.assertTrue(result["passed"])
        self.assertTrue(result["by_threshold"]["0.25"]["passed"])

    def test_semantic_parity_zero_detections_both_sides_pass(self) -> None:
        # No detections on either side at this threshold — vacuous pass.
        raw = self._make_raw(detections=[(100, 100, 40, 60, 0.05)])
        result = semantic_parity_case("test", raw, raw.copy(), conf_thresholds=(0.25,))
        self.assertTrue(result["passed"])

    def test_semantic_parity_sub_pixel_raw_fail_is_semantic_pass(self) -> None:
        # Same detection, but onnx output perturbed by 0.0006 (exceeds raw tol 0.0005).
        # The perturbation shifts the box by <<0.25px and conf by <<0.001 — semantic PASS.
        pt = self._make_raw(detections=[(100.000, 100.000, 40.0, 60.0, 0.85)])
        onnx = self._make_raw(detections=[(100.001, 100.001, 40.0, 60.0, 0.8501)])
        result = semantic_parity_case("DT4", pt, onnx, conf_thresholds=(0.25,))
        self.assertTrue(result["passed"])
        t = result["by_threshold"]["0.25"]
        self.assertIsNotNone(t["center_diff_max_px"])
        self.assertLess(t["center_diff_max_px"], 0.25)

    def test_semantic_parity_different_count_fails(self) -> None:
        # PyTorch outputs 1 detection; ONNX outputs 0.
        pt = self._make_raw(detections=[(100, 100, 40, 60, 0.9)])
        onnx = self._make_raw()  # all zeros → no detections
        result = semantic_parity_case("bad_frame", pt, onnx, conf_thresholds=(0.25,))
        self.assertFalse(result["passed"])
        self.assertFalse(result["by_threshold"]["0.25"]["count_match"])

    def test_semantic_parity_large_center_shift_fails(self) -> None:
        # A 10px centre shift makes the boxes non-overlapping enough that IoU < 0.999,
        # so geometric matching rejects the pair on IoU (not centre_diff).
        pt = self._make_raw(detections=[(100.0, 100.0, 40.0, 60.0, 0.9)])
        onnx = self._make_raw(detections=[(110.0, 110.0, 40.0, 60.0, 0.9)])
        result = semantic_parity_case("shifted", pt, onnx, conf_thresholds=(0.25,))
        self.assertFalse(result["passed"])
        # Matching fails due to IoU below 0.999 (large shift → low IoU)
        t = result["by_threshold"]["0.25"]
        self.assertIsNotNone(t["box_iou_min"])
        self.assertLess(t["box_iou_min"], 0.999)

    def test_semantic_parity_order_swap_still_passes(self) -> None:
        # Two detections whose confidence ordering swaps between PyTorch and ONNX.
        # PyTorch: A(conf=0.9005) > B(conf=0.9000)  → sorted [A, B]
        # ONNX:    B(conf=0.9004) > A(conf=0.9001)  → sorted [B, A]
        # Conf diffs: A=0.0004, B=0.0004 — both within 0.001 tolerance.
        # Positional matching would compare A↔B and B↔A (large centre diff → FAIL).
        # Geometric matching should correctly pair A↔A and B↔B.
        pt = self._make_raw(
            n_anchors=30,
            detections=[
                (50.0, 50.0, 20.0, 30.0, 0.9005),    # detection A
                (300.0, 300.0, 20.0, 30.0, 0.9000),   # detection B
            ],
        )
        onnx = self._make_raw(
            n_anchors=30,
            detections=[
                (300.001, 300.001, 20.0, 30.0, 0.9004),  # B has higher conf in ONNX
                (50.001, 50.001, 20.0, 30.0, 0.9001),    # A has lower conf in ONNX
            ],
        )
        result = semantic_parity_case("swapped", pt, onnx, conf_thresholds=(0.25,))
        self.assertTrue(result["passed"], msg=f"Geometric matching should handle order swap: {result}")


if __name__ == "__main__":
    unittest.main()
