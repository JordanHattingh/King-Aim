"""Shared King Aim neural feature/data-contract identifiers.

The C# runtime mirrors these exact identifiers in
Aimmy2/AILogic/NeuralFeatureSchemas.cs.  Keep this module dependency-free so all
training/export tools can import it when executed directly from ``training/``.
"""

TEMPORAL_FEATURE_SCHEMA = "track-motion-8x8-v2"
CALIBRATION_FEATURE_SCHEMA = "detection-context-v2"
MOVEMENT_FEATURE_SCHEMA = "pointing-velocity-v1"

GRU_FRAME_FIELDS = ("cx", "cy", "w", "h", "conf", "observed", "dt", "age")
GRU_NORM_FIELDS = (
    "log_w_mean",
    "log_w_std",
    "log_h_mean",
    "log_h_std",
    "dt_mean",
    "dt_std",
    "age_mean",
    "age_std",
)
CALIBRATION_LOG_FIELDS = (
    "frame_id",
    "detection_index",
    "raw_conf",
    "w_norm",
    "h_norm",
    "cx_norm",
    "cy_norm",
    "frame_age_ms",
    "pose_quality",
    "label",
)
CALIBRATION_FEATURE_FIELDS = (
    "raw_conf",
    "w_norm",
    "h_norm",
    "cx_norm",
    "cy_norm",
    "frame_age_ms",
)
MOVEMENT_FEATURE_FIELDS = (
    "dx",
    "dy",
    "distance",
    "speed_pix_per_ms",
    "target_size",
    "dt_sec",
    "prev_vx",
    "prev_vy",
)
MOVEMENT_TARGET_FIELDS = ("human_vx", "human_vy")


def yolo_output_channel_count(task: str, class_count: int, keypoint_count: int = 0) -> int:
    if task not in {"detect", "pose"}:
        raise ValueError(f"Unsupported YOLO task: {task}")
    if class_count <= 0 or keypoint_count < 0:
        raise ValueError("class_count must be positive and keypoint_count non-negative")
    return 4 + class_count + (keypoint_count * 3 if task == "pose" else 0)


def yolo_keypoint_visibility_rows(class_count: int, keypoint_count: int) -> list[int]:
    return [4 + class_count + keypoint_index * 3 + 2 for keypoint_index in range(keypoint_count)]
