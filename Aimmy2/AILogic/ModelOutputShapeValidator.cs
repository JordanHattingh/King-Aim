namespace Aimmy2.AILogic
{
    /// <summary>
    /// Validates the tensor contract consumed by PredictionFilter.
    /// PredictionFilter expects [batch, channels, detections].
    /// The channel count is decoder-specific and includes pose keypoints when present.
    /// </summary>
    public static class ModelOutputShapeValidator
    {
        public static bool TryValidate(
            IReadOnlyList<int> dimensions,
            int classCount,
            int expectedDetections,
            ModelManifest? manifest,
            out string error)
        {
            if (dimensions.Count != 3)
            {
                error = $"Expected a rank-3 output [batch, channels, detections], got rank {dimensions.Count}.";
                return false;
            }

            int batch = dimensions[0];
            int channels = dimensions[1];
            int detections = dimensions[2];

            if (batch != 1 && batch != -1)
            {
                error = $"Expected batch dimension 1 or dynamic (-1), got {batch}.";
                return false;
            }

            int keypointCount = manifest?.IsPoseModel == true ? manifest.KeypointCount : 0;
            int expectedChannels = 4 + Math.Max(classCount, 1) + keypointCount * 3;

            if (channels != expectedChannels && channels != -1)
            {
                string schema = keypointCount > 0
                    ? $"pose ({keypointCount} keypoints)"
                    : "detection";
                error = $"Expected {expectedChannels} channels for {schema} output " +
                        $"[4 box + {Math.Max(classCount, 1)} classes + {keypointCount * 3} keypoint values], got {channels}.";
                return false;
            }

            if (expectedDetections > 0 && detections != expectedDetections && detections != -1)
            {
                error = $"Expected {expectedDetections} detections for the configured input size, got {detections}.";
                return false;
            }

            error = string.Empty;
            return true;
        }
    }
}
