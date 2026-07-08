using System.Drawing;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Aimmy2.AILogic
{
    internal static class PredictionFilter
    {
        internal static List<Prediction> CreatePredictions(
            Tensor<float> outputTensor,
            Rectangle detectionBox,
            int imageSize,
            int numDetections,
            int numClasses,
            IReadOnlyDictionary<int, string> modelClasses,
            float minConfidence,
            string selectedClass,
            float fovMinX,
            float fovMaxX,
            float fovMinY,
            float fovMaxY,
            float viewmodelExclusionFraction = 0f,
            PointF? cursorLocalPosition = null,
            float cursorExclusionRadius = 0f)
        {
            // First-person viewmodels (own gun/hands) almost always render in the bottom-center
            // of the screen and can be misclassified as an enemy by detectors not trained with
            // viewmodel negatives. viewmodelExclusionFraction (0..1) is the portion of the capture
            // box's height, measured up from the bottom, where detections are discarded entirely.
            // This is a heuristic, not a fix for the model itself: at high values it can also
            // discard a legitimate enemy standing very close/low in frame.
            float viewmodelExclusionY = imageSize * (1f - Math.Clamp(viewmodelExclusionFraction, 0f, 1f));

            // The OS mouse cursor icon rendered on top of the game can itself be misclassified as
            // a target. cursorLocalPosition (in the same 0..imageSize local space as detections)
            // plus cursorExclusionRadius discards any detection centered within that radius of it.
            float cursorExclusionRadiusSq = cursorExclusionRadius * cursorExclusionRadius;

            int selectedClassId = ResolveSelectedClassId(modelClasses, selectedClass);

            var predictions = new List<Prediction>(numDetections);

            for (int i = 0; i < numDetections; i++)
            {
                float xCenter = outputTensor[0, 0, i];
                float yCenter = outputTensor[0, 1, i];
                float width = outputTensor[0, 2, i];
                float height = outputTensor[0, 3, i];

                int bestClassId = 0;
                float bestConfidence = 0f;

                if (numClasses == 1)
                {
                    bestConfidence = outputTensor[0, 4, i];
                }
                else
                {
                    if (selectedClassId == -1)
                    {
                        for (int classId = 0; classId < numClasses; classId++)
                        {
                            float classConfidence = outputTensor[0, 4 + classId, i];
                            if (classConfidence > bestConfidence)
                            {
                                bestConfidence = classConfidence;
                                bestClassId = classId;
                            }
                        }
                    }
                    else
                    {
                        bestConfidence = outputTensor[0, 4 + selectedClassId, i];
                        bestClassId = selectedClassId;
                    }
                }

                if (bestConfidence < minConfidence) continue;

                float xMin = xCenter - width / 2;
                float yMin = yCenter - height / 2;
                float xMax = xCenter + width / 2;
                float yMax = yCenter + height / 2;

                if (xMin < fovMinX || xMax > fovMaxX || yMin < fovMinY || yMax > fovMaxY) continue;

                // Discard detections whose center falls inside the bottom viewmodel-exclusion band.
                if (viewmodelExclusionFraction > 0f && yCenter >= viewmodelExclusionY) continue;

                // Discard detections centered near the OS mouse cursor.
                if (cursorLocalPosition.HasValue && cursorExclusionRadius > 0f)
                {
                    float dx = xCenter - cursorLocalPosition.Value.X;
                    float dy = yCenter - cursorLocalPosition.Value.Y;
                    if (dx * dx + dy * dy <= cursorExclusionRadiusSq) continue;
                }

                predictions.Add(new Prediction
                {
                    Rectangle = new RectangleF(xMin, yMin, width, height),
                    Confidence = bestConfidence,
                    ClassId = bestClassId,
                    ClassName = modelClasses.GetValueOrDefault(bestClassId, $"Class_{bestClassId}"),
                    CenterXTranslated = xCenter / imageSize,
                    CenterYTranslated = yCenter / imageSize,
                    ScreenCenterX = detectionBox.Left + xCenter,
                    ScreenCenterY = detectionBox.Top + yCenter
                });
            }

            return predictions;
        }

        internal static int ResolveSelectedClassId(IReadOnlyDictionary<int, string> modelClasses, string selectedClass)
        {
            if (selectedClass == "Best Confidence")
            {
                return -1;
            }

            foreach (var (classId, className) in modelClasses)
            {
                if (className == selectedClass)
                {
                    return classId;
                }
            }

            return -1;
        }
    }
}
