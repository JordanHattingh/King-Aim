using System.Drawing;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Aimmy2.AILogic
{
    internal static class PredictionFilter
    {
        internal readonly record struct OutputSummary(float MaximumConfidence, int RowsAbove005, int RowsAbove050, int RowsScanned);

        internal static OutputSummary SummarizeSingleClassPose(Tensor<float> outputTensor, int rows)
        {
            int available = Math.Min(rows, outputTensor.Dimensions[2]);
            float maximum = float.NegativeInfinity;
            int above005 = 0, above050 = 0;
            for (int i = 0; i < available; i++)
            {
                float confidence = outputTensor[0, 4, i];
                if (!float.IsFinite(confidence)) continue;
                maximum = Math.Max(maximum, confidence);
                if (confidence >= 0.05f) above005++;
                if (confidence >= 0.50f) above050++;
            }
            return new(float.IsNegativeInfinity(maximum) ? 0 : maximum, above005, above050, available);
        }

        internal static List<Prediction> RankPoseCandidates(
            IEnumerable<Prediction> predictions,
            float weakConfidence,
            float minimumPoseQuality,
            float poseBypassConfidence,
            int maximumTargets)
        {
            var candidates = new List<Prediction>();
            foreach (var p in predictions)
            {
                // Strong confidence: trust the model directly — keypoints are secondary outputs
                // that are often noisier than the box confidence. Requiring secondary evidence
                // here rejects valid high-confidence detections with imperfect pose.
                if (p.Confidence >= poseBypassConfidence)
                {
                    p.AdmissionReason = "strong-conf";
                    p.Evidence = BuildEvidence(p);
                    candidates.Add(p);
                    continue;
                }

                // Weak-confidence band: require at least one secondary condition to filter noise.
                if (p.Confidence < weakConfidence)
                    continue;

                bool goodPose     = p.PoseQuality >= minimumPoseQuality;
                bool oneKeypoint  = CountReliableKeypoints(p.Keypoints, 0.4f) >= 1;
                bool goodProps    = HasValidBodyProportions(
                    p.ScreenRectangle.Width > 0 ? p.ScreenRectangle : p.Rectangle);

                string reason;
                if (goodPose)         reason = "weak-conf+pose";
                else if (oneKeypoint) reason = "weak-conf+keypoint";
                else if (goodProps)   reason = "weak-conf+proportions";
                else continue;

                p.AdmissionReason = reason;
                p.Evidence = BuildEvidence(p);
                candidates.Add(p);
            }

            candidates.Sort((a, b) => b.Evidence.QualityScore.CompareTo(a.Evidence.QualityScore));
            // Target ceiling enforced by TrackManager.MaximumActiveTracks after cross-tile
            // deduplication — not here, where it would discard valid detections from other tiles.
            return candidates;
        }

        private static int CountReliableKeypoints(PlayerKeypoints? kp, float threshold)
        {
            if (kp == null) return 0;
            int n = 0;
            if (kp.Head.Visibility   >= threshold) n++;
            if (kp.Neck.Visibility   >= threshold) n++;
            if (kp.Chest.Visibility  >= threshold) n++;
            if (kp.Hip.Visibility    >= threshold) n++;
            return n;
        }

        private static bool HasValidBodyProportions(RectangleF box)
        {
            if (box.Width <= 0 || box.Height <= 0) return false;
            float aspect = box.Height / box.Width;
            // 0.7 covers wide crouching/prone poses; 8.0 covers distant thin silhouettes.
            return aspect is >= 0.7f and <= 8.0f;
        }

        private static float MeanKeypointVisibility(PlayerKeypoints kp) =>
            (kp.Head.Visibility + kp.Neck.Visibility + kp.Chest.Visibility + kp.Hip.Visibility) / 4f;

        private static CandidateEvidence BuildEvidence(Prediction p)
        {
            int validKps = CountReliableKeypoints(p.Keypoints, 0.4f);
            float meanKpConf = p.Keypoints != null ? MeanKeypointVisibility(p.Keypoints) : 0f;
            RectangleF box   = p.ScreenRectangle.Width > 0 ? p.ScreenRectangle : p.Rectangle;
            float aspect     = box.Width > 0 ? box.Height / box.Width : 0f;
            float propScore  = (aspect is >= 1.2f and <= 6.0f)
                ? Math.Clamp(1f - MathF.Abs(aspect - 2.5f) / 3.5f, 0f, 1f) : 0f;

            float quality = p.Confidence  * 0.30f
                          + p.PoseQuality * 0.30f
                          + validKps / 4f * 0.20f
                          + propScore     * 0.20f;

            return new CandidateEvidence
            {
                ModelConfidence        = p.Confidence,
                ValidKeypoints         = validKps,
                MeanKeypointConfidence = meanKpConf,
                PoseGeometry           = p.PoseQuality,
                BodyProportion         = propScore,
                QualityScore           = Math.Clamp(quality, 0f, 1f),
            };
        }

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
            float cursorExclusionRadius = 0f,
            float captureToModelScale = 1f,
            int keypointCount = 0,
            bool keypointVisibilityIsLogit = false,
            float nmsIouThreshold = 0.45f,
            float keypointConfidenceThreshold = 0.40f,
            bool circularFov = true)
        {
            // When the actual captured screen region is smaller than the model's input size
            // (true optical zoom: capture fewer real pixels, then upscale to imageSize before
            // inference), model-space coordinates (0..imageSize) no longer match real screen
            // pixels 1:1. captureToModelScale = capturedPixels / imageSize converts model-space
            // distances back to real screen-pixel distances for ScreenCenterX/Y and Rectangle,
            // which mouse-aim, the overlay, and the gamepad track/selection pipeline all consume
            // as if they were plain screen pixels.
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

            // Circular FOV: inscribed circle of the rectangular FOV box.
            // Keeps the effective detection zone round (matching game reticles) and discards
            // corner detections that slip through the rectangular check but are outside the FOV circle.
            float fovCenterX = (fovMinX + fovMaxX) * 0.5f;
            float fovCenterY = (fovMinY + fovMaxY) * 0.5f;
            float fovCircleRadius = Math.Min(fovMaxX - fovMinX, fovMaxY - fovMinY) * 0.5f;
            float fovCircleRadiusSq = fovCircleRadius * fovCircleRadius;

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

                // Circular FOV check: discard detections whose center is outside the FOV circle.
                float fdx = xCenter - fovCenterX;
                float fdy = yCenter - fovCenterY;
                if (circularFov && fdx * fdx + fdy * fdy > fovCircleRadiusSq) continue;

                // Discard detections whose center falls inside the bottom viewmodel-exclusion band.
                if (viewmodelExclusionFraction > 0f && yCenter >= viewmodelExclusionY) continue;

                // Discard detections centered near the OS mouse cursor.
                if (cursorLocalPosition.HasValue && cursorExclusionRadius > 0f)
                {
                    float dx = xCenter - cursorLocalPosition.Value.X;
                    float dy = yCenter - cursorLocalPosition.Value.Y;
                    if (dx * dx + dy * dy <= cursorExclusionRadiusSq) continue;
                }

                float realX = xMin * captureToModelScale;
                float realY = yMin * captureToModelScale;
                float realWidth = width * captureToModelScale;
                float realHeight = height * captureToModelScale;
                float realCenterX = xCenter * captureToModelScale;
                float realCenterY = yCenter * captureToModelScale;

                // Decode pose keypoints when model has a pose head.
                // YOLO-pose tensor layout (1 class, K keypoints):
                //   row 0..3 = cx,cy,w,h   row 4 = conf   rows 5.. = kx,ky,kv × K
                PlayerKeypoints? keypoints = null;
                if (keypointCount > 0)
                {
                    int kptBase = 4 + numClasses;
                    if (outputTensor.Dimensions[1] >= kptBase + keypointCount * 3)
                    {
                        keypoints = DecodeKeypoints(
                            outputTensor, i, kptBase, keypointCount,
                            captureToModelScale, detectionBox, keypointVisibilityIsLogit);
                    }
                }

                float heightFraction = height / imageSize;
                CandidateScale scale = heightFraction >= 0.5f ? CandidateScale.Large
                                     : heightFraction >= 0.15f ? CandidateScale.Medium
                                     : CandidateScale.Small;

                var prediction = new Prediction
                {
                    Rectangle = new RectangleF(realX, realY, realWidth, realHeight),
                    ScreenRectangle = new RectangleF(
                        detectionBox.Left + realX,
                        detectionBox.Top + realY,
                        realWidth,
                        realHeight),
                    Confidence = bestConfidence,
                    ClassId = bestClassId,
                    ClassName = modelClasses.GetValueOrDefault(bestClassId, $"Class_{bestClassId}"),
                    CenterXTranslated = xCenter / imageSize,
                    CenterYTranslated = yCenter / imageSize,
                    ScreenCenterX = detectionBox.Left + realCenterX,
                    ScreenCenterY = detectionBox.Top + realCenterY,
                    Keypoints = keypoints,
                    Scale = scale,
                };
                prediction.PoseQuality = PoseGeometryValidator.Score(
                    keypoints, prediction.ScreenRectangle, keypointConfidenceThreshold);
                predictions.Add(prediction);
            }

            // NMS: remove duplicate boxes for the same target (sorted by confidence, suppress overlapping same-class boxes)
            return ApplyNms(predictions, Math.Clamp(nmsIouThreshold, 0f, 1f));
        }

        private static PlayerKeypoints DecodeKeypoints(
            Tensor<float> t, int detIdx, int kptBase, int count,
            float scale, Rectangle box, bool visibilityIsLogit)
        {
            Keypoint Kp(int k)
            {
                float kx = t[0, kptBase + k * 3,     detIdx] * scale;
                float ky = t[0, kptBase + k * 3 + 1, detIdx] * scale;
                float rawKv = t[0, kptBase + k * 3 + 2, detIdx];
                float kv = visibilityIsLogit ? Sigmoid(rawKv) : Math.Clamp(rawKv, 0f, 1f);
                return new Keypoint
                {
                    X = box.Left + kx,
                    Y = box.Top  + ky,
                    Visibility = kv,
                };
            }

            return new PlayerKeypoints
            {
                Head  = count > 0 ? Kp(0) : Keypoint.Empty,
                Neck  = count > 1 ? Kp(1) : Keypoint.Empty,
                Chest = count > 2 ? Kp(2) : Keypoint.Empty,
                Hip   = count > 3 ? Kp(3) : Keypoint.Empty,
            };
        }

        private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

        private static List<Prediction> ApplyNms(List<Prediction> predictions, float nmsIouThreshold)
        {
            if (predictions.Count <= 1) return predictions;

            predictions.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
            var keep = new List<Prediction>(predictions.Count);
            var suppressed = new bool[predictions.Count];

            for (int i = 0; i < predictions.Count; i++)
            {
                if (suppressed[i]) continue;
                keep.Add(predictions[i]);
                for (int j = i + 1; j < predictions.Count; j++)
                {
                    if (suppressed[j]) continue;
                    if (predictions[i].ClassId == predictions[j].ClassId &&
                        ComputeIoU(predictions[i].Rectangle, predictions[j].Rectangle) >= nmsIouThreshold)
                    {
                        suppressed[j] = true;
                    }
                }
            }

            return keep;
        }

        private static float ComputeIoU(RectangleF a, RectangleF b)
        {
            float x1 = Math.Max(a.Left, b.Left);
            float y1 = Math.Max(a.Top, b.Top);
            float x2 = Math.Min(a.Right, b.Right);
            float y2 = Math.Min(a.Bottom, b.Bottom);
            if (x2 <= x1 || y2 <= y1) return 0f;
            float intersection = (x2 - x1) * (y2 - y1);
            float union = a.Width * a.Height + b.Width * b.Height - intersection;
            return union > 0 ? intersection / union : 0f;
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
