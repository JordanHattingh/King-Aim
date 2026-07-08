using System.Drawing;
using Aimmy2.AILogic;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace Aimmy2.Tests
{
    public class PredictionFilterTests
    {
        private const int ImageSize = 640;
        private const int NumClasses = 1;

        private static readonly IReadOnlyDictionary<int, string> ModelClasses = new Dictionary<int, string>
        {
            { 0, "enemy" },
        };

        private static DenseTensor<float> MakeSingleDetectionTensor(float xCenter, float yCenter, float width, float height, float confidence)
        {
            // Layout: [1, 4 + numClasses, numDetections] — one detection at index 0.
            var tensor = new DenseTensor<float>(new[] { 1, 4 + NumClasses, 1 });
            tensor[0, 0, 0] = xCenter;
            tensor[0, 1, 0] = yCenter;
            tensor[0, 2, 0] = width;
            tensor[0, 3, 0] = height;
            tensor[0, 4, 0] = confidence;
            return tensor;
        }

        private static readonly Rectangle DetectionBox = new(0, 0, ImageSize, ImageSize);

        [Fact]
        public void ViewmodelExclusion_DropsDetectionInBottomBand()
        {
            // With a 0.2 exclusion fraction on a 640-tall frame, the band starts at y=512.
            // Center at y=560 falls inside it and should be dropped.
            var tensor = MakeSingleDetectionTensor(xCenter: 320, yCenter: 560, width: 40, height: 40, confidence: 0.9f);

            var predictions = PredictionFilter.CreatePredictions(
                tensor, DetectionBox, ImageSize, numDetections: 1, numClasses: NumClasses,
                modelClasses: ModelClasses, minConfidence: 0.5f, selectedClass: "Best Confidence",
                fovMinX: 0, fovMaxX: ImageSize, fovMinY: 0, fovMaxY: ImageSize,
                viewmodelExclusionFraction: 0.2f);

            Assert.Empty(predictions);
        }

        [Fact]
        public void ViewmodelExclusion_KeepsDetectionAboveBand()
        {
            // Center near screen middle — well above the y=512 exclusion band start.
            var tensor = MakeSingleDetectionTensor(xCenter: 320, yCenter: 320, width: 40, height: 40, confidence: 0.9f);

            var predictions = PredictionFilter.CreatePredictions(
                tensor, DetectionBox, ImageSize, numDetections: 1, numClasses: NumClasses,
                modelClasses: ModelClasses, minConfidence: 0.5f, selectedClass: "Best Confidence",
                fovMinX: 0, fovMaxX: ImageSize, fovMinY: 0, fovMaxY: ImageSize,
                viewmodelExclusionFraction: 0.2f);

            Assert.Single(predictions);
        }

        [Fact]
        public void ViewmodelExclusion_ZeroFraction_KeepsBottomDetection()
        {
            // Same low position as the "dropped" case above, but with exclusion disabled (0f) —
            // must be kept, proving the feature is fully opt-in.
            var tensor = MakeSingleDetectionTensor(xCenter: 320, yCenter: 560, width: 40, height: 40, confidence: 0.9f);

            var predictions = PredictionFilter.CreatePredictions(
                tensor, DetectionBox, ImageSize, numDetections: 1, numClasses: NumClasses,
                modelClasses: ModelClasses, minConfidence: 0.5f, selectedClass: "Best Confidence",
                fovMinX: 0, fovMaxX: ImageSize, fovMinY: 0, fovMaxY: ImageSize,
                viewmodelExclusionFraction: 0f);

            Assert.Single(predictions);
        }

        [Fact]
        public void ViewmodelExclusion_DefaultParameter_DoesNotFilterAnything()
        {
            // Calling without the new parameter (default 0f) must preserve old behaviour exactly —
            // existing call sites / models that don't pass it shouldn't have detections silently dropped.
            var tensor = MakeSingleDetectionTensor(xCenter: 320, yCenter: 560, width: 40, height: 40, confidence: 0.9f);

            var predictions = PredictionFilter.CreatePredictions(
                tensor, DetectionBox, ImageSize, numDetections: 1, numClasses: NumClasses,
                modelClasses: ModelClasses, minConfidence: 0.5f, selectedClass: "Best Confidence",
                fovMinX: 0, fovMaxX: ImageSize, fovMinY: 0, fovMaxY: ImageSize);

            Assert.Single(predictions);
        }

        [Fact]
        public void CursorExclusion_DropsDetectionNearCursor()
        {
            var tensor = MakeSingleDetectionTensor(xCenter: 320, yCenter: 320, width: 20, height: 20, confidence: 0.9f);

            var predictions = PredictionFilter.CreatePredictions(
                tensor, DetectionBox, ImageSize, numDetections: 1, numClasses: NumClasses,
                modelClasses: ModelClasses, minConfidence: 0.5f, selectedClass: "Best Confidence",
                fovMinX: 0, fovMaxX: ImageSize, fovMinY: 0, fovMaxY: ImageSize,
                cursorLocalPosition: new PointF(325, 322), cursorExclusionRadius: 30f);

            Assert.Empty(predictions);
        }

        [Fact]
        public void CursorExclusion_KeepsDetectionOutsideRadius()
        {
            var tensor = MakeSingleDetectionTensor(xCenter: 320, yCenter: 320, width: 20, height: 20, confidence: 0.9f);

            var predictions = PredictionFilter.CreatePredictions(
                tensor, DetectionBox, ImageSize, numDetections: 1, numClasses: NumClasses,
                modelClasses: ModelClasses, minConfidence: 0.5f, selectedClass: "Best Confidence",
                fovMinX: 0, fovMaxX: ImageSize, fovMinY: 0, fovMaxY: ImageSize,
                cursorLocalPosition: new PointF(500, 500), cursorExclusionRadius: 30f);

            Assert.Single(predictions);
        }

        [Fact]
        public void CursorExclusion_NullPosition_DoesNotFilterAnything()
        {
            // No cursor position known (e.g. cursor is on a different monitor) — must never
            // filter based on a stale/default (0,0) position.
            var tensor = MakeSingleDetectionTensor(xCenter: 5, yCenter: 5, width: 10, height: 10, confidence: 0.9f);

            var predictions = PredictionFilter.CreatePredictions(
                tensor, DetectionBox, ImageSize, numDetections: 1, numClasses: NumClasses,
                modelClasses: ModelClasses, minConfidence: 0.5f, selectedClass: "Best Confidence",
                fovMinX: 0, fovMaxX: ImageSize, fovMinY: 0, fovMaxY: ImageSize,
                cursorLocalPosition: null, cursorExclusionRadius: 30f);

            Assert.Single(predictions);
        }

        [Fact]
        public void CursorExclusion_ZeroRadius_DoesNotFilterAnything()
        {
            var tensor = MakeSingleDetectionTensor(xCenter: 320, yCenter: 320, width: 20, height: 20, confidence: 0.9f);

            var predictions = PredictionFilter.CreatePredictions(
                tensor, DetectionBox, ImageSize, numDetections: 1, numClasses: NumClasses,
                modelClasses: ModelClasses, minConfidence: 0.5f, selectedClass: "Best Confidence",
                fovMinX: 0, fovMaxX: ImageSize, fovMinY: 0, fovMaxY: ImageSize,
                cursorLocalPosition: new PointF(320, 320), cursorExclusionRadius: 0f);

            Assert.Single(predictions);
        }
    }
}
