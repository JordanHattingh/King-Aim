using System.Drawing;
using System.IO;
using Aimmy2.AILogic;
using Microsoft.ML.OnnxRuntime.Tensors;
using InputLogic;
using System.Reflection;
using Xunit;

namespace Aimmy2.Tests
{
    public sealed class NeuralContractTests
    {
        private static Prediction MakePrediction(
            float x,
            float y,
            float size = 40f,
            float confidence = 0.9f,
            int classId = 0) => new()
        {
            Rectangle = new RectangleF(x, y, size, size),
            ScreenRectangle = new RectangleF(x, y, size, size),
            Confidence = confidence,
            ClassId = classId,
            ClassName = classId == 0 ? "enemy" : "unknown",
            ScreenCenterX = x + size / 2f,
            ScreenCenterY = y + size / 2f,
            CenterXTranslated = (x + size / 2f) / 640f,
            CenterYTranslated = (y + size / 2f) / 640f,
        };

        [Fact]
        public void PoseOutputShape_AcceptsFourKeypointChannels()
        {
            var manifest = ModelManifest.CreateFallback(
                "pose-test",
                "Pose Test",
                new Dictionary<int, string> { [0] = "enemy" },
                512);
            manifest.IsPoseModel = true;
            manifest.OutputSchema = "yolo-pose-kpt-v1";
            manifest.KeypointCount = 4;
            manifest.KeypointNames = ["head", "neck", "chest", "hip"];

            bool valid = ModelOutputShapeValidator.TryValidate(
                new[] { 1, 17, 5376 },
                classCount: 1,
                expectedDetections: 5376,
                manifest,
                out string error);

            Assert.True(valid, error);
        }

        [Fact]
        public void PoseOutputShape_RejectsDetectorOnlyChannelsForPoseManifest()
        {
            var manifest = ModelManifest.CreateFallback(
                "pose-test",
                "Pose Test",
                new Dictionary<int, string> { [0] = "enemy" },
                512);
            manifest.IsPoseModel = true;
            manifest.KeypointCount = 4;
            manifest.KeypointNames = ["head", "neck", "chest", "hip"];

            bool valid = ModelOutputShapeValidator.TryValidate(
                new[] { 1, 5, 5376 },
                classCount: 1,
                expectedDetections: 5376,
                manifest,
                out _);

            Assert.False(valid);
        }

        [Fact]
        public void PoseVisibility_ActivatedValue_IsNotSigmoidedTwice()
        {
            const int keypointCount = 4;
            var tensor = new DenseTensor<float>(new[] { 1, 4 + 1 + keypointCount * 3, 1 });
            tensor[0, 0, 0] = 256;
            tensor[0, 1, 0] = 256;
            tensor[0, 2, 0] = 80;
            tensor[0, 3, 0] = 160;
            tensor[0, 4, 0] = 0.9f;
            for (int k = 0; k < keypointCount; k++)
            {
                int row = 5 + k * 3;
                tensor[0, row, 0] = 256;
                tensor[0, row + 1, 0] = 200 + k * 10;
                tensor[0, row + 2, 0] = 0.95f;
            }

            var predictions = PredictionFilter.CreatePredictions(
                tensor,
                new Rectangle(100, 50, 512, 512),
                512,
                1,
                1,
                new Dictionary<int, string> { [0] = "enemy" },
                0.05f,
                "Best Confidence",
                0,
                512,
                0,
                512,
                keypointCount: keypointCount,
                keypointVisibilityIsLogit: false);

            Prediction prediction = Assert.Single(predictions);
            Assert.NotNull(prediction.Keypoints);
            Assert.Equal(0.95f, prediction.Keypoints!.Neck.Visibility, 3);
            Assert.Equal(356f, prediction.Keypoints.Neck.X, 2);
        }

        [Fact]
        public void ContinuousObservedTrack_AlwaysWritesZeroAgeSeconds()
        {
            var manager = new TrackManager
            {
                ScreenWidth = 640,
                ScreenHeight = 640,
                ScreenLeft = 0,
                ScreenTop = 0,
            };
            var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
            DateTime now = new(2026, 7, 10, 12, 0, 0, DateTimeKind.Utc);

            IReadOnlyList<Track> tracks = Array.Empty<Track>();
            for (int i = 0; i < 70; i++)
            {
                tracks = manager.Update([MakePrediction(100 + i, 100)], roles, now);
                now = now.AddMilliseconds(16);
            }

            Track track = Assert.Single(tracks);
            Assert.True(track.RingBuffer.IsReady);
            Assert.All(track.RingBuffer.GetSequence(), sample =>
            {
                Assert.Equal(1f, sample.ObservedMask);
                Assert.Equal(0f, sample.AgeSeconds, 6);
            });
        }

        [Fact]
        public void MissingTrackAge_UsesActualIncrementAndResetsOnObservation()
        {
            var manager = new TrackManager
            {
                ScreenWidth = 640,
                ScreenHeight = 640,
                MaxLostSeconds = 1.0,
            };
            var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
            DateTime now = new(2026, 7, 10, 12, 0, 0, DateTimeKind.Utc);

            var tracks = manager.Update([MakePrediction(100, 100)], roles, now);
            now = now.AddMilliseconds(20);
            tracks = manager.Update([], roles, now);
            Assert.Equal(0.020f, Assert.Single(tracks).RingBuffer.Tail.AgeSeconds, 3);

            now = now.AddMilliseconds(20);
            tracks = manager.Update([], roles, now);
            Assert.Equal(0.040f, Assert.Single(tracks).RingBuffer.Tail.AgeSeconds, 3);

            now = now.AddMilliseconds(20);
            tracks = manager.Update([MakePrediction(106, 100)], roles, now);
            Assert.Equal(0f, Assert.Single(tracks).RingBuffer.Tail.AgeSeconds, 6);
            Assert.Equal(1f, Assert.Single(tracks).RingBuffer.Tail.ObservedMask);
        }

        [Fact]
        public void TrackVelocity_IsComparableAcrossThirtyAndTwoHundredFortyHz()
        {
            static PointF Simulate(double stepMs, int steps)
            {
                var manager = new TrackManager
                {
                    ScreenWidth = 1000,
                    ScreenHeight = 1000,
                    VelocityResponseRatePerSecond = 20f,
                };
                var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
                DateTime start = new(2026, 7, 10, 12, 0, 0, DateTimeKind.Utc);
                IReadOnlyList<Track> tracks = Array.Empty<Track>();
                for (int i = 0; i <= steps; i++)
                {
                    double seconds = i * stepMs / 1000.0;
                    float x = 100f + (float)(200.0 * seconds); // 0.2 display widths / second
                    tracks = manager.Update([MakePrediction(x, 100)], roles, start.AddMilliseconds(i * stepMs));
                }
                return Assert.Single(tracks).Velocity;
            }

            PointF at30 = Simulate(1000.0 / 30.0, 30);
            PointF at240 = Simulate(1000.0 / 240.0, 240);

            Assert.InRange(at30.X, 0.15f, 0.25f);
            Assert.InRange(at240.X, 0.15f, 0.25f);
            Assert.InRange(Math.Abs(at30.X - at240.X), 0f, 0.04f);
        }

        [Fact]
        public void NegativeDisplayOrigin_NormalizesTrackIntoZeroToOneSpace()
        {
            var manager = new TrackManager
            {
                ScreenLeft = -1920,
                ScreenTop = 0,
                ScreenWidth = 1920,
                ScreenHeight = 1080,
            };
            var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
            DateTime now = DateTime.UtcNow;
            var prediction = MakePrediction(-980, 500, size: 40);

            Track track = Assert.Single(manager.Update([prediction], roles, now));
            TrackObservation observation = track.RingBuffer.Tail;
            Assert.InRange(observation.CxNorm, 0.49f, 0.51f);
            Assert.InRange(observation.CyNorm, 0.47f, 0.49f);
        }

        [Fact]
        public void MissingManifestClass_DefaultsToUnknownRole()
        {
            var manager = new TrackManager { ScreenWidth = 640, ScreenHeight = 640 };
            Track track = Assert.Single(manager.Update(
                [MakePrediction(100, 100, classId: 7)],
                new Dictionary<int, SemanticRole>(),
                DateTime.UtcNow));
            Assert.Equal(SemanticRole.Unknown, track.Role);
        }

        [Fact]
        public void RingBuffer_WrapsOldestFirst_TailIsNewest_AndCountCapsAtEight()
        {
            var buffer = new TrackRingBuffer();
            for (int i = 0; i < 12; i++)
            {
                buffer.Push(new TrackObservation
                {
                    CxNorm = i / 20f,
                    CyNorm = 0.5f,
                    WNorm = 0.1f,
                    HNorm = 0.2f,
                    Confidence = 0.9f,
                    ObservedMask = 1f,
                    DtSeconds = 0.016f,
                    AgeSeconds = 0f,
                });
            }

            Assert.Equal(TrackRingBuffer.Capacity, buffer.Count);
            Assert.Equal(11 / 20f, buffer.Tail.CxNorm, 6);
            float[] values = buffer.GetSequence().Select(sample => sample.CxNorm).ToArray();
            Assert.Equal(Enumerable.Range(4, 8).Select(i => i / 20f), values);
        }

        [Fact]
        public void GruNormalizationDefaults_FailClosedInsteadOfGuessingDatasetStatistics()
        {
            Assert.Throws<InvalidDataException>(() => new GruNormConstants().Validate());
        }

        [Fact]
        public void TemporalDeltaConversion_ExactlyInvertsTrainingTimesTwoTarget()
        {
            const float screenDelta = 0.125f;
            float trainingTarget = screenDelta * TemporalPredictor.TrainingDeltaScale;
            Assert.Equal(
                screenDelta,
                TemporalPredictor.ConvertModelDeltaToScreenFraction(trainingTarget),
                6);
        }

        [Fact]
        public void ObservedTrackDt_UsesPreviousLastSeenBeforeTimestampMutation()
        {
            var manager = new TrackManager { ScreenWidth = 640, ScreenHeight = 640 };
            var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
            DateTime first = new(2026, 7, 10, 12, 0, 0, DateTimeKind.Utc);

            Track track = Assert.Single(manager.Update([MakePrediction(100, 100)], roles, first));
            DateTime originalFirstSeen = track.FirstSeen;
            Track updated = Assert.Single(manager.Update(
                [MakePrediction(110, 100)],
                roles,
                first.AddMilliseconds(50)));

            Assert.Equal(0.050f, updated.RingBuffer.Tail.DtSeconds, 3);
            Assert.Equal(0f, updated.RingBuffer.Tail.AgeSeconds, 6);
            Assert.Equal(originalFirstSeen, updated.FirstSeen);
            Assert.Equal(first.AddMilliseconds(50), updated.LastSeen);
        }

        [Fact]
        public void CalibrationFeatureEncoding_MatchesPythonDetectionContextV2Formula()
        {
            Span<float> features = stackalloc float[CalibrationMlp.FeatureCount];
            CalibrationMlp.EncodeFeatures(
                features,
                rawConfidence: 0.8f,
                wNorm: 0.2f,
                hNorm: 0.4f,
                cxNorm: 0.75f,
                cyNorm: 0.25f,
                frameAgeMs: 125f);

            Assert.Equal(MathF.Log(4f), features[0], 5);
            Assert.Equal(MathF.Log(0.08f), features[1], 5);
            Assert.Equal(MathF.Log(2f), features[2], 5);
            Assert.Equal(MathF.Sqrt(0.25f * 0.25f + 0.25f * 0.25f) / 0.70710678f, features[3], 5);
            Assert.Equal(1.25f, features[4], 5);
            Assert.Equal(0f, features[5], 5);
        }

        [Fact]
        public void MovementFeatureEncoding_MatchesPythonPointingVelocityV1Order()
        {
            Span<float> features = stackalloc float[MovementMlp.FeatureCount];
            MovementMlp.EncodeFeatures(
                features,
                targetDx: 3f,
                targetDy: 4f,
                targetW: 20f,
                targetH: 30f,
                dtSeconds: 0.02f,
                currentSpeedPixPerMs: 1.5f,
                previousVx: -0.2f,
                previousVy: 0.3f);

            Assert.Equal(new[] { 3f, 4f, 5f, 1.5f, 30f, 0.02f, -0.2f, 0.3f }, features.ToArray());
        }

        [Fact]
        public void MovementContextChange_ResetsVelocityAndResidualState()
        {
            using var model = new MovementMlp();
            model.SetContext(10);

            static FieldInfo Field(string name) => typeof(MovementMlp).GetField(
                name,
                BindingFlags.Instance | BindingFlags.NonPublic)!;

            Field("_prevVx").SetValue(model, 0.4f);
            Field("_prevVy").SetValue(model, -0.3f);
            Field("_residualX").SetValue(model, 0.75f);
            Field("_residualY").SetValue(model, -0.5f);

            model.SetContext(11);

            Assert.Equal(0f, (float)Field("_prevVx").GetValue(model)!);
            Assert.Equal(0f, (float)Field("_prevVy").GetValue(model)!);
            Assert.Equal(0f, (float)Field("_residualX").GetValue(model)!);
            Assert.Equal(0f, (float)Field("_residualY").GetValue(model)!);
        }

        [Fact]
        public void Manifest_RejectsCompanionFeatureSchemaMismatch()
        {
            var calibration = ModelManifest.CreateFallback(
                "cal",
                "Calibration",
                new Dictionary<int, string> { [0] = "human" },
                512);
            calibration.CalibrationModelPath = "calibration.onnx";
            calibration.CalibrationFeatureSchema = "wrong";
            Assert.Throws<InvalidDataException>(() => calibration.Validate());

            var movement = ModelManifest.CreateFallback(
                "move",
                "Movement",
                new Dictionary<int, string> { [0] = "human" },
                512);
            movement.MovementModelPath = "movement.onnx";
            movement.MovementFeatureSchema = "wrong";
            Assert.Throws<InvalidDataException>(() => movement.Validate());
        }
    }
}
