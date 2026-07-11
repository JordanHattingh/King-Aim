using System.Drawing;
using Aimmy2.AILogic;
using Xunit;

namespace Aimmy2.Tests
{
    public sealed class Da1ProductionTests
    {
        private static Prediction MakePrediction(
            float x,
            float y,
            float size = 40f,
            float confidence = 0.9f,
            int classId = 0,
            PlayerKeypoints? keypoints = null) => new()
        {
            Rectangle = new RectangleF(x, y, size, size),
            ScreenRectangle = new RectangleF(x, y, size, size),
            Confidence = confidence,
            ClassId = classId,
            ClassName = classId == 0 ? "enemy" : "friendly",
            ScreenCenterX = x + size / 2f,
            ScreenCenterY = y + size / 2f,
            CenterXTranslated = (x + size / 2f) / 1000f,
            CenterYTranslated = (y + size / 2f) / 1000f,
            Keypoints = keypoints,
        };

        private static PlayerKeypoints MakePose(float left, float top, bool reversed = false)
        {
            float[] ys = reversed
                ? [top + 34f, top + 26f, top + 18f, top + 10f]
                : [top + 10f, top + 18f, top + 26f, top + 34f];

            return new PlayerKeypoints
            {
                Head = new Keypoint { X = left + 20f, Y = ys[0], Visibility = 0.95f },
                Neck = new Keypoint { X = left + 20f, Y = ys[1], Visibility = 0.95f },
                Chest = new Keypoint { X = left + 20f, Y = ys[2], Visibility = 0.95f },
                Hip = new Keypoint { X = left + 20f, Y = ys[3], Visibility = 0.95f },
            };
        }

        [Fact]
        public void CoordinateContracts_RoundTripNegativeDesktopOrigin()
        {
            var desktop = new DesktopPixel(-960f, 540f);

            ScreenFraction fraction = desktop.ToScreenFraction(-1920, 0, 1920, 1080);
            DesktopPixel roundTrip = fraction.ToDesktopPixel(-1920, 0, 1920, 1080);

            Assert.Equal(0.5f, fraction.X, 6);
            Assert.Equal(0.5f, fraction.Y, 6);
            Assert.Equal(desktop.X, roundTrip.X, 5);
            Assert.Equal(desktop.Y, roundTrip.Y, 5);
        }

        [Fact]
        public void TrackManager_CarriesPoseForExactlyConfiguredMissingPoseFrames()
        {
            var manager = new TrackManager
            {
                ScreenWidth = 1000,
                ScreenHeight = 1000,
                KeypointCarryForwardFrames = 3,
            };
            var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
            DateTime now = new(2026, 7, 11, 6, 0, 0, DateTimeKind.Utc);

            Track track = Assert.Single(manager.Update(
                [MakePrediction(100, 100, keypoints: MakePose(100, 100))],
                roles,
                now));
            Assert.NotNull(track.Keypoints);

            for (int frame = 1; frame <= 3; frame++)
            {
                track = Assert.Single(manager.Update(
                    [MakePrediction(100 + frame, 100, keypoints: null)],
                    roles,
                    now.AddMilliseconds(frame * 16)));
                Assert.NotNull(track.Keypoints);
            }

            track = Assert.Single(manager.Update(
                [MakePrediction(104, 100, keypoints: null)],
                roles,
                now.AddMilliseconds(64)));
            Assert.Null(track.Keypoints);
        }

        [Fact]
        public void TrackManager_ExtrapolatesLostBoxThenFreezesAfterHalfExpiry()
        {
            var manager = new TrackManager
            {
                ScreenWidth = 1000,
                ScreenHeight = 1000,
                MaxLostSeconds = 0.6,
                UsePredictiveBoundingBox = true,
                VelocityResponseRatePerSecond = 100f,
            };
            var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
            DateTime start = new(2026, 7, 11, 6, 0, 0, DateTimeKind.Utc);

            _ = manager.Update([MakePrediction(100, 100)], roles, start);
            Track observed = Assert.Single(manager.Update(
                [MakePrediction(140, 100)],
                roles,
                start.AddMilliseconds(100)));
            float observedX = observed.BoundingBox.X;

            Track extrapolated = Assert.Single(manager.Update(
                [],
                roles,
                start.AddMilliseconds(150)));
            Assert.True(extrapolated.BoundingBoxIsExtrapolated);
            Assert.True(extrapolated.BoundingBox.X > observedX);
            float extrapolatedX = extrapolated.BoundingBox.X;

            Track frozen = Assert.Single(manager.Update(
                [],
                roles,
                start.AddMilliseconds(500)));
            Assert.False(frozen.BoundingBoxIsExtrapolated);
            Assert.Equal(extrapolatedX, frozen.BoundingBox.X, 4);
        }

        [Fact]
        public void AccessibilityObserver_SortsRolesAndPrefersEnemyPrimary()
        {
            var manager = new TrackManager
            {
                ScreenWidth = 1000,
                ScreenHeight = 1000,
            };
            var roles = new Dictionary<int, SemanticRole>
            {
                [0] = SemanticRole.Enemy,
                [1] = SemanticRole.Friendly,
            };
            DateTime now = new(2026, 7, 11, 6, 0, 0, DateTimeKind.Utc);

            _ = manager.Update(
                [
                    MakePrediction(700, 100, confidence: 0.70f, classId: 1),
                    MakePrediction(100, 100, confidence: 0.65f, classId: 0),
                ],
                roles,
                now);

            var observer = new AccessibilityObserver(manager);
            IReadOnlyList<AccessibilityObservation> observations = observer.Observe(now);
            AccessibilityObservation? primary = observer.PrimaryTarget();

            Assert.Equal(2, observations.Count);
            Assert.Equal(SemanticRole.Enemy, observations[0].Role);
            Assert.Equal(SemanticRole.Friendly, observations[1].Role);
            Assert.NotNull(primary);
            Assert.Equal(SemanticRole.Enemy, primary!.Role);
        }

        [Fact]
        public void GruAssociationPrior_LowersMotionCostWhenPredictionMatchesDetection()
        {
            var manager = new TrackManager
            {
                ScreenWidth = 1000,
                ScreenHeight = 1000,
            };
            var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
            DateTime now = new(2026, 7, 11, 6, 0, 0, DateTimeKind.Utc);
            Track track = Assert.Single(manager.Update([MakePrediction(100, 100)], roles, now));
            track.GruPredictedCenter = (520f, 120f);
            Prediction detection = MakePrediction(500, 100);

            var noGru = new TrackAssociationSettings
            {
                GruAssociationWeight = 0f,
                MaximumCenterDistanceScreenFraction = 1f,
                MaximumAcceptedCost = 2f,
            };
            var withGru = new TrackAssociationSettings
            {
                GruAssociationWeight = 1f,
                MaximumCenterDistanceScreenFraction = 1f,
                MaximumAcceptedCost = 2f,
            };
            var strategy = new GlobalCostTrackAssociation();

            float kalmanCost = Assert.Single(strategy.Associate(
                [track], [detection], now.AddMilliseconds(16), 1000, 1000, noGru)).Cost;
            float gruCost = Assert.Single(strategy.Associate(
                [track], [detection], now.AddMilliseconds(16), 1000, 1000, withGru)).Cost;

            Assert.True(gruCost < kalmanCost);
        }

        [Fact]
        public void PoseSimilarityWeight_PrefersMatchingSkeletonGeometry()
        {
            var manager = new TrackManager
            {
                ScreenWidth = 1000,
                ScreenHeight = 1000,
            };
            var roles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
            DateTime now = new(2026, 7, 11, 6, 0, 0, DateTimeKind.Utc);
            Track track = Assert.Single(manager.Update(
                [MakePrediction(100, 100, keypoints: MakePose(100, 100))],
                roles,
                now));

            Prediction reversed = MakePrediction(100, 100, keypoints: MakePose(100, 100, reversed: true));
            Prediction matching = MakePrediction(100, 100, keypoints: MakePose(100, 100));
            var settings = new TrackAssociationSettings
            {
                PoseSimilarityWeight = 1f,
                MaximumAcceptedCost = 2f,
            };
            var strategy = new GlobalCostTrackAssociation();

            TrackDetectionMatch match = Assert.Single(strategy.Associate(
                [track],
                [reversed, matching],
                now.AddMilliseconds(16),
                1000,
                1000,
                settings));

            Assert.Equal(1, match.DetectionIndex);
        }
    }
}
