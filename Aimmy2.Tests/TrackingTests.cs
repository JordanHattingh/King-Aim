using System.Drawing;
using Aimmy2.AILogic;
using Xunit;

namespace Aimmy2.Tests
{
    public class TrackingTests
    {
        private static readonly Dictionary<int, SemanticRole> ClassRoles = new()
        {
            { 0, SemanticRole.Enemy },
            { 1, SemanticRole.Player },
            { 2, SemanticRole.Friendly },
        };

        private static Prediction MakePrediction(int classId, string className, float x, float y, float size = 40f, float confidence = 0.9f) =>
            new()
            {
                Rectangle = new RectangleF(x, y, size, size),
                Confidence = confidence,
                ClassId = classId,
                ClassName = className,
                CenterXTranslated = (x + size / 2f) / 640f,
                CenterYTranslated = (y + size / 2f) / 640f,
                ScreenCenterX = x + size / 2f,
                ScreenCenterY = y + size / 2f,
            };

        [Fact]
        public void PlayerDetection_CreatesTrack()
        {
            var manager = new TrackManager();
            var detections = new List<Prediction> { MakePrediction(1, "player", 100, 100) };

            var tracks = manager.Update(detections, ClassRoles, DateTime.UtcNow);

            Assert.Single(tracks);
            Assert.Equal(SemanticRole.Player, tracks[0].Role);
            Assert.Equal(1, manager.PlayerCount);
        }

        [Fact]
        public void TrackId_SurvivesShortDetectionGap()
        {
            var manager = new TrackManager { MaxLostSeconds = 1.0 };
            var now = DateTime.UtcNow;

            var tracks = manager.Update(new List<Prediction> { MakePrediction(0, "enemy", 100, 100) }, ClassRoles, now);
            int trackId = tracks[0].TrackId;

            // Simulate a gap just under MaxLostSeconds at a normal ~60fps cadence.
            for (int i = 1; i <= 50; i++)
            {
                now = now.AddMilliseconds(16);
                tracks = manager.Update(new List<Prediction>(), ClassRoles, now);
            }

            Assert.Contains(tracks, t => t.TrackId == trackId);

            // Push past MaxLostSeconds — should now be dropped.
            now = now.AddMilliseconds(300);
            tracks = manager.Update(new List<Prediction>(), ClassRoles, now);
            Assert.DoesNotContain(tracks, t => t.TrackId == trackId);
        }

        [Fact]
        public void TrackId_SurvivesShortDetectionGap_EvenAtLowFramerate()
        {
            // At low FPS, a fixed missed-frame-count rule would previously drop the track almost
            // instantly (e.g. 5 missed frames at 5fps = 1 second gone). Time-based expiry must
            // keep the track alive for MaxLostSeconds regardless of how few frames that spans.
            var manager = new TrackManager { MaxLostSeconds = 1.0 };
            var now = DateTime.UtcNow;

            var tracks = manager.Update(new List<Prediction> { MakePrediction(0, "enemy", 100, 100) }, ClassRoles, now);
            int trackId = tracks[0].TrackId;

            // Simulate ~5fps: only 3 missed frames, but 600ms of wall-clock time — under MaxLostSeconds.
            for (int i = 1; i <= 3; i++)
            {
                now = now.AddMilliseconds(200);
                tracks = manager.Update(new List<Prediction>(), ClassRoles, now);
            }

            Assert.Contains(tracks, t => t.TrackId == trackId);
        }

        [Fact]
        public void EnemyOnly_NeverSelectsFriendly()
        {
            var manager = new TrackManager();
            var now = DateTime.UtcNow;
            var detections = new List<Prediction>
            {
                MakePrediction(2, "friendly", 320, 320), // dead center, closest possible
                MakePrediction(0, "enemy", 50, 50),
            };

            var tracks = manager.Update(detections, ClassRoles, now);
            var selector = new TargetSelector();

            var result = selector.Select(tracks, TargetMode.EnemyOnly, SemanticRole.Enemy, null, new PointF(320, 320), 320f);

            Assert.NotNull(result.SelectedTrack);
            Assert.Equal(SemanticRole.Enemy, result.SelectedTrack!.Role);
        }

        [Fact]
        public void SelectedEnemy_DoesNotFlickerBetweenTracks()
        {
            var manager = new TrackManager();
            var selector = new TargetSelector();
            var now = DateTime.UtcNow;

            int? firstSelected = null;
            for (int i = 0; i < 10; i++)
            {
                now = now.AddMilliseconds(16);
                var detections = new List<Prediction> { MakePrediction(0, "enemy", 100 + i, 100) };
                var tracks = manager.Update(detections, ClassRoles, now);
                var result = selector.Select(tracks, TargetMode.EnemyOnly, SemanticRole.Enemy, null, new PointF(320, 320), 320f);

                Assert.NotNull(result.SelectedTrack);
                firstSelected ??= result.SelectedTrack!.TrackId;
                Assert.Equal(firstSelected, result.SelectedTrack!.TrackId);
            }
        }

        [Fact]
        public void PersistentChallenger_EventuallySwitchesSelection()
        {
            var manager = new TrackManager();
            var selector = new TargetSelector { ConfirmationFrames = 3 };
            var now = DateTime.UtcNow;

            // Frame 0: only the "current" far enemy is present, becomes selected.
            now = now.AddMilliseconds(16);
            var tracks = manager.Update(
                new List<Prediction> { MakePrediction(0, "enemy_far", 20, 20) },
                ClassRoles, now);
            var result = selector.Select(tracks, TargetMode.EnemyOnly, SemanticRole.Enemy, null, new PointF(320, 320), 320f);
            int originalId = result.SelectedTrack!.TrackId;

            // Now introduce a much closer challenger for N+1 frames.
            TargetSelectionResult? last = null;
            for (int i = 0; i < 4; i++)
            {
                now = now.AddMilliseconds(16);
                var detections = new List<Prediction>
                {
                    MakePrediction(0, "enemy_far", 20, 20),
                    MakePrediction(0, "enemy_near", 310, 310),
                };
                tracks = manager.Update(detections, ClassRoles, now);
                last = selector.Select(tracks, TargetMode.EnemyOnly, SemanticRole.Enemy, null, new PointF(320, 320), 320f);
            }

            Assert.NotNull(last!.SelectedTrack);
            Assert.NotEqual(originalId, last.SelectedTrack!.TrackId);
        }
    }
}
