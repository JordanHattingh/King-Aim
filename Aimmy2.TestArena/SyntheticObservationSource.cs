using Aimmy2.AILogic;
using System.Drawing;

namespace Aimmy2.TestArena
{
    /// <summary>
    /// Converts ArenaTarget ground truth into synthetic Predictions and feeds them
    /// directly into a standalone TrackManager + Kalman + AccessibilityObserver pipeline.
    /// YOLO is bypassed entirely — this is the observation injection path for
    /// SyntheticTracking scenarios so tracker_metrics_gated can be true.
    /// </summary>
    public sealed class SyntheticObservationSource : IDisposable
    {
        private static readonly Dictionary<int, SemanticRole> ClassRoles = new()
        {
            { 0, SemanticRole.Enemy },
            { 1, SemanticRole.Friendly },
            { 2, SemanticRole.Player },
        };

        private readonly TrackManager _trackManager = new();
        private readonly AccessibilityObserver _accessibilityObserver;
        private readonly SyntheticNoiseConfig _noise;
        private readonly Random _rng;

        public IReadOnlyList<AccessibilityObservation> CurrentObservations { get; private set; }
            = Array.Empty<AccessibilityObservation>();

        public int ActiveTracks { get; private set; }

        public SyntheticObservationSource(SyntheticNoiseConfig? noise = null)
        {
            _noise = noise ?? SyntheticNoiseConfig.Clean;
            _rng = new Random(_noise.Seed);
            _accessibilityObserver = new AccessibilityObserver(_trackManager);
        }

        /// <summary>
        /// Inject one frame of ground-truth observations into the tracker.
        /// Must be called on the UI thread because <paramref name="toScreen"/> uses WPF layout coordinates.
        /// </summary>
        public void Inject(
            IReadOnlyList<ArenaTarget> targets,
            DateTime frameTime,
            Func<System.Windows.Point, PointF> toScreen,
            int screenLeft,
            int screenTop,
            int screenWidth,
            int screenHeight)
        {
            _trackManager.ScreenLeft = screenLeft;
            _trackManager.ScreenTop = screenTop;
            _trackManager.ScreenWidth = screenWidth;
            _trackManager.ScreenHeight = screenHeight;

            var predictions = new List<Prediction>();

            foreach (ArenaTarget target in targets.Where(t => t.Visible))
            {
                if (_noise.DropProbability > 0.0 && _rng.NextDouble() < _noise.DropProbability)
                    continue;

                PointF center = toScreen(target.Position);

                float cx = center.X + (float)((_rng.NextDouble() * 2 - 1) * _noise.PositionNoisePx);
                float cy = center.Y + (float)((_rng.NextDouble() * 2 - 1) * _noise.PositionNoisePx);
                float w = (float)target.Size.Width * (1f + (float)((_rng.NextDouble() * 2 - 1) * _noise.SizeNoise));
                float h = (float)target.Size.Height * (1f + (float)((_rng.NextDouble() * 2 - 1) * _noise.SizeNoise));
                float conf = Math.Clamp(
                    _noise.BaseConfidence + (float)((_rng.NextDouble() * 2 - 1) * _noise.ConfidenceNoise),
                    0.05f, 1.0f);

                int classId = target.Kind switch
                {
                    TargetKind.Friendly => 1,
                    TargetKind.Player => 2,
                    _ => 0,
                };

                var screenRect = new RectangleF(cx - w / 2f, cy - h / 2f, w, h);
                predictions.Add(new Prediction
                {
                    Rectangle = screenRect,
                    ScreenRectangle = screenRect,
                    Confidence = conf,
                    ClassId = classId,
                    ClassName = classId switch { 1 => "friendly", 2 => "player", _ => "enemy" },
                });
            }

            for (int i = 0; i < _noise.FalsePositiveCount; i++)
            {
                float fpX = screenLeft + (float)(_rng.NextDouble() * screenWidth);
                float fpY = screenTop + (float)(_rng.NextDouble() * screenHeight);
                float conf = Math.Clamp(
                    _noise.BaseConfidence + (float)((_rng.NextDouble() * 2 - 1) * _noise.ConfidenceNoise),
                    0.05f, 1.0f);
                var fpRect = new RectangleF(fpX - 30f, fpY - 50f, 60f, 100f);
                predictions.Add(new Prediction
                {
                    Rectangle = fpRect,
                    ScreenRectangle = fpRect,
                    Confidence = conf,
                    ClassId = 0,
                    ClassName = "enemy",
                });
            }

            var tracks = _trackManager.Update(predictions, ClassRoles, frameTime);
            CurrentObservations = _accessibilityObserver.Observe(frameTime);
            ActiveTracks = tracks.Count;
        }

        public void Dispose() { }
    }

    /// <summary>
    /// Noise and FP injection parameters for one SyntheticTracking profile.
    /// </summary>
    public sealed record SyntheticNoiseConfig(
        double PositionNoisePx,
        double SizeNoise,
        float BaseConfidence,
        float ConfidenceNoise,
        double DropProbability,
        int FalsePositiveCount,
        int Seed)
    {
        /// <summary>Tight position noise, no drops, no FPs. Default for automated reports.</summary>
        public static readonly SyntheticNoiseConfig Clean =
            new(PositionNoisePx: 1.5, SizeNoise: 0.01f, BaseConfidence: 0.90f,
                ConfidenceNoise: 0.03f, DropProbability: 0.0, FalsePositiveCount: 0, Seed: 42);

        /// <summary>High position noise and occasional detection drops.</summary>
        public static readonly SyntheticNoiseConfig Noisy =
            new(PositionNoisePx: 8.0, SizeNoise: 0.05f, BaseConfidence: 0.75f,
                ConfidenceNoise: 0.10f, DropProbability: 0.05, FalsePositiveCount: 0, Seed: 42);

        /// <summary>30 % per-frame drop probability simulates heavy occlusion.</summary>
        public static readonly SyntheticNoiseConfig Occluded =
            new(PositionNoisePx: 3.0, SizeNoise: 0.02f, BaseConfidence: 0.80f,
                ConfidenceNoise: 0.05f, DropProbability: 0.30, FalsePositiveCount: 0, Seed: 42);

        /// <summary>Clean track plus 3 random FP injections per frame.</summary>
        public static readonly SyntheticNoiseConfig FalsePositiveStress =
            new(PositionNoisePx: 1.5, SizeNoise: 0.01f, BaseConfidence: 0.90f,
                ConfidenceNoise: 0.03f, DropProbability: 0.0, FalsePositiveCount: 3, Seed: 42);
    }
}
