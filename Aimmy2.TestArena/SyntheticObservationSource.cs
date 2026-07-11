using Aimmy2.AILogic;
using System.Drawing;

namespace Aimmy2.TestArena
{
    public enum SyntheticProfile
    {
        Clean,
        Noisy,
        Occluded,
        FalsePositiveStress,
    }

    public static class SyntheticProfileExtensions
    {
        public static SyntheticNoiseConfig Configuration(this SyntheticProfile profile) => profile switch
        {
            SyntheticProfile.Clean => SyntheticNoiseConfig.Clean,
            SyntheticProfile.Noisy => SyntheticNoiseConfig.Noisy,
            SyntheticProfile.Occluded => SyntheticNoiseConfig.Occluded,
            SyntheticProfile.FalsePositiveStress => SyntheticNoiseConfig.FalsePositiveStress,
            _ => throw new ArgumentOutOfRangeException(nameof(profile), profile, null),
        };
    }

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
        private int _frameIndex;

        public IReadOnlyList<AccessibilityObservation> CurrentObservations { get; private set; }
            = Array.Empty<AccessibilityObservation>();

        public int ActiveTracks { get; private set; }

        /// <summary>Last injected box size in physical screen pixels, for diagnostics.</summary>
        public (float Width, float Height) LastInjectedBoxSize { get; private set; }

        public SyntheticObservationSource(SyntheticNoiseConfig? noise = null)
        {
            _noise = noise ?? SyntheticNoiseConfig.Clean;
            _rng = new Random(_noise.Seed);
            _accessibilityObserver = new AccessibilityObserver(_trackManager);
        }

        /// <summary>
        /// Inject one frame of ground-truth observations into the tracker.
        /// Must be called on the UI thread because <paramref name="toScreen"/> uses WPF layout coordinates.
        ///
        /// Box dimensions are derived by projecting the four half-extent points through
        /// <paramref name="toScreen"/> so they are correct at any Windows DPI scale.
        /// Ground-truth <see cref="ArenaTarget.Id"/> is retained only for drop decisions here
        /// and is never passed to <see cref="TrackManager"/>; association sees only box/class/conf.
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

            // Deterministic contiguous occlusion: all observations are dropped during
            // [StartFrame, StartFrame + DurationFrames). The gap is entered only after
            // targets have been tracked for StartFrame frames.
            bool inContiguousOcclusion = _noise.ContiguousOcclusionStartFrame > 0
                && _noise.ContiguousOcclusionDurationFrames > 0
                && _frameIndex >= _noise.ContiguousOcclusionStartFrame
                && _frameIndex < _noise.ContiguousOcclusionStartFrame + _noise.ContiguousOcclusionDurationFrames;

            var predictions = new List<Prediction>();

            if (!inContiguousOcclusion)
            {
                (float Width, float Height) lastBox = default;

                foreach (ArenaTarget target in targets.Where(t => t.Visible))
                {
                    // Independent per-frame random dropout.
                    if (_noise.DropProbability > 0.0 && _rng.NextDouble() < _noise.DropProbability)
                        continue;

                    // Project centre through PointToScreen — physical pixel coords.
                    PointF center = toScreen(target.Position);

                    // Derive width/height by projecting the half-extent points so the box
                    // is correct at any Windows DPI scale (not just the WPF DIP size).
                    PointF right  = toScreen(new System.Windows.Point(
                        target.Position.X + target.Size.Width  / 2.0, target.Position.Y));
                    PointF bottom = toScreen(new System.Windows.Point(
                        target.Position.X, target.Position.Y + target.Size.Height / 2.0));

                    float baseWidth  = Math.Abs(right.X  - center.X) * 2f;
                    float baseHeight = Math.Abs(bottom.Y - center.Y) * 2f;

                    float cx = center.X + (float)((_rng.NextDouble() * 2 - 1) * _noise.PositionNoiseMaxPx);
                    float cy = center.Y + (float)((_rng.NextDouble() * 2 - 1) * _noise.PositionNoiseMaxPx);
                    float w  = baseWidth  * (1f + (float)((_rng.NextDouble() * 2 - 1) * _noise.SizeNoise));
                    float h  = baseHeight * (1f + (float)((_rng.NextDouble() * 2 - 1) * _noise.SizeNoise));
                    float conf = Math.Clamp(
                        _noise.BaseConfidence + (float)((_rng.NextDouble() * 2 - 1) * _noise.ConfidenceNoise),
                        0.05f, 1.0f);

                    int classId = target.Kind switch
                    {
                        TargetKind.Friendly => 1,
                        TargetKind.Player   => 2,
                        _                   => 0,
                    };

                    var screenRect = new RectangleF(cx - w / 2f, cy - h / 2f, w, h);
                    predictions.Add(new Prediction
                    {
                        Rectangle       = screenRect,
                        ScreenRectangle = screenRect,
                        Confidence      = conf,
                        ClassId         = classId,
                        ClassName       = classId switch { 1 => "friendly", 2 => "player", _ => "enemy" },
                    });

                    lastBox = (w, h);
                }

                for (int i = 0; i < _noise.FalsePositiveCount; i++)
                {
                    float fpX  = screenLeft + (float)(_rng.NextDouble() * screenWidth);
                    float fpY  = screenTop  + (float)(_rng.NextDouble() * screenHeight);
                    float conf = Math.Clamp(
                        _noise.BaseConfidence + (float)((_rng.NextDouble() * 2 - 1) * _noise.ConfidenceNoise),
                        0.05f, 1.0f);
                    var fpRect = new RectangleF(fpX - 30f, fpY - 50f, 60f, 100f);
                    predictions.Add(new Prediction
                    {
                        Rectangle       = fpRect,
                        ScreenRectangle = fpRect,
                        Confidence      = conf,
                        ClassId         = 0,
                        ClassName       = "enemy",
                    });
                }

                if (lastBox != default) LastInjectedBoxSize = lastBox;
            }

            var tracks = _trackManager.Update(predictions, ClassRoles, frameTime);
            CurrentObservations = _accessibilityObserver.Observe(frameTime);
            ActiveTracks = tracks.Count;
            _frameIndex++;
        }

        /// <summary>True during the deterministic contiguous occlusion window.</summary>
        public bool InContiguousOcclusion =>
            _noise.ContiguousOcclusionStartFrame > 0
            && _noise.ContiguousOcclusionDurationFrames > 0
            && _frameIndex >= _noise.ContiguousOcclusionStartFrame
            && _frameIndex < _noise.ContiguousOcclusionStartFrame + _noise.ContiguousOcclusionDurationFrames;

        public void Dispose() { }
    }

    /// <summary>
    /// Noise and FP injection parameters for one SyntheticTracking profile.
    /// All parameters use bounded uniform noise (not Gaussian) — see NoiseDistribution in reports.
    /// Parameters with defaults are backward-compatible with existing call sites.
    /// </summary>
    public sealed record SyntheticNoiseConfig(
        double PositionNoiseMaxPx,
        double SizeNoise,
        float  BaseConfidence,
        float  ConfidenceNoise,
        double DropProbability,
        int    FalsePositiveCount,
        int    Seed,
        string ProfileName                   = "Custom",
        int    ContiguousOcclusionStartFrame  = 0,
        int    ContiguousOcclusionDurationFrames = 0)
    {
        /// <summary>
        /// Tight position noise, no drops, no FPs.
        /// Establishes tracker correctness. Default for automated reports.
        /// </summary>
        public static readonly SyntheticNoiseConfig Clean =
            new(PositionNoiseMaxPx: 1.5, SizeNoise: 0.01f, BaseConfidence: 0.90f,
                ConfidenceNoise: 0.03f, DropProbability: 0.0, FalsePositiveCount: 0,
                Seed: 42, ProfileName: "Clean");

        /// <summary>
        /// High position noise and occasional independent frame drops.
        /// Tests Kalman stability under measurement noise.
        /// </summary>
        public static readonly SyntheticNoiseConfig Noisy =
            new(PositionNoiseMaxPx: 8.0, SizeNoise: 0.05f, BaseConfidence: 0.75f,
                ConfidenceNoise: 0.10f, DropProbability: 0.05, FalsePositiveCount: 0,
                Seed: 42, ProfileName: "Noisy");

        /// <summary>
        /// 30 % independent per-frame drop probability. Resembles bursty detection noise,
        /// not a realistic contiguous occlusion. Use <see cref="Occluded"/> for that.
        /// </summary>
        public static readonly SyntheticNoiseConfig RandomDropout =
            new(PositionNoiseMaxPx: 3.0, SizeNoise: 0.02f, BaseConfidence: 0.80f,
                ConfidenceNoise: 0.05f, DropProbability: 0.30, FalsePositiveCount: 0,
                Seed: 42, ProfileName: "RandomDropout");

        /// <summary>
        /// Deterministic single contiguous gap: tracks established for 90 frames, then all
        /// observations dropped for 45 consecutive frames, then visible again for reacquisition.
        /// At 60 fps: ~1.5 s tracking, ~0.75 s gap. Tests prediction continuity and track-ID
        /// preservation across a known occlusion event.
        /// </summary>
        public static readonly SyntheticNoiseConfig Occluded =
            new(PositionNoiseMaxPx: 1.5, SizeNoise: 0.01f, BaseConfidence: 0.90f,
                ConfidenceNoise: 0.03f, DropProbability: 0.0, FalsePositiveCount: 0,
                Seed: 42, ProfileName: "Occluded",
                ContiguousOcclusionStartFrame: 90,
                ContiguousOcclusionDurationFrames: 45);

        /// <summary>Clean track plus 3 random FP injections per frame. Tests association resistance.</summary>
        public static readonly SyntheticNoiseConfig FalsePositiveStress =
            new(PositionNoiseMaxPx: 1.5, SizeNoise: 0.01f, BaseConfidence: 0.90f,
                ConfidenceNoise: 0.03f, DropProbability: 0.0, FalsePositiveCount: 3,
                Seed: 42, ProfileName: "FalsePositiveStress");
    }
}
