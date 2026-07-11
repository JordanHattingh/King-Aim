using System.Windows;

namespace Aimmy2.TestArena
{
    public enum TargetKind
    {
        Player,
        Enemy,
        Friendly,
    }

    /// <summary>
    /// Which type of benchmark produced this report.
    ///
    /// GameplayReplay: real game frames fed through the full YOLO + tracker pipeline.
    ///   detector_metrics_gated = true.  This is the production accuracy gate.
    ///
    /// SyntheticTracking: synthetic coloured rectangles rendered in the arena.
    ///   detector_metrics_gated = false (rectangles are OOD for a model trained on
    ///   real gameplay, so zero detections are expected).
    ///   TODO: bypass YOLO and inject synthetic observations directly into TrackManager
    ///   so that tracker_metrics_gated can be set to true.
    /// </summary>
    public enum BenchmarkDomain
    {
        GameplayReplay,
        SyntheticTracking,
    }

    public sealed class ArenaTarget
    {
        public required string Id { get; init; }
        public required TargetKind Kind { get; init; }
        public Point Position { get; set; }
        public Size Size { get; init; } = new(60, 100);
        public bool Visible { get; set; } = true;
    }

    public enum ScenarioKind
    {
        // --- Synthetic tracking scenarios (BenchmarkDomain.SyntheticTracking) ---
        // These render coloured rectangles to exercise the Kalman + GRU tracker.
        // YOLO detects nothing on these inputs (OOD).  Once direct observation
        // injection is implemented, tracker_metrics_gated will become true.
        StaticEnemy,
        MovingEnemyHorizontal,
        MovingEnemyVertical,
        MovingEnemyDiagonal,
        DirectionReversal,
        StopStart,
        TemporaryOcclusion,
        // Split versions of TemporaryOcclusion with well-specified gate expectations:
        //   ShortOcclusion gap (0.5 s) < MaxLostSeconds (0.6 s) → same TrackId expected.
        //   LongOcclusion  gap (1.0 s) > MaxLostSeconds (0.6 s) → new TrackId is acceptable.
        ShortOcclusion,
        LongOcclusion,
        TwoEnemies,
        EnemyAndFriendlyCrossing,
        MultipleEnemiesCrossing,
        HighSpeedCross,
        SuddenAcceleration,
        ThreeEnemiesConverging,
        NoTarget,

        // --- Gameplay replay scenarios (BenchmarkDomain.GameplayReplay) ---
        // Future: feed real gameplay frames through the full pipeline so detector
        // accuracy metrics are meaningful.
        // TODO: implement image-sequence loading and frame injection.
        GameplayReplay,
    }

    public static class ScenarioKindExtensions
    {
        public static BenchmarkDomain Domain(this ScenarioKind kind) => kind switch
        {
            ScenarioKind.GameplayReplay => BenchmarkDomain.GameplayReplay,
            _ => BenchmarkDomain.SyntheticTracking,
        };

        /// <summary>
        /// Whether the detector runs on real inputs for this scenario and its
        /// output accuracy metrics should be used as a gate.
        /// </summary>
        public static bool DetectorMetricsGated(this ScenarioKind kind) =>
            kind.Domain() == BenchmarkDomain.GameplayReplay;

        /// <summary>
        /// Whether tracker metrics (identity switches, track losses, reacquisition)
        /// are gated for this scenario.
        /// SyntheticTracking scenarios inject observations directly into TrackManager
        /// (YOLO bypassed), so tracker metrics are actionable.
        /// </summary>
        public static bool TrackerMetricsGated(this ScenarioKind kind) =>
            kind.Domain() == BenchmarkDomain.SyntheticTracking;
    }

    public sealed class Scenario
    {
        private readonly double _width;
        private readonly double _height;
        private double _elapsedSeconds;

        public ScenarioKind Kind { get; }
        public IReadOnlyList<ArenaTarget> Targets { get; }

        public Scenario(ScenarioKind kind, double arenaWidth, double arenaHeight)
        {
            Kind = kind;
            _width = arenaWidth;
            _height = arenaHeight;
            Targets = BuildInitialTargets(kind, arenaWidth, arenaHeight);
        }

        private static List<ArenaTarget> BuildInitialTargets(ScenarioKind kind, double w, double h) => kind switch
        {
            ScenarioKind.NoTarget
                or ScenarioKind.GameplayReplay => new List<ArenaTarget>(),

            ScenarioKind.TwoEnemies
                or ScenarioKind.MultipleEnemiesCrossing
                or ScenarioKind.HighSpeedCross => new List<ArenaTarget>
            {
                new() { Id = "enemy1", Kind = TargetKind.Enemy, Position = new Point(w * 0.20, h * 0.42) },
                new() { Id = "enemy2", Kind = TargetKind.Enemy, Position = new Point(w * 0.80, h * 0.58) },
            },

            ScenarioKind.EnemyAndFriendlyCrossing => new List<ArenaTarget>
            {
                new() { Id = "enemy1", Kind = TargetKind.Enemy, Position = new Point(w * 0.2, h * 0.5) },
                new() { Id = "friendly1", Kind = TargetKind.Friendly, Position = new Point(w * 0.8, h * 0.5) },
            },

            ScenarioKind.ThreeEnemiesConverging => new List<ArenaTarget>
            {
                new() { Id = "enemy1", Kind = TargetKind.Enemy, Position = new Point(w * 0.12, h * 0.15) },
                new() { Id = "enemy2", Kind = TargetKind.Enemy, Position = new Point(w * 0.88, h * 0.15) },
                new() { Id = "enemy3", Kind = TargetKind.Enemy, Position = new Point(w * 0.50, h * 0.88) },
            },

            _ => new List<ArenaTarget>
            {
                new() { Id = "enemy1", Kind = TargetKind.Enemy, Position = new Point(w * 0.5, h * 0.5) },
            },
        };

        public void Advance(double dtSeconds)
        {
            _elapsedSeconds += dtSeconds;

            switch (Kind)
            {
                case ScenarioKind.StaticEnemy:
                case ScenarioKind.NoTarget:
                case ScenarioKind.GameplayReplay:
                    break;

                case ScenarioKind.MovingEnemyHorizontal:
                    Move(Targets[0], x: Oscillate(_width * 0.15, _width * 0.85, period: 4.0), y: null);
                    break;

                case ScenarioKind.MovingEnemyVertical:
                    Move(Targets[0], x: null, y: Oscillate(_height * 0.15, _height * 0.85, period: 4.0));
                    break;

                case ScenarioKind.MovingEnemyDiagonal:
                    Move(
                        Targets[0],
                        x: Oscillate(_width * 0.15, _width * 0.85, period: 5.0),
                        y: Oscillate(_height * 0.15, _height * 0.85, period: 3.0));
                    break;

                case ScenarioKind.DirectionReversal:
                    // Triangle-wave horizontal motion with a short period so direction flips are frequent.
                    Move(Targets[0], x: Triangle(_width * 0.15, _width * 0.85, period: 1.5), y: null);
                    break;

                case ScenarioKind.StopStart:
                    AdvanceStopStart(Targets[0]);
                    break;

                case ScenarioKind.TemporaryOcclusion:
                    Move(Targets[0], x: Oscillate(_width * 0.2, _width * 0.8, period: 4.0), y: null);
                    Targets[0].Visible = (_elapsedSeconds % 4.0) < 3.0;
                    break;

                case ScenarioKind.ShortOcclusion:
                    // Gap 0.5 s every 5.0 s — well under the 0.6 s persistence window.
                    // Gate expectation: track survives, same TrackId returned on reappearance.
                    Move(Targets[0], x: Oscillate(_width * 0.2, _width * 0.8, period: 5.0), y: null);
                    Targets[0].Visible = (_elapsedSeconds % 5.0) < 4.5;
                    break;

                case ScenarioKind.LongOcclusion:
                    // Gap 1.0 s every 4.0 s — well beyond the 0.6 s persistence window.
                    // Gate expectation: old track expires, new TrackId on reappearance is acceptable,
                    // but ghost frames must stop within 0.6 s and reacquisition must be prompt.
                    Move(Targets[0], x: Oscillate(_width * 0.2, _width * 0.8, period: 4.0), y: null);
                    Targets[0].Visible = (_elapsedSeconds % 4.0) < 3.0;
                    break;

                case ScenarioKind.TwoEnemies:
                    Move(Targets[0], x: Oscillate(_width * 0.15, _width * 0.45, period: 3.0), y: null);
                    Move(Targets[1], x: Oscillate(_width * 0.55, _width * 0.85, period: 3.5), y: null);
                    break;

                case ScenarioKind.EnemyAndFriendlyCrossing:
                    Move(Targets[0], x: Oscillate(_width * 0.15, _width * 0.85, period: 4.0), y: null);
                    Move(Targets[1], x: Oscillate(_width * 0.85, _width * 0.15, period: 4.0), y: null);
                    break;

                case ScenarioKind.MultipleEnemiesCrossing:
                    Move(Targets[0], x: Oscillate(_width * 0.15, _width * 0.85, period: 3.0), y: null);
                    Move(Targets[1], x: Oscillate(_width * 0.85, _width * 0.15, period: 3.7), y: null);
                    break;

                case ScenarioKind.HighSpeedCross:
                    // Same crossing stress as the standard scenario but >3x faster.
                    Move(Targets[0], x: Oscillate(_width * 0.12, _width * 0.88, period: 1.2), y: _height * 0.42);
                    Move(Targets[1], x: Oscillate(_width * 0.88, _width * 0.12, period: 1.2), y: _height * 0.58);
                    break;

                case ScenarioKind.SuddenAcceleration:
                    AdvanceSuddenAcceleration(Targets[0]);
                    break;

                case ScenarioKind.ThreeEnemiesConverging:
                    AdvanceThreeEnemyConvergence();
                    break;
            }
        }

        private void AdvanceStopStart(ArenaTarget target)
        {
            // 4-phase continuous cycle — no position jump at cycle boundaries.
            //   0.0–1.5 s: move right at constant speed
            //   1.5–3.0 s: hold at right edge
            //   3.0–4.5 s: move left at constant speed
            //   4.5–6.0 s: hold at left edge
            // Using Lerp (constant velocity) rather than Oscillate so the tracker
            // sees a true velocity→zero transition, not a sinusoidal one.
            double left = _width * 0.2;
            double right = _width * 0.8;
            double cycle = _elapsedSeconds % 6.0;
            double x;
            if (cycle < 1.5)
                x = Lerp(left, right, cycle / 1.5);
            else if (cycle < 3.0)
                x = right;
            else if (cycle < 4.5)
                x = Lerp(right, left, (cycle - 3.0) / 1.5);
            else
                x = left;
            Move(target, x: x, y: _height * 0.5);
        }

        private void AdvanceSuddenAcceleration(ArenaTarget target)
        {
            double cycle = _elapsedSeconds % 4.0;
            double left = _width * 0.12;
            double right = _width * 0.88;

            if (cycle < 1.5)
            {
                // Deliberately slow first phase: only 12% of the span in 1.5 seconds.
                double slowT = cycle / 1.5;
                Move(target, x: Lerp(left, left + (right - left) * 0.12, slowT), y: _height * 0.5);
                return;
            }

            // Abruptly accelerate for 1.25s, then return for the next cycle.
            double fastT = Math.Clamp((cycle - 1.5) / 1.25, 0.0, 1.0);
            if (cycle < 2.75)
            {
                Move(target, x: Lerp(left + (right - left) * 0.12, right, fastT), y: _height * 0.5);
            }
            else
            {
                double returnT = Math.Clamp((cycle - 2.75) / 1.25, 0.0, 1.0);
                Move(target, x: Lerp(right, left, returnT), y: _height * 0.5);
            }
        }

        private void AdvanceThreeEnemyConvergence()
        {
            double t = Triangle01(period: 4.0);
            Point center = new(_width * 0.5, _height * 0.5);
            Point[] starts =
            [
                new(_width * 0.12, _height * 0.15),
                new(_width * 0.88, _height * 0.15),
                new(_width * 0.50, _height * 0.88),
            ];

            for (int i = 0; i < Targets.Count && i < starts.Length; i++)
            {
                Move(
                    Targets[i],
                    x: Lerp(starts[i].X, center.X, t),
                    y: Lerp(starts[i].Y, center.Y, t));
            }
        }

        private void Move(ArenaTarget target, double? x, double? y)
        {
            target.Position = new Point(x ?? target.Position.X, y ?? target.Position.Y);
        }

        private double Oscillate(double min, double max, double period)
        {
            double phase = (_elapsedSeconds % period) / period;
            double t = (Math.Sin(phase * 2 * Math.PI - Math.PI / 2) + 1) / 2;
            return min + (max - min) * t;
        }

        private double Triangle(double min, double max, double period)
            => min + (max - min) * Triangle01(period);

        private double Triangle01(double period)
        {
            double phase = (_elapsedSeconds % period) / period;
            return phase < 0.5 ? phase * 2 : 2 - phase * 2;
        }

        private static double Lerp(double start, double end, double t)
            => start + (end - start) * Math.Clamp(t, 0.0, 1.0);
    }
}
