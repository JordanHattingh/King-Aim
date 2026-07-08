using System.Windows;

namespace Aimmy2.TestArena
{
    public enum TargetKind
    {
        Player,
        Enemy,
        Friendly,
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
        StaticEnemy,
        MovingEnemyHorizontal,
        MovingEnemyVertical,
        MovingEnemyDiagonal,
        DirectionReversal,
        StopStart,
        TemporaryOcclusion,
        TwoEnemies,
        EnemyAndFriendlyCrossing,
        MultipleEnemiesCrossing,
        NoTarget,
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
            ScenarioKind.NoTarget => new List<ArenaTarget>(),

            ScenarioKind.TwoEnemies or ScenarioKind.MultipleEnemiesCrossing => new List<ArenaTarget>
            {
                new() { Id = "enemy1", Kind = TargetKind.Enemy, Position = new Point(w * 0.25, h * 0.5) },
                new() { Id = "enemy2", Kind = TargetKind.Enemy, Position = new Point(w * 0.75, h * 0.5) },
            },

            ScenarioKind.EnemyAndFriendlyCrossing => new List<ArenaTarget>
            {
                new() { Id = "enemy1", Kind = TargetKind.Enemy, Position = new Point(w * 0.2, h * 0.5) },
                new() { Id = "friendly1", Kind = TargetKind.Friendly, Position = new Point(w * 0.8, h * 0.5) },
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
                    // Moves for 1.5s, then holds still for 1.5s, repeating.
                    double cyclePos = _elapsedSeconds % 3.0;
                    if (cyclePos < 1.5)
                    {
                        Move(Targets[0], x: Oscillate(_width * 0.2, _width * 0.8, period: 3.0), y: null);
                    }
                    break;

                case ScenarioKind.TemporaryOcclusion:
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
        {
            double phase = (_elapsedSeconds % period) / period;
            double t = phase < 0.5 ? phase * 2 : 2 - phase * 2;
            return min + (max - min) * t;
        }
    }
}
