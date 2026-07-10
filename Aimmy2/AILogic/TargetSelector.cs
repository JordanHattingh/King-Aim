using System.Drawing;

namespace Aimmy2.AILogic
{
    public sealed class TargetSelectionResult
    {
        public Track? SelectedTrack { get; init; }
        public float ErrorX { get; init; }
        public float ErrorY { get; init; }

        /// <summary>Normalized display units per second.</summary>
        public float TargetVelocityX { get; init; }
        public float TargetVelocityY { get; init; }
        public double ObservationAgeMs { get; init; }
        public PointF ObservationPoint { get; init; }
        public ObservationPointSource ObservationPointSource { get; init; } = ObservationPointSource.BoxFallback;
    }

    public sealed class TargetSelector
    {
        private int? _currentTrackId;
        private int? _challengerTrackId;
        private DateTime? _challengerSince;

        /// <summary>How long a better challenger must remain best before selection switches.</summary>
        public double SwitchConfirmationMs { get; set; } = 100.0;

        /// <summary>Maximum age of a real observation before the track becomes ineligible.</summary>
        public double MaximumObservationAgeMs { get; set; } = 250.0;

        /// <summary>Required fractional distance improvement. 0.30 means challenger must be at least 30% closer.</summary>
        public float SwitchAdvantageThreshold { get; set; } = 0.30f;

        public TargetSelectionResult Select(
            IReadOnlyList<Track> tracks,
            TargetMode mode,
            SemanticRole roleFilter,
            int? fixedTrackId,
            PointF screenCenter,
            float normalizationRadius,
            DateTime now,
            float aimPointFraction = 0.25f)
        {
            ArgumentNullException.ThrowIfNull(tracks);

            var eligible = FilterEligible(
                tracks,
                mode,
                roleFilter,
                fixedTrackId,
                now,
                MaximumObservationAgeMs);

            if (eligible.Count == 0)
            {
                ResetSelection();
                return EmptyResult();
            }

            var scored = eligible
                .Select(t => (Track: t, DistSq: DistanceSq(TrackCenter(t.BoundingBox), screenCenter)))
                .OrderBy(x => x.DistSq)
                .ToList();

            Track best = scored[0].Track;
            Track? current = _currentTrackId.HasValue
                ? eligible.FirstOrDefault(t => t.TrackId == _currentTrackId.Value)
                : null;

            Track selected;
            if (current == null)
            {
                selected = best;
                _currentTrackId = selected.TrackId;
                ResetChallenger();
            }
            else if (best.TrackId == current.TrackId)
            {
                selected = current;
                ResetChallenger();
            }
            else
            {
                float currentDistSq = DistanceSq(TrackCenter(current.BoundingBox), screenCenter);
                float bestDistSq = scored[0].DistSq;
                float requiredRatio = Math.Clamp(1f - SwitchAdvantageThreshold, 0f, 1f);
                bool meaningfullyBetter = bestDistSq < currentDistSq * requiredRatio;

                if (meaningfullyBetter)
                {
                    if (_challengerTrackId != best.TrackId)
                    {
                        _challengerTrackId = best.TrackId;
                        _challengerSince = now;
                    }

                    double elapsedMs = _challengerSince.HasValue
                        ? Math.Max(0, (now - _challengerSince.Value).TotalMilliseconds)
                        : 0;

                    if (elapsedMs >= Math.Max(0, SwitchConfirmationMs))
                    {
                        selected = best;
                        _currentTrackId = selected.TrackId;
                        ResetChallenger();
                    }
                    else
                    {
                        selected = current;
                    }
                }
                else
                {
                    selected = current;
                    ResetChallenger();
                }
            }

            ResolvedObservationPoint resolved = ObservationPointResolver.Resolve(selected, aimPointFraction);
            PointF point = resolved.Point;

            // Selection publishes the latest semantic observation point. Temporal projection is a
            // separate predictor concern; applying it here previously mixed normalized per-second
            // velocity with an approximate screen span and duplicated GRU/Kalman prediction.

            float errorX = normalizationRadius > 0
                ? Math.Clamp((point.X - screenCenter.X) / normalizationRadius, -1f, 1f)
                : 0f;
            float errorY = normalizationRadius > 0
                ? Math.Clamp((point.Y - screenCenter.Y) / normalizationRadius, -1f, 1f)
                : 0f;

            return new TargetSelectionResult
            {
                SelectedTrack = selected,
                ErrorX = errorX,
                ErrorY = errorY,
                TargetVelocityX = selected.Velocity.X,
                TargetVelocityY = selected.Velocity.Y,
                ObservationAgeMs = Math.Max(0, (now - selected.LastSeen).TotalMilliseconds),
                ObservationPoint = point,
                ObservationPointSource = resolved.Source,
            };
        }

        private static List<Track> FilterEligible(
            IReadOnlyList<Track> tracks,
            TargetMode mode,
            SemanticRole roleFilter,
            int? fixedTrackId,
            DateTime now,
            double maximumObservationAgeMs)
        {
            IEnumerable<Track> query = tracks.Where(t =>
                t.Role != SemanticRole.Ignore &&
                Math.Max(0, (now - t.LastSeen).TotalMilliseconds) <= maximumObservationAgeMs);

            switch (mode)
            {
                case TargetMode.EnemyOnly:
                    query = query.Where(t => t.Role == SemanticRole.Enemy);
                    break;
                case TargetMode.PlayerClass:
                    query = query.Where(t => t.Role == SemanticRole.Player);
                    break;
                case TargetMode.SpecificClass:
                    query = query.Where(t => t.Role == roleFilter);
                    break;
                case TargetMode.FixedTrackId:
                    query = fixedTrackId.HasValue
                        ? query.Where(t => t.TrackId == fixedTrackId.Value)
                        : Enumerable.Empty<Track>();
                    break;
                case TargetMode.TestTarget:
                    break;
            }

            return query.ToList();
        }

        private void ResetSelection()
        {
            _currentTrackId = null;
            ResetChallenger();
        }

        private void ResetChallenger()
        {
            _challengerTrackId = null;
            _challengerSince = null;
        }

        private static TargetSelectionResult EmptyResult() => new()
        {
            SelectedTrack = null,
            ErrorX = 0f,
            ErrorY = 0f,
            TargetVelocityX = 0f,
            TargetVelocityY = 0f,
            ObservationAgeMs = double.PositiveInfinity,
            ObservationPoint = PointF.Empty,
        };

        private static PointF TrackCenter(RectangleF box) =>
            new(box.X + box.Width / 2f, box.Y + box.Height / 2f);

        private static float DistanceSq(PointF a, PointF b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return dx * dx + dy * dy;
        }
    }
}
