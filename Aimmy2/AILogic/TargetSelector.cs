using System.Drawing;

namespace Aimmy2.AILogic
{
    public sealed class TargetSelectionResult
    {
        public Track? SelectedTrack { get; init; }
        public float ErrorX { get; init; }
        public float ErrorY { get; init; }
        public float TargetVelocityX { get; init; }
        public float TargetVelocityY { get; init; }
    }

    public sealed class TargetSelector
    {
        private const int DefaultConfirmationFrames = 1;

        private int? _currentTrackId;
        private int? _challengerTrackId;
        private int _challengerStreak;

        public int ConfirmationFrames { get; set; } = DefaultConfirmationFrames;

        /// <summary>Max frames of lead applied when target moves at ReferenceVelocity. Scales to zero when stationary.</summary>
        public float LeadFrames { get; set; } = 1.5f;
        /// <summary>Pixels/frame at which LeadFrames is applied at full value. Above this speed, lead is capped at 2×LeadFrames.</summary>
        public float ReferenceVelocity { get; set; } = 8f;

        public TargetSelectionResult Select(
            IReadOnlyList<Track> tracks,
            TargetMode mode,
            SemanticRole roleFilter,
            int? fixedTrackId,
            PointF screenCenter,
            float normalizationRadius,
            float aimPointFraction = 0.25f)
        {
            ArgumentNullException.ThrowIfNull(tracks);

            var eligible = FilterEligible(tracks, mode, roleFilter, fixedTrackId);

            if (eligible.Count == 0)
            {
                _currentTrackId = null;
                _challengerTrackId = null;
                _challengerStreak = 0;
                return EmptyResult();
            }

            var scored = eligible
                .Select(t => (Track: t, DistSq: DistanceSq(TrackCenter(t.BoundingBox), screenCenter)))
                .OrderBy(x => x.DistSq)
                .ToList();

            var best = scored[0].Track;
            Track? current = _currentTrackId.HasValue
                ? eligible.FirstOrDefault(t => t.TrackId == _currentTrackId.Value)
                : null;

            Track selected;

            if (current == null)
            {
                selected = best;
                _currentTrackId = selected.TrackId;
                _challengerTrackId = null;
                _challengerStreak = 0;
            }
            else if (best.TrackId == current.TrackId)
            {
                selected = current;
                _challengerTrackId = null;
                _challengerStreak = 0;
            }
            else
            {
                float currentDistSq = DistanceSq(TrackCenter(current.BoundingBox), screenCenter);
                float bestDistSq = scored[0].DistSq;

                bool meaningfullyBetter = bestDistSq < currentDistSq * 0.7f;

                if (meaningfullyBetter && best.TrackId == _challengerTrackId)
                {
                    _challengerStreak++;
                }
                else if (meaningfullyBetter)
                {
                    _challengerTrackId = best.TrackId;
                    _challengerStreak = 1;
                }
                else
                {
                    _challengerTrackId = null;
                    _challengerStreak = 0;
                }

                if (_challengerStreak >= ConfirmationFrames)
                {
                    selected = best;
                    _currentTrackId = selected.TrackId;
                    _challengerTrackId = null;
                    _challengerStreak = 0;
                }
                else
                {
                    selected = current;
                }
            }

            // Aim point: vertical fraction of the bounding box (0=top, 0.5=center, 1=bottom).
            // Full-body models use 0.25 (head/chest). Head-only models use 0.5 (center of head box).
            float aimX = selected.BoundingBox.X + selected.BoundingBox.Width * 0.5f;
            float aimY = selected.BoundingBox.Y + selected.BoundingBox.Height * aimPointFraction;

            // Dynamic lead: scale lead frames by how fast the target is actually moving.
            // Stationary targets get zero lead (no phantom offset); fast strafers get up to 2× lead.
            float velocityMag = MathF.Sqrt(selected.Velocity.X * selected.Velocity.X + selected.Velocity.Y * selected.Velocity.Y);
            float velocityScale = ReferenceVelocity > 0
                ? Math.Clamp(velocityMag / ReferenceVelocity, 0f, 2f)
                : 0f;
            float effectiveLead = LeadFrames * velocityScale;

            float maxDim = Math.Max(selected.BoundingBox.Width, selected.BoundingBox.Height);
            float maxLead = maxDim * 2f;
            aimX += Math.Clamp(selected.Velocity.X * effectiveLead, -maxLead, maxLead);
            aimY += Math.Clamp(selected.Velocity.Y * effectiveLead, -maxLead, maxLead);

            float errorX = normalizationRadius > 0
                ? Math.Clamp((aimX - screenCenter.X) / normalizationRadius, -1f, 1f)
                : 0f;
            float errorY = normalizationRadius > 0
                ? Math.Clamp((aimY - screenCenter.Y) / normalizationRadius, -1f, 1f)
                : 0f;

            // Normalize velocity into the same -1..+1 space as errorX/Y so feed-forward works correctly.
            float velX = normalizationRadius > 0 ? selected.Velocity.X / normalizationRadius : 0f;
            float velY = normalizationRadius > 0 ? selected.Velocity.Y / normalizationRadius : 0f;

            return new TargetSelectionResult
            {
                SelectedTrack = selected,
                ErrorX = errorX,
                ErrorY = errorY,
                TargetVelocityX = velX,
                TargetVelocityY = velY,
            };
        }

        private static List<Track> FilterEligible(
            IReadOnlyList<Track> tracks,
            TargetMode mode,
            SemanticRole roleFilter,
            int? fixedTrackId)
        {
            IEnumerable<Track> query = tracks.Where(t => t.Role != SemanticRole.Ignore && t.Role != SemanticRole.Friendly);

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
                        ? tracks.Where(t => t.TrackId == fixedTrackId.Value && t.Role != SemanticRole.Ignore)
                        : Enumerable.Empty<Track>();
                    break;

                case TargetMode.TestTarget:
                    // TestTarget mode accepts any non-ignored role for use against synthetic test-arena targets.
                    break;
            }

            return query.ToList();
        }

        private static TargetSelectionResult EmptyResult() => new()
        {
            SelectedTrack = null,
            ErrorX = 0f,
            ErrorY = 0f,
            TargetVelocityX = 0f,
            TargetVelocityY = 0f,
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
