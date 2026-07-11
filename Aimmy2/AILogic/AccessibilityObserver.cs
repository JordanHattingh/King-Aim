using System.Drawing;

namespace Aimmy2.AILogic
{
    public sealed record AccessibilityObservation(
        int TrackId,
        SemanticRole Role,
        DesktopPixel Center,
        ScreenFraction CenterFraction,
        float Confidence,
        PointF VelocityNormPerSecond,
        ScreenFraction? GruPredictedCenterFraction,
        PlayerKeypoints? Keypoints,
        bool IsOccluded,
        bool BoundingBoxIsExtrapolated,
        TimeSpan ObservationAge,
        TimeSpan TrackAge);

    /// <summary>
    /// Read-only semantic projection of TrackManager state for accessibility outputs.
    /// The observer never mutates tracks and publishes immutable snapshots.
    /// </summary>
    public sealed class AccessibilityObserver
    {
        private readonly TrackManager _trackManager;
        private AccessibilityObservation[] _latest = Array.Empty<AccessibilityObservation>();

        public AccessibilityObserver(TrackManager trackManager)
        {
            _trackManager = trackManager ?? throw new ArgumentNullException(nameof(trackManager));
        }

        public IReadOnlyList<AccessibilityObservation> Observe()
            => Observe(DateTime.UtcNow);

        public IReadOnlyList<AccessibilityObservation> Observe(DateTime observedAt)
        {
            AccessibilityObservation[] snapshot = _trackManager.ActiveTracks
                .Where(track => track.Role != SemanticRole.Ignore)
                .OrderBy(track => RolePriority(track.Role))
                .ThenBy(track => track.TrackId)
                .Select(track => CreateObservation(track, observedAt))
                .ToArray();

            Volatile.Write(ref _latest, snapshot);
            return snapshot;
        }

        public AccessibilityObservation? PrimaryTarget()
        {
            Track? track = _trackManager.ActiveTracks
                .Where(candidate => candidate.Role != SemanticRole.Ignore)
                .OrderBy(candidate => candidate.FramesSinceLastSeen)
                .ThenBy(candidate => RolePriority(candidate.Role))
                .ThenByDescending(candidate => candidate.Confidence)
                .ThenBy(candidate => candidate.TrackId)
                .FirstOrDefault();

            return track == null
                ? null
                : CreateObservation(track, DateTime.UtcNow);
        }

        public IReadOnlyList<AccessibilityObservation> Latest
            => Volatile.Read(ref _latest);

        private AccessibilityObservation CreateObservation(Track track, DateTime observedAt)
        {
            ScreenFraction? gruCenter = track.GruPredictedCenter is { } point
                ? new DesktopPixel(point.X, point.Y).ToScreenFraction(
                    _trackManager.ScreenLeft,
                    _trackManager.ScreenTop,
                    _trackManager.ScreenWidth,
                    _trackManager.ScreenHeight)
                : null;

            return new AccessibilityObservation(
                track.TrackId,
                track.Role,
                track.CenterDesktop,
                track.CenterFraction,
                track.Confidence,
                track.Velocity,
                gruCenter,
                track.Keypoints,
                track.FramesSinceLastSeen > 0,
                track.BoundingBoxIsExtrapolated,
                TimeSpan.FromSeconds(Math.Max(0, (observedAt - track.LastSeen).TotalSeconds)),
                TimeSpan.FromSeconds(Math.Max(0, (observedAt - track.FirstSeen).TotalSeconds)));
        }

        private static int RolePriority(SemanticRole role) => role switch
        {
            SemanticRole.Enemy => 0,
            SemanticRole.Player => 1,
            SemanticRole.Friendly => 2,
            SemanticRole.Npc => 3,
            SemanticRole.Objective => 4,
            SemanticRole.Interactable => 5,
            SemanticRole.Unknown => 6,
            SemanticRole.Ignore => 7,
            _ => 8,
        };
    }
}
