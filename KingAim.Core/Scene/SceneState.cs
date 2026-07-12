using KingAim.Core.Tracking;

namespace KingAim.Core.Scene;

/// <summary>
/// The current interpreted scene: which targets are active, which is primary,
/// and derived spatial properties.
/// </summary>
public sealed class SceneState
{
    public long   FrameId              { get; init; }
    public IReadOnlyList<TrackState> ActiveTracks { get; init; } = [];

    /// <summary>The track currently selected as the accessibility focus. May be null.</summary>
    public TrackState? PrimaryTrack    { get; init; }

    public int    VisibleEnemyCount    { get; init; }

    /// <summary>Track closest to screen centre. May differ from PrimaryTrack.</summary>
    public TrackState? NearestCentreTrack { get; init; }

    /// <summary>Track with the highest object confidence.</summary>
    public TrackState? HighestConfidenceTrack { get; init; }

    /// <summary>
    /// Scene stability 0–1. Low during rapid target changes or tracker instability.
    /// </summary>
    public float  SceneStability       { get; init; }

    public bool   HasAnyTarget         => PrimaryTrack != null;
    public bool   HasMultipleTargets   => VisibleEnemyCount > 1;
}
