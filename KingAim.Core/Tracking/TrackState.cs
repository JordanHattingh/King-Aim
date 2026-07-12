using KingAim.Core.Perception;

namespace KingAim.Core.Tracking;

/// <summary>
/// The current state of one tracked target. Coordinates are in source-frame pixel space.
/// </summary>
public sealed class TrackState
{
    public int   TrackId          { get; init; }
    public DetectionBoundingBox Box { get; init; }

    /// <summary>Smoothed keypoints. May have fewer entries than the model provides if some are rejected.</summary>
    public IReadOnlyList<PoseKeypoint> Keypoints { get; init; } = [];

    public float  DetectionConfidence { get; init; }
    public int    Age                 { get; init; }   // frames since first detection
    public int    VisibleFrames       { get; init; }
    public int    MissingFrames       { get; init; }

    /// <summary>Pixels/frame in each axis.</summary>
    public float  VelocityX           { get; init; }
    public float  VelocityY           { get; init; }

    public float  StabilityScore      { get; init; }   // 0–1

    public bool   IsNew      => Age <= 2;
    public bool   IsStable   => StabilityScore > 0.7f && VisibleFrames >= 5;
    public bool   IsLost     => MissingFrames > 0;
}
