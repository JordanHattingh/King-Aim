namespace KingAim.Core.Perception;

/// <summary>
/// A single detected body landmark in source-frame pixel space.
/// </summary>
public readonly record struct PoseKeypoint(
    KeypointName      Name,
    float             X,
    float             Y,
    float             Confidence,
    KeypointVisibility Visibility);
