namespace KingAim.Core.Decoding;

/// <summary>
/// Static configuration for a pose decoder. Injected at construction time;
/// not passed per-frame. Replace NmsParameters at the decoder boundary.
/// </summary>
public sealed record PoseDecoderOptions
{
    public static readonly PoseDecoderOptions Default = new();

    /// <summary>Minimum object confidence to keep a detection before NMS.</summary>
    public float DetectionConfidenceThreshold { get; init; } = 0.25f;

    /// <summary>Minimum per-keypoint visibility score to mark a keypoint Visible.</summary>
    public float KeypointConfidenceThreshold  { get; init; } = 0.50f;

    /// <summary>IoU threshold for non-maximum suppression.</summary>
    public float NmsIouThreshold             { get; init; } = 0.45f;

    /// <summary>Maximum detections to return per frame after NMS.</summary>
    public int   MaximumDetections           { get; init; } = 300;
}
