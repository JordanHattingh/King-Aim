namespace KingAim.Core.Perception;

/// <summary>
/// The normalized perception contract. Everything downstream of the decoder uses this type.
/// Allows the tracker, scene analyser, cue engine, and UI to work with any model.
/// All coordinates are in source-frame pixel space.
/// </summary>
public sealed class PoseDetection
{
    public Guid   DetectionId        { get; init; } = Guid.NewGuid();
    public DetectionBoundingBox BoundingBox     { get; init; }
    public float  ObjectConfidence   { get; init; }

    /// <summary>
    /// Decoded keypoints indexed by <see cref="KeypointName"/>.
    /// Always contains exactly 4 entries for pose models.
    /// Empty for detection-only models.
    /// </summary>
    public IReadOnlyList<PoseKeypoint> Keypoints { get; init; } = [];

    public long   SourceFrameId       { get; init; }
    public long   CaptureTimestampUs  { get; init; }
    public long   InferenceTimestampUs { get; init; }
    public string ModelId             { get; init; } = "";

    // Convenience accessors
    public PoseKeypoint? Head       => KeypointOrNull(KeypointName.Head);
    public PoseKeypoint? Neck       => KeypointOrNull(KeypointName.Neck);
    public PoseKeypoint? UpperChest => KeypointOrNull(KeypointName.UpperChest);
    public PoseKeypoint? Hip        => KeypointOrNull(KeypointName.Hip);

    private PoseKeypoint? KeypointOrNull(KeypointName name)
    {
        foreach (var kp in Keypoints)
            if (kp.Name == name) return kp;
        return null;
    }
}
