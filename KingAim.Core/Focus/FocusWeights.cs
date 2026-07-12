namespace KingAim.Core.Focus;

/// <summary>Weights used for focus-target scoring. All values should sum to 1.0.</summary>
public sealed class FocusWeights
{
    public float Confidence  { get; init; } = 0.30f;
    public float Stability   { get; init; } = 0.25f;
    public float CentreProximity { get; init; } = 0.25f;
    public float PoseQuality { get; init; } = 0.10f;
    public float Persistence { get; init; } = 0.10f;

    /// <summary>
    /// A new candidate must beat the current focus target by this margin
    /// to prevent rapid switching.
    /// </summary>
    public float StickinessThreshold { get; init; } = 0.15f;

    public static FocusWeights Default { get; } = new();
}
