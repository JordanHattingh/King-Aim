namespace KingAim.Core.Accessibility.Pointing;

/// <summary>
/// Hard limits on controlled pointing assistance.
/// These values exist to prevent misuse even if the user disables all other checks.
/// </summary>
public sealed class PointingConstraints
{
    /// <summary>Maximum movement per inference frame in fractional screen units.</summary>
    public float MaxMovementPerFrameFraction { get; init; } = 0.005f;

    /// <summary>Maximum velocity as a multiple of the per-frame limit.</summary>
    public float MaxAccelerationMultiplier   { get; init; } = 2.0f;

    /// <summary>Total correction per second in fractional screen units.</summary>
    public float MaxCorrectionPerSecond      { get; init; } = 0.05f;

    /// <summary>Dead zone radius in fractional screen units. No correction inside this zone.</summary>
    public float DeadZoneFraction            { get; init; } = 0.015f;

    /// <summary>Minimum model confidence before assistance is allowed.</summary>
    public float MinConfidenceToAssist       { get; init; } = 0.65f;

    /// <summary>Minimum tracking stability score before assistance is allowed.</summary>
    public float MinStabilityToAssist        { get; init; } = 0.50f;

    /// <summary>If confidence drops below this for this many frames, disengage.</summary>
    public int   MaxLowConfidenceFrames      { get; init; } = 3;

    public static PointingConstraints Default { get; } = new();
    public static PointingConstraints Conservative { get; } = new()
    {
        MaxMovementPerFrameFraction = 0.002f,
        MaxCorrectionPerSecond      = 0.02f,
        DeadZoneFraction            = 0.025f,
        MinConfidenceToAssist       = 0.75f,
        MinStabilityToAssist        = 0.70f,
    };
}
