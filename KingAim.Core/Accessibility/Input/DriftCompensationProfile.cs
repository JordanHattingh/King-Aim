namespace KingAim.Core.Accessibility.Input;

/// <summary>
/// User-adjustable constant-offset drift correction applied continuously to pointer output.
/// Units are fractional screen width/height per second (same as IPointingAssistController).
/// Positive X moves right; positive Y moves down.
/// </summary>
public sealed record DriftCompensationProfile
{
    public static readonly DriftCompensationProfile Disabled = new();

    /// <summary>Horizontal correction in fractional screen widths per second. Negative = left.</summary>
    public float HorizontalOffsetPerSecond { get; init; } = 0f;

    /// <summary>Vertical correction in fractional screen heights per second. Negative = up.</summary>
    public float VerticalOffsetPerSecond   { get; init; } = 0f;

    public bool Enabled { get; init; } = false;
}
