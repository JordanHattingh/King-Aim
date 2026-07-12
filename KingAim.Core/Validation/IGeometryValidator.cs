using KingAim.Core.Perception;

namespace KingAim.Core.Validation;

public sealed class ValidationResult
{
    public DetectionQuality Quality { get; init; }
    public IReadOnlyList<string> Issues { get; init; } = [];
}

/// <summary>
/// Applies pose-geometry sanity checks to raw decoder output
/// before forwarding to the tracker.
/// </summary>
public interface IGeometryValidator
{
    ValidationResult Validate(PoseDetection detection, int sourceWidth, int sourceHeight);
}
