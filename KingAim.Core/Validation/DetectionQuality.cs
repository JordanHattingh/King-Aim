namespace KingAim.Core.Validation;

/// <summary>
/// Quality level assigned to a raw detection after geometry validation.
/// Determines which accessibility outputs are safe to derive from the detection.
/// </summary>
public enum DetectionQuality
{
    /// <summary>All checks passed. Full accessibility outputs permitted.</summary>
    High,

    /// <summary>Minor issues (slightly out of bounds, one uncertain keypoint). Cues permitted.</summary>
    Usable,

    /// <summary>Body detected but keypoint confidence is too low. Visual body cue only.</summary>
    CueOnly,

    /// <summary>Detection is borderline. Report but do not drive any output.</summary>
    Uncertain,

    /// <summary>Failed hard checks. Discarded — not forwarded to tracker.</summary>
    Rejected,
}
