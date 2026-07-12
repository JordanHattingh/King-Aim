namespace KingAim.Core.Perception;

/// <summary>
/// Per-keypoint visibility flag, matching the training annotation convention.
/// </summary>
public enum KeypointVisibility
{
    /// <summary>Not detected or extrapolated. Coordinates are unreliable.</summary>
    Absent    = 0,

    /// <summary>Inferred/occluded. Coordinates estimated but may be imprecise.</summary>
    Occluded  = 1,

    /// <summary>Clearly visible in the frame. High-confidence coordinates.</summary>
    Visible   = 2,
}
