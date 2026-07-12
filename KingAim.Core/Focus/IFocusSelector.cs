using KingAim.Core.Scene;
using KingAim.Core.Tracking;

namespace KingAim.Core.Focus;

/// <summary>
/// Selects a single focus target from the active tracks.
/// Uses deterministic weighted scoring with hysteresis to prevent rapid switching.
/// </summary>
public interface IFocusSelector
{
    /// <summary>
    /// Returns the selected focus track, or null if no suitable track exists.
    /// The currently focused track is given a stickiness bonus.
    /// </summary>
    TrackState? SelectFocus(SceneState scene, int sourceWidth, int sourceHeight);

    /// <summary>The track selected in the previous call. Null on first call.</summary>
    TrackState? CurrentFocus { get; }
}
