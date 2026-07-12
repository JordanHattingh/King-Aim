using KingAim.Core.Scene;
using KingAim.Core.Tracking;

namespace KingAim.Core.Accessibility.Pointing;

/// <summary>
/// Controls pointing assistance. Requires continuous user hold to operate.
/// Never fires automatically. Never overrides user input direction.
/// </summary>
public interface IPointingAssistController
{
    PointingConstraints Constraints { get; }
    bool IsEngaged   { get; }
    bool IsAvailable { get; }   // false if safety, confidence, or stability conditions not met

    /// <summary>Update focus target. Must be called every frame whether engaged or not.</summary>
    void SetFocusTarget(TrackState? focus, SceneState scene);

    /// <summary>
    /// Compute the correction for this frame.
    /// Returns (0, 0) if not engaged or conditions not met.
    /// X and Y are in fractional screen units.
    /// </summary>
    (float DeltaX, float DeltaY) Update();

    /// <summary>User is holding the enable control. Arm assistance if conditions allow.</summary>
    void Engage();

    /// <summary>User released the enable control. Stop all assistance immediately.</summary>
    void Disengage();

    /// <summary>Immediately stops all assistance. Cannot be re-engaged until reset.</summary>
    void EmergencyStop();
}
