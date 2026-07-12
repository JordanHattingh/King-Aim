namespace KingAim.Core.Accessibility.Input;

/// <summary>
/// Applies a continuous user-configured drift correction to pointer output.
/// Outputs fractional screen units per frame, identical units to IPointingAssistController.
/// The correction is constant (user-set), not pattern-driven or game-state-aware.
/// </summary>
public interface IDriftCompensator
{
    DriftCompensationProfile Profile { get; }

    /// <summary>Update the active profile. Takes effect on the next Compensate() call.</summary>
    void UpdateProfile(DriftCompensationProfile profile);

    /// <summary>
    /// Returns the correction delta for this frame.
    /// Result is (0, 0) when disabled or when deltaSeconds is not positive.
    /// </summary>
    (float DeltaX, float DeltaY) Compensate(double deltaSeconds);
}
