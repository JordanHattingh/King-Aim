namespace KingAim.Core.Accessibility.Input;

/// <summary>
/// Drift compensator driven entirely by user-supplied offset values.
/// Thread-safe profile updates; compensation runs on the pipeline thread.
/// </summary>
public sealed class ManualDriftCompensator : IDriftCompensator
{
    private volatile DriftCompensationProfile _profile = DriftCompensationProfile.Disabled;

    public DriftCompensationProfile Profile => _profile;

    public void UpdateProfile(DriftCompensationProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        _profile = profile;
    }

    public (float DeltaX, float DeltaY) Compensate(double deltaSeconds)
    {
        var p = _profile;
        if (!p.Enabled || deltaSeconds <= 0.0) return (0f, 0f);

        return (
            p.HorizontalOffsetPerSecond * (float)deltaSeconds,
            p.VerticalOffsetPerSecond   * (float)deltaSeconds
        );
    }
}
