namespace KingAim.Core.Safety;

/// <summary>
/// Enforces the King Aim safety contract.
/// Every capability check goes through here so there is a single place to audit.
/// </summary>
public sealed class SafetyPolicy
{
    private static readonly IReadOnlySet<AssistanceCapability> ProhibitedAlways =
        new HashSet<AssistanceCapability>
        {
            AssistanceCapability.AutomaticFiring,
            AssistanceCapability.RecoilControl,
            AssistanceCapability.ProcessInjectionOrMemoryAccess,
            AssistanceCapability.AntiCheatInteraction,
            AssistanceCapability.StealthOperation,
            AssistanceCapability.UnattendedOperation,
            AssistanceCapability.NetworkManipulation,
        };

    private readonly OperatingMode _mode;
    private volatile bool _emergencyDisabled;

    public SafetyPolicy(OperatingMode mode)
    {
        _mode = mode;
    }

    public OperatingMode Mode => _mode;
    public bool IsEmergencyDisabled => _emergencyDisabled;

    /// <summary>
    /// Returns true if the capability may be exercised in the current mode.
    /// Always returns false for prohibited capabilities regardless of mode.
    /// </summary>
    public bool IsPermitted(AssistanceCapability capability)
    {
        if (_emergencyDisabled)
            return false;

        if (ProhibitedAlways.Contains(capability))
            return false;

        return capability switch
        {
            AssistanceCapability.ControlledPointingAssistance =>
                _mode is OperatingMode.LiveCapture or OperatingMode.RecordedVideo,

            AssistanceCapability.DiagnosticRecording =>
                _mode is not OperatingMode.AccessibilityTestHarness,

            _ => true,
        };
    }

    /// <summary>Throws if the capability is not permitted.</summary>
    public void Require(AssistanceCapability capability)
    {
        if (!IsPermitted(capability))
            throw new SafetyViolationException(capability, _mode);
    }

    /// <summary>
    /// Immediately disables all capabilities until the application is restarted.
    /// Safe to call from any thread.
    /// </summary>
    public void EmergencyDisable() => _emergencyDisabled = true;

    /// <summary>Returns all currently permitted capabilities.</summary>
    public IEnumerable<AssistanceCapability> PermittedCapabilities()
        => Enum.GetValues<AssistanceCapability>().Where(IsPermitted);
}
