namespace KingAim.Core.Safety;

public sealed class SafetyViolationException : InvalidOperationException
{
    public AssistanceCapability Capability { get; }
    public OperatingMode Mode { get; }

    public SafetyViolationException(AssistanceCapability capability, OperatingMode mode)
        : base($"Capability '{capability}' is not permitted in mode '{mode}'.")
    {
        Capability = capability;
        Mode = mode;
    }
}
