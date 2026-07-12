namespace KingAim.Core.Diagnostics;

public enum SystemHealthState
{
    Healthy,
    Degraded,
    CpuFallback,
    CaptureUnavailable,
    ModelUnavailable,
    TrackingUnstable,
    AccessibilityOutputDisabled,
    EmergencyDisabled,
}
