namespace KingAim.Core.Accessibility.Events;

public enum AccessibilityEventKind
{
    FocusTargetAppeared,
    FocusTargetLost,
    FocusTargetMovedLeft,
    FocusTargetMovedRight,
    FocusTargetMovedUp,
    FocusTargetMovedDown,
    FocusTargetNearCentre,
    MultipleTargetsVisible,
    PoseConfidenceReduced,
    TrackingInterrupted,
    TrackingRestored,
    EmergencyDisabled,
}

public sealed class AccessibilityEvent
{
    public AccessibilityEventKind Kind      { get; init; }
    public long   FrameId                  { get; init; }
    public long   TimestampUs              { get; init; }
    public int?   TrackId                  { get; init; }
    public float? Intensity                { get; init; }   // 0–1, used by audio/haptic
    public string? Detail                  { get; init; }
}
