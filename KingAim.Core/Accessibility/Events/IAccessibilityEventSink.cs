namespace KingAim.Core.Accessibility.Events;

/// <summary>
/// Receives accessibility events. Visual, audio, and haptic providers implement this.
/// </summary>
public interface IAccessibilityEventSink
{
    void OnEvent(AccessibilityEvent evt);
}
