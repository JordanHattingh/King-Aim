using KingAim.Core.Accessibility.Events;
using KingAim.Core.Scene;
using KingAim.Core.Tracking;

namespace KingAim.Core.Accessibility.Cues;

/// <summary>
/// Draws accessibility overlays. Operates through a transparent OS-level overlay.
/// Never injects into game processes.
/// </summary>
public interface IVisualCueProvider : IAccessibilityEventSink
{
    /// <summary>
    /// Updates the overlay for the current frame.
    /// Called once per inference cycle from the pipeline thread.
    /// Must not block.
    /// </summary>
    void Update(SceneState scene, TrackState? focus);

    bool IsEnabled { get; set; }
    bool HighContrastMode { get; set; }
    bool ColourBlindSafeMode { get; set; }
    bool ReducedMotionMode { get; set; }
    float Opacity { get; set; }
    float LineThickness { get; set; }
}
