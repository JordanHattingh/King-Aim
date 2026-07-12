using KingAim.Core.Accessibility.Events;
using KingAim.Core.Scene;
using KingAim.Core.Tracking;

namespace KingAim.Core.Accessibility.Cues;

/// <summary>
/// Drives controller haptics mapped to target position and confidence.
/// All vibration is bounded and cannot run unattended.
/// </summary>
public interface IHapticCueProvider : IAccessibilityEventSink
{
    void Update(SceneState scene, TrackState? focus);

    bool  IsEnabled              { get; set; }
    float MaxStrength            { get; set; }   // 0–1
    int   MaxPulseDurationMs     { get; set; }
    int   MinPulseIntervalMs     { get; set; }
    int   ContinuousTimeoutMs    { get; set; }
    bool  InvertLeftRight        { get; set; }
}
