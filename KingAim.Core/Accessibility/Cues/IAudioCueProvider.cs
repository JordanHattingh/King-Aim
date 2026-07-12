using KingAim.Core.Accessibility.Events;
using KingAim.Core.Scene;
using KingAim.Core.Tracking;

namespace KingAim.Core.Accessibility.Cues;

/// <summary>
/// Emits spatial audio cues mapped to target position.
/// Rate-limited to prevent sensory overload.
/// </summary>
public interface IAudioCueProvider : IAccessibilityEventSink
{
    void Update(SceneState scene, TrackState? focus);

    bool  IsEnabled          { get; set; }
    float MaxVolumeNormalized { get; set; }   // 0–1
    float MinCueIntervalMs   { get; set; }
    bool  MonoCompatibility  { get; set; }
    bool  ReducedSensoryLoad { get; set; }
}
