using KingAim.Core.Accessibility.Pointing;
using KingAim.Core.Focus;

namespace KingAim.Core.Profiles;

/// <summary>
/// Contains all user-facing behaviour settings.
/// Profiles are exported/imported without containing private recordings.
/// </summary>
public sealed class AccessibilityProfile
{
    public string  Id          { get; init; } = Guid.NewGuid().ToString("N");
    public string  Name        { get; init; } = "Default";
    public string? ModelId     { get; init; }

    // Thresholds
    public float BodyActivationThreshold   { get; init; } = 0.70f;
    public float BodyRemovalThreshold      { get; init; } = 0.50f;
    public float PoseActivationThreshold   { get; init; } = 0.50f;
    public float KeypointMinConfidence     { get; init; } = 0.30f;
    public int   MinCompleteKeypoints      { get; init; } = 2;
    public int   MinStableFrames           { get; init; } = 3;
    public int   MaxMissingFrames          { get; init; } = 8;

    // Output settings
    public OverlaySettings  Overlay  { get; init; } = new();
    public AudioSettings    Audio    { get; init; } = new();
    public HapticSettings   Haptic   { get; init; } = new();

    // Pointing
    public bool IsPointingAssistanceEnabled { get; init; } = false;
    public PointingConstraints PointingConstraints { get; init; } = PointingConstraints.Default;

    // Focus selection
    public FocusWeights FocusWeights { get; init; } = FocusWeights.Default;

    public static AccessibilityProfile VisualOnly  { get; } = new() { Name = "Visual Only",   Audio  = new() { IsEnabled = false }, Haptic = new() { IsEnabled = false } };
    public static AccessibilityProfile AudioOnly   { get; } = new() { Name = "Audio Only",    Overlay = new() { IsEnabled = false }, Haptic = new() { IsEnabled = false } };
    public static AccessibilityProfile HapticOnly  { get; } = new() { Name = "Haptic Only",   Overlay = new() { IsEnabled = false }, Audio  = new() { IsEnabled = false } };
    public static AccessibilityProfile CombinedCues{ get; } = new() { Name = "Combined Cues" };
}

public sealed class OverlaySettings
{
    public bool  IsEnabled        { get; init; } = true;
    public bool  HighContrast     { get; init; } = false;
    public bool  ColourBlindSafe  { get; init; } = false;
    public bool  ReducedMotion    { get; init; } = false;
    public float Opacity          { get; init; } = 0.85f;
    public float LineThickness    { get; init; } = 2f;
}

public sealed class AudioSettings
{
    public bool  IsEnabled        { get; init; } = true;
    public float MaxVolume        { get; init; } = 0.70f;
    public float MinCueIntervalMs { get; init; } = 150f;
    public bool  MonoCompatibility{ get; init; } = false;
    public bool  ReducedLoad      { get; init; } = false;
}

public sealed class HapticSettings
{
    public bool  IsEnabled           { get; init; } = true;
    public float MaxStrength         { get; init; } = 0.60f;
    public int   MaxPulseDurationMs  { get; init; } = 200;
    public int   MinPulseIntervalMs  { get; init; } = 100;
    public int   ContinuousTimeoutMs { get; init; } = 2000;
    public bool  InvertLeftRight     { get; init; } = false;
}
