using KingAim.Core.Accessibility;
using KingAim.Core.Accessibility.Cues;
using KingAim.Core.Accessibility.Events;
using KingAim.Core.Accessibility.Input;
using KingAim.Core.Accessibility.Pointing;
using KingAim.Core.Capture.Sources;
using KingAim.Core.Decoding;
using KingAim.Core.Diagnostics;
using KingAim.Core.Focus;
using KingAim.Core.Inference;
using KingAim.Core.Perception;
using KingAim.Core.Pipeline;
using KingAim.Core.Preprocessing;
using KingAim.Core.Safety;
using KingAim.Core.Scene;
using KingAim.Core.Scheduling;
using KingAim.Core.Tracking;
using KingAim.Core.Validation;
using Xunit;

namespace Aimmy2.Tests;

public sealed class PipelineGamepadAssistTests
{
    // ── Stubs ──────────────────────────────────────────────────────────────────

    private sealed class NullVisualCue : IVisualCueProvider
    {
        public bool  IsEnabled          { get; set; } = true;
        public bool  HighContrastMode   { get; set; }
        public bool  ColourBlindSafeMode { get; set; }
        public bool  ReducedMotionMode  { get; set; }
        public float Opacity            { get; set; } = 1f;
        public float LineThickness      { get; set; } = 2f;
        public void Update(SceneState s, TrackState? f) { }
        public void OnEvent(AccessibilityEvent e) { }
    }

    private sealed class NullAudioCue : IAudioCueProvider
    {
        public bool  IsEnabled           { get; set; } = true;
        public float MaxVolumeNormalized { get; set; } = 1f;
        public float MinCueIntervalMs    { get; set; } = 100f;
        public bool  MonoCompatibility   { get; set; }
        public bool  ReducedSensoryLoad  { get; set; }
        public void Update(SceneState s, TrackState? f) { }
        public void OnEvent(AccessibilityEvent e) { }
    }

    private sealed class NullHapticCue : IHapticCueProvider
    {
        public bool  IsEnabled           { get; set; } = true;
        public float MaxStrength         { get; set; } = 1f;
        public int   MaxPulseDurationMs  { get; set; } = 500;
        public int   MinPulseIntervalMs  { get; set; } = 200;
        public int   ContinuousTimeoutMs { get; set; } = 5000;
        public bool  InvertLeftRight     { get; set; }
        public void Update(SceneState s, TrackState? f) { }
        public void OnEvent(AccessibilityEvent e) { }
    }

    private sealed class NullPointing : IPointingAssistController
    {
        public PointingConstraints Constraints { get; } = new();
        public bool IsEngaged   { get; private set; }
        public bool IsAvailable => false;
        public void SetFocusTarget(TrackState? f, SceneState s) { }
        public (float DeltaX, float DeltaY) Update() => (0f, 0f);
        public void Engage()        { IsEngaged = true; }
        public void Disengage()     { IsEngaged = false; }
        public void EmergencyStop() { IsEngaged = false; }
    }

    // ── Builder ────────────────────────────────────────────────────────────────

    private static PipelineRunner MakeRunner(
        IGamepadAssistController? gamepadAssist = null,
        IDriftCompensator?        drift         = null)
    {
        var safety = new SafetyPolicy(OperatingMode.AccessibilityTestHarness);
        var source = new SyntheticPatternSource(1920, 1080);
        return new PipelineRunner(
            safety,
            source,
            new FrameScheduler(),
            new LetterboxPreprocessor(512),
            new MockInferenceEngine(),
            new MockPoseDecoder(),
            new GeometryValidator(),
            new SimpleTrackerService(),
            new SceneAnalyzer(),
            new FocusSelector(),
            new AccessibilityEventDispatcher(),
            new NullVisualCue(),
            new NullAudioCue(),
            new NullHapticCue(),
            new NullPointing(),
            drift ?? new ManualDriftCompensator(),
            gamepadAssist ?? new GamepadAssistController(),
            new DiagnosticsService());
    }

    // ── Helper: build a stable TrackState at an arbitrary screen position ──────

    private static TrackState FocusAt(float cx, float cy, float r = 30f, float conf = 0.9f, int trackId = 1)
        => new()
        {
            TrackId             = trackId,
            Box                 = new DetectionBoundingBox(cx - r, cy - r, cx + r, cy + r),
            DetectionConfidence = conf,
            Age                 = 10,
            VisibleFrames       = 10,
            MissingFrames       = 0,
            StabilityScore      = 0.9f,
            VelocityX           = 0f,
            VelocityY           = 0f,
        };

    // ── Tests ──────────────────────────────────────────────────────────────────

    [Fact]
    public void GetGamepadAssistDelta_NoFocus_ReturnsZero()
    {
        var runner = MakeRunner();
        var (rx, ry) = runner.GetGamepadAssistDelta(null, 1920, 1080, 1.0 / 60.0);
        Assert.Equal(0f, rx);
        Assert.Equal(0f, ry);
    }

    [Fact]
    public void GetGamepadAssistDelta_EmergencyStopped_ReturnsZero()
    {
        var runner = MakeRunner();
        runner.EmergencyStop();
        var focus = FocusAt(100f, 100f);  // far from center
        var (rx, ry) = runner.GetGamepadAssistDelta(focus, 1920, 1080, 1.0 / 60.0);
        Assert.Equal(0f, rx);
        Assert.Equal(0f, ry);
    }

    [Fact]
    public void GetGamepadAssistDelta_TargetAtCenter_IsNearZero()
    {
        // Focus exactly at screen center → error is 0 → assist should be in deadband
        var runner = MakeRunner();
        var focus = FocusAt(960f, 540f);
        var (rx, ry) = runner.GetGamepadAssistDelta(focus, 1920, 1080, 1.0 / 60.0);
        Assert.InRange(rx, -0.02f, 0.02f);
        Assert.InRange(ry, -0.02f, 0.02f);
    }

    [Fact]
    public void GetGamepadAssistDelta_TargetRight_ProducesPositiveRx()
    {
        // Target at right side → errorX > 0 → after several frames, RX > 0
        var runner = MakeRunner(new GamepadAssistController { MaxSlewRate = 100f });
        var focus = FocusAt(1440f, 540f);  // 75% across screen = errorX = +0.25

        float rx = 0f;
        for (int i = 0; i < 10; i++)
            (rx, _) = runner.GetGamepadAssistDelta(focus, 1920, 1080, 1.0 / 60.0);

        Assert.True(rx > 0f, $"Expected RX > 0 for right-side target, got {rx}");
    }

    [Fact]
    public void GetGamepadAssistDelta_TargetLeft_ProducesNegativeRx()
    {
        var runner = MakeRunner(new GamepadAssistController { MaxSlewRate = 100f });
        var focus = FocusAt(480f, 540f);  // 25% across = errorX = -0.25

        float rx = 0f;
        for (int i = 0; i < 10; i++)
            (rx, _) = runner.GetGamepadAssistDelta(focus, 1920, 1080, 1.0 / 60.0);

        Assert.True(rx < 0f, $"Expected RX < 0 for left-side target, got {rx}");
    }

    [Fact]
    public void GetGamepadAssistDelta_TargetBelow_ProducesPositiveRy()
    {
        var runner = MakeRunner(new GamepadAssistController { MaxSlewRate = 100f });
        var focus = FocusAt(960f, 810f);  // 75% down = errorY = +0.25

        float ry = 0f;
        for (int i = 0; i < 10; i++)
            (_, ry) = runner.GetGamepadAssistDelta(focus, 1920, 1080, 1.0 / 60.0);

        Assert.True(ry > 0f, $"Expected RY > 0 for below-center target, got {ry}");
    }

    [Fact]
    public void GetGamepadAssistDelta_DriftAdded()
    {
        // Drift right at 0.2 sticks/s for 1 second should produce noticeable positive RX
        var drift = new ManualDriftCompensator();
        drift.UpdateProfile(new DriftCompensationProfile
        {
            HorizontalOffsetPerSecond = 0.2f,
            VerticalOffsetPerSecond   = 0f,
            Enabled                   = true,
        });

        var runner = MakeRunner(
            // Controller with gain=0 so assist is zero, isolating drift
            new GamepadAssistController { Gain = 0f, GainNear = 0f, IntegralGain = 0f },
            drift);

        float sumRx = 0f;
        for (int i = 0; i < 60; i++)
            (sumRx, _) = runner.GetGamepadAssistDelta(null, 1920, 1080, 1.0 / 60.0);

        // 60 frames × (0.2 / 60) per frame ≈ 0.2, but sumRx is the last value not sum.
        // Last returned value after 1 second of drift: drift per call = 0.2/60 ≈ 0.003333 per frame.
        // The returned value IS the single-frame delta (not cumulative), so last rx = drift + assist.
        // For drift-only (no focus), rx = driftX = 0.2 × (1/60) ≈ 0.003333
        float expectedDriftPerFrame = 0.2f / 60f;
        Assert.Equal(expectedDriftPerFrame, sumRx, precision: 4);
    }

    [Fact]
    public void GetGamepadAssistDelta_LostTrack_OutputDecaysToZero()
    {
        var runner = MakeRunner(new GamepadAssistController
        {
            MaxSlewRate            = 100f,
            NoTargetErrorDecayTimeMs   = 10.0,
            NoTargetIntegralDecayTimeMs = 20.0,
        });

        // Build up assist for a right-side target
        var focus = FocusAt(1440f, 540f);
        for (int i = 0; i < 10; i++)
            runner.GetGamepadAssistDelta(focus, 1920, 1080, 1.0 / 60.0);

        // Now remove focus and drive to zero
        float rx = 0f;
        for (int i = 0; i < 200; i++)
            (rx, _) = runner.GetGamepadAssistDelta(null, 1920, 1080, 1.0 / 60.0);

        Assert.Equal(0f, rx, 2);
    }

    [Fact]
    public void GetGamepadAssistDelta_OutputClampedToUnitRange()
    {
        var runner = MakeRunner(new GamepadAssistController { Gain = 100f, MaxSlewRate = 1000f });

        // Extreme error: target at far corner
        var focus = FocusAt(1900f, 1060f);
        (float rx, float ry) = (0f, 0f);
        for (int i = 0; i < 10; i++)
            (rx, ry) = runner.GetGamepadAssistDelta(focus, 1920, 1080, 1.0 / 60.0);

        Assert.InRange(rx, -1f, 1f);
        Assert.InRange(ry, -1f, 1f);
    }
}
