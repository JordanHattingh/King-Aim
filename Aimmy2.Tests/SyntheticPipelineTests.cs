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

public sealed class SyntheticPipelineTests
{
    // ---------------------------------------------------------------------------
    // Test doubles
    // ---------------------------------------------------------------------------

    private sealed class RecordingEventSink : IAccessibilityEventSink
    {
        private readonly List<AccessibilityEvent> _events = [];
        public IReadOnlyList<AccessibilityEvent> Events => _events;
        public void OnEvent(AccessibilityEvent evt) => _events.Add(evt);
        public void Clear() => _events.Clear();
    }

    private sealed class NullVisualCue : IVisualCueProvider
    {
        public bool  IsEnabled          { get; set; } = true;
        public bool  HighContrastMode   { get; set; }
        public bool  ColourBlindSafeMode { get; set; }
        public bool  ReducedMotionMode  { get; set; }
        public float Opacity            { get; set; } = 1f;
        public float LineThickness      { get; set; } = 2f;
        public void Update(SceneState scene, TrackState? focus) { }
        public void OnEvent(AccessibilityEvent evt) { }
    }

    private sealed class NullAudioCue : IAudioCueProvider
    {
        public bool  IsEnabled           { get; set; } = true;
        public float MaxVolumeNormalized { get; set; } = 1f;
        public float MinCueIntervalMs    { get; set; } = 100f;
        public bool  MonoCompatibility   { get; set; }
        public bool  ReducedSensoryLoad  { get; set; }
        public void Update(SceneState scene, TrackState? focus) { }
        public void OnEvent(AccessibilityEvent evt) { }
    }

    private sealed class NullHapticCue : IHapticCueProvider
    {
        public bool  IsEnabled           { get; set; } = true;
        public float MaxStrength         { get; set; } = 1f;
        public int   MaxPulseDurationMs  { get; set; } = 500;
        public int   MinPulseIntervalMs  { get; set; } = 200;
        public int   ContinuousTimeoutMs { get; set; } = 5000;
        public bool  InvertLeftRight     { get; set; }
        public void Update(SceneState scene, TrackState? focus) { }
        public void OnEvent(AccessibilityEvent evt) { }
    }

    private sealed class NullPointing : IPointingAssistController
    {
        public PointingConstraints Constraints { get; } = new();
        public bool IsEngaged   { get; private set; }
        public bool IsAvailable => false;
        public void SetFocusTarget(TrackState? focus, SceneState scene) { }
        public (float DeltaX, float DeltaY) Update() => (0f, 0f);
        public void Engage()        { IsEngaged = true; }
        public void Disengage()     { IsEngaged = false; }
        public void EmergencyStop() { IsEngaged = false; }
    }

    // ---------------------------------------------------------------------------
    // Fixture helper
    // ---------------------------------------------------------------------------

    private sealed class PipelineFixture : IDisposable
    {
        public readonly SyntheticPatternSource Source;
        public readonly FrameScheduler         Scheduler;
        public readonly LetterboxPreprocessor  Preprocessor;
        public readonly MockInferenceEngine    Engine;
        public readonly MockPoseDecoder        Decoder;
        public readonly GeometryValidator      Validator;
        public readonly SimpleTrackerService   Tracker;
        public readonly SceneAnalyzer          Scene;
        public readonly FocusSelector          Focus;
        public readonly AccessibilityEventDispatcher Dispatcher;
        public readonly RecordingEventSink     Sink;
        public readonly DiagnosticsService     Diagnostics;
        public readonly PipelineRunner         Runner;
        public readonly SafetyPolicy           Safety;

        public PipelineFixture(OperatingMode mode = OperatingMode.AccessibilityTestHarness,
                               int width = 1920, int height = 1080)
        {
            Safety      = new SafetyPolicy(mode);
            Source      = new SyntheticPatternSource(width, height);
            Scheduler   = new FrameScheduler();
            Preprocessor = new LetterboxPreprocessor(512);
            Engine      = new MockInferenceEngine();
            Decoder     = new MockPoseDecoder();
            Validator   = new GeometryValidator();
            Tracker     = new SimpleTrackerService();
            Scene       = new SceneAnalyzer();
            Focus       = new FocusSelector();
            Dispatcher  = new AccessibilityEventDispatcher();
            Sink        = new RecordingEventSink();
            Diagnostics = new DiagnosticsService();

            Dispatcher.AddSink(Sink);

            Runner = new PipelineRunner(
                Safety, Source, Scheduler, Preprocessor,
                Engine, Decoder, Validator,
                Tracker, Scene, Focus, Dispatcher,
                new NullVisualCue(), new NullAudioCue(), new NullHapticCue(),
                new NullPointing(), new ManualDriftCompensator(), Diagnostics);
        }

        public async Task<PipelineRunResult> TickAsync()
        {
            return await Runner.TickAsync();
        }

        public void Dispose() => Source.Dispose();

        private static PoseDetection MakeDetection(
            float l, float t, float r, float b,
            float conf = 0.85f, Guid? id = null)
            => new()
            {
                DetectionId       = id ?? Guid.NewGuid(),
                BoundingBox       = new DetectionBoundingBox(l, t, r, b),
                ObjectConfidence  = conf,
                SourceFrameId     = 0,
                CaptureTimestampUs = 0,
                InferenceTimestampUs = 0,
                ModelId           = "mock",
                Keypoints         = new[]
                {
                    new PoseKeypoint { Name = KeypointName.Head,       X = (l+r)/2, Y = t + (b-t)*0.1f, Confidence = 0.9f, Visibility = KeypointVisibility.Visible },
                    new PoseKeypoint { Name = KeypointName.Neck,       X = (l+r)/2, Y = t + (b-t)*0.25f, Confidence = 0.9f, Visibility = KeypointVisibility.Visible },
                    new PoseKeypoint { Name = KeypointName.UpperChest, X = (l+r)/2, Y = t + (b-t)*0.5f,  Confidence = 0.9f, Visibility = KeypointVisibility.Visible },
                    new PoseKeypoint { Name = KeypointName.Hip,        X = (l+r)/2, Y = b - (b-t)*0.1f,  Confidence = 0.9f, Visibility = KeypointVisibility.Visible },
                },
            };

        public PoseDetection EnemyAt(float l, float t, float r, float b, float conf = 0.85f)
            => MakeDetection(l, t, r, b, conf);
    }

    // ---------------------------------------------------------------------------
    // Test 1 — End-to-end produces a stable tracked focus across 30 frames
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task SyntheticPipeline_EndToEnd_ProducesStableTrackedFocus()
    {
        using var fx = new PipelineFixture();
        await fx.Source.StartAsync();

        var enemy = fx.EnemyAt(460, 250, 560, 380);
        fx.Decoder.SetDetections([enemy]);

        int? firstFocusId = null;
        for (int i = 0; i < 30; i++)
        {
            var result = await fx.TickAsync();
            Assert.False(result.Skipped);

            if (i >= 5 && result.Focus != null)
            {
                firstFocusId ??= result.Focus.TrackId;
                Assert.Equal(firstFocusId, result.Focus.TrackId);
            }
        }

        Assert.NotNull(firstFocusId);
    }

    // ---------------------------------------------------------------------------
    // Test 2 — Letterbox round-trip preserves source coordinates
    // ---------------------------------------------------------------------------

    [Fact]
    public void SyntheticPipeline_LetterboxRoundTrip_PreservesCoordinates()
    {
        var preprocessor = new LetterboxPreprocessor(512);
        var frame = new KingAim.Core.Capture.CapturedFrame
        {
            FrameId     = 1,
            SourceWidth = 1920,
            SourceHeight = 1080,
            PixelData   = new byte[1920 * 1080 * 3],
        };

        var pf = preprocessor.Preprocess(frame);

        float srcX = 960f, srcY = 540f;
        var (mx, my) = pf.Meta.SourceToModel(srcX, srcY);
        var (rx, ry) = pf.Meta.ModelToSource(mx, my);

        Assert.InRange(rx, srcX - 0.5f, srcX + 0.5f);
        Assert.InRange(ry, srcY - 0.5f, srcY + 0.5f);
    }

    // ---------------------------------------------------------------------------
    // Test 3 — Stale frames are dropped, not queued
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task SyntheticPipeline_DropsStaleFrames()
    {
        var scheduler = new FrameScheduler();
        var source    = new SyntheticPatternSource(320, 240);
        await source.StartAsync();

        // Post 5 without consuming
        for (int i = 0; i < 5; i++)
        {
            var f = source.GetLatestFrame()!;
            scheduler.Post(f);
        }

        // Only one should survive
        Assert.True(scheduler.TryTake(out _));
        Assert.False(scheduler.TryTake(out _));
        Assert.True(scheduler.FramesDropped >= 4);
    }

    // ---------------------------------------------------------------------------
    // Test 4 — Target loss emits exactly one FocusTargetLost event
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task SyntheticPipeline_TargetLoss_EmitsSingleLostEvent()
    {
        using var fx = new PipelineFixture();
        await fx.Source.StartAsync();

        // Warm up focus
        fx.Decoder.SetDetections([fx.EnemyAt(460, 250, 560, 380)]);
        for (int i = 0; i < 8; i++) await fx.TickAsync();

        fx.Sink.Clear();

        // Remove target
        fx.Decoder.ClearDetections();

        // Tick until tracker expires the track (maxMissingFrames = 5 + 1 more to confirm)
        for (int i = 0; i < 8; i++) await fx.TickAsync();

        var lostEvents = fx.Sink.Events
            .Where(e => e.Kind == AccessibilityEventKind.FocusTargetLost)
            .ToList();

        Assert.Single(lostEvents);
    }

    // ---------------------------------------------------------------------------
    // Test 5 — Target reappears and tracking is restored safely
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task SyntheticPipeline_TargetReappears_RestoresTrackingSafely()
    {
        using var fx = new PipelineFixture();
        await fx.Source.StartAsync();

        fx.Decoder.SetDetections([fx.EnemyAt(460, 250, 560, 380)]);
        for (int i = 0; i < 8; i++) await fx.TickAsync();

        // Disappear for 3 frames (within maxMissingFrames)
        fx.Decoder.ClearDetections();
        for (int i = 0; i < 3; i++) await fx.TickAsync();

        fx.Sink.Clear();

        // Reappear at same location
        fx.Decoder.SetDetections([fx.EnemyAt(460, 250, 560, 380)]);
        for (int i = 0; i < 5; i++) await fx.TickAsync();

        var result = await fx.TickAsync();
        Assert.NotNull(result.Focus);
        Assert.True(result.Focus!.DetectionConfidence > 0.5f);
    }

    // ---------------------------------------------------------------------------
    // Test 6 — Multiple targets: focus stays sticky on the closer one
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task SyntheticPipeline_MultipleTargets_UsesStickyFocusSelection()
    {
        using var fx = new PipelineFixture();
        await fx.Source.StartAsync();

        // Enemy A close to centre, Enemy B off to the side
        var enemyA = fx.EnemyAt(900, 490, 1020, 610, conf: 0.90f);
        var enemyB = fx.EnemyAt(100,  50,  200, 150, conf: 0.85f);
        fx.Decoder.SetDetections([enemyA, enemyB]);

        // Warm up so focus settles on A
        for (int i = 0; i < 12; i++) await fx.TickAsync();

        int? warmFocusId = fx.Focus.CurrentFocus?.TrackId;
        Assert.NotNull(warmFocusId);

        // Run 10 more frames — focus must not flip constantly
        int switches = 0;
        int? prevId  = warmFocusId;
        for (int i = 0; i < 10; i++)
        {
            var r = await fx.TickAsync();
            if (r.Focus?.TrackId != prevId) { switches++; prevId = r.Focus?.TrackId; }
        }

        // Stickiness should prevent more than 1 switch over 10 frames
        Assert.True(switches <= 1, $"Focus switched {switches} times — expected stickiness");
    }

    // ---------------------------------------------------------------------------
    // Test 7 — EmergencyDisable suppresses all outputs
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task SyntheticPipeline_EmergencyDisable_SuppressesAllOutputs()
    {
        using var fx = new PipelineFixture();
        await fx.Source.StartAsync();
        fx.Decoder.SetDetections([fx.EnemyAt(460, 250, 560, 380)]);
        for (int i = 0; i < 5; i++) await fx.TickAsync();

        fx.Sink.Clear();
        fx.Runner.EmergencyStop();

        // New enemy should not cause any events
        fx.Decoder.SetDetections([fx.EnemyAt(100, 100, 200, 200)]);
        for (int i = 0; i < 5; i++) await fx.TickAsync();

        Assert.DoesNotContain(fx.Sink.Events, e => e.Kind != AccessibilityEventKind.EmergencyDisabled);
    }

    // ---------------------------------------------------------------------------
    // Test 8 — Prohibited capabilities are always rejected
    // ---------------------------------------------------------------------------

    [Fact]
    public void SyntheticPipeline_ProhibitedCapabilities_AreAlwaysRejected()
    {
        var modes = new[] { OperatingMode.LiveCapture, OperatingMode.StaticImages,
                            OperatingMode.RecordedVideo, OperatingMode.AccessibilityTestHarness };

        var prohibited = new[]
        {
            AssistanceCapability.AutomaticFiring,
            AssistanceCapability.RecoilControl,
            AssistanceCapability.ProcessInjectionOrMemoryAccess,
            AssistanceCapability.AntiCheatInteraction,
            AssistanceCapability.StealthOperation,
            AssistanceCapability.UnattendedOperation,
            AssistanceCapability.NetworkManipulation,
        };

        foreach (var mode in modes)
        {
            var policy = new SafetyPolicy(mode);
            foreach (var cap in prohibited)
            {
                Assert.False(policy.IsPermitted(cap), $"{cap} should be prohibited in {mode}");
                Assert.Throws<SafetyViolationException>(() => policy.Require(cap));
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Test 9 — ControlledPointingAssistance is disabled in non-live modes
    // ---------------------------------------------------------------------------

    [Fact]
    public void SyntheticPipeline_PointingDisabledOutsideApprovedMode()
    {
        var noPointing = new[]
        {
            OperatingMode.StaticImages,
            OperatingMode.AccessibilityTestHarness,
        };
        var hasPointing = new[]
        {
            OperatingMode.LiveCapture,
            OperatingMode.RecordedVideo,
        };

        foreach (var mode in noPointing)
        {
            var policy = new SafetyPolicy(mode);
            Assert.False(policy.IsPermitted(AssistanceCapability.ControlledPointingAssistance),
                $"Pointing should be off in {mode}");
        }

        foreach (var mode in hasPointing)
        {
            var policy = new SafetyPolicy(mode);
            Assert.True(policy.IsPermitted(AssistanceCapability.ControlledPointingAssistance),
                $"Pointing should be on in {mode}");
        }
    }

    // ---------------------------------------------------------------------------
    // Test 10 — Pipeline produces a populated diagnostics snapshot
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task SyntheticPipeline_ProducesDiagnosticsSnapshot()
    {
        using var fx = new PipelineFixture();
        await fx.Source.StartAsync();
        fx.Decoder.SetDetections([fx.EnemyAt(460, 250, 560, 380)]);

        for (int i = 0; i < 10; i++) await fx.TickAsync();

        var snap = fx.Diagnostics.Current;
        Assert.True(snap.ActiveTracks >= 1, "At least one track should be active");
        Assert.NotNull(snap.FocusTrackId);
        Assert.True(snap.InferenceMedianMs >= 0);
    }
}
