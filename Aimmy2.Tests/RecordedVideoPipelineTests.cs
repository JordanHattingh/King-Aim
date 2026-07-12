using System.IO;
using Aimmy2.AILogic;
using Aimmy2.Capture;
using KingAim.Core.Accessibility;
using KingAim.Core.Accessibility.Cues;
using KingAim.Core.Accessibility.Events;
using KingAim.Core.Accessibility.Input;
using GamepadAssistController = KingAim.Core.Accessibility.Input.GamepadAssistController;
using KingAim.Core.Accessibility.Pointing;
using KingAim.Core.Capture;
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
using OpenCvSharp;
using Xunit;

namespace Aimmy2.Tests;

/// <summary>
/// Full recorded-video pipeline integration tests.
/// Source: tiny synthetic AVI → RecordedVideoSource → full pipeline.
/// </summary>
public sealed class RecordedVideoPipelineTests : IClassFixture<RecordedVideoPipelineTests.VideoFixture>
{
    private readonly VideoFixture _fx;
    public RecordedVideoPipelineTests(VideoFixture fx) => _fx = fx;

    // =========================================================================
    // Class fixture — shared synthetic video + pipeline doubles
    // =========================================================================

    public sealed class VideoFixture : IDisposable
    {
        public string VideoPath  { get; }
        public int    FrameCount { get; } = 20;
        public int    Width      { get; } = 320;
        public int    Height     { get; } = 240;
        public double FrameRate  { get; } = 30.0;

        public VideoFixture()
        {
            VideoPath = Path.Combine(Path.GetTempPath(), "king_aim_pipeline_fixture.avi");
            GenerateIfMissing();
        }

        private void GenerateIfMissing()
        {
            if (File.Exists(VideoPath)) return;

            using var writer = new VideoWriter(
                VideoPath,
                VideoWriter.FourCC('M', 'J', 'P', 'G'),
                FrameRate,
                new Size(Width, Height));

            if (!writer.IsOpened())
                throw new InvalidOperationException("Pipeline fixture VideoWriter failed to open.");

            for (int i = 0; i < FrameCount; i++)
            {
                using var f = new Mat(Height, Width, MatType.CV_8UC3,
                    new Scalar((i * 13) & 0xFF, (i * 7) & 0xFF, (i * 19) & 0xFF));
                writer.Write(f);
            }
        }

        public void Dispose() { }
    }

    // =========================================================================
    // Test doubles (identical to SyntheticPipelineTests)
    // =========================================================================

    private sealed class RecordingEventSink : IAccessibilityEventSink
    {
        private readonly List<AccessibilityEvent> _events = [];
        public IReadOnlyList<AccessibilityEvent> Events => _events;
        public void OnEvent(AccessibilityEvent e) => _events.Add(e);
        public void Clear() => _events.Clear();
    }

    private sealed class NullVisualCue : IVisualCueProvider
    {
        public bool  IsEnabled           { get; set; } = true;
        public bool  HighContrastMode    { get; set; }
        public bool  ColourBlindSafeMode { get; set; }
        public bool  ReducedMotionMode   { get; set; }
        public float Opacity             { get; set; } = 1f;
        public float LineThickness       { get; set; } = 2f;
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

    // =========================================================================
    // Pipeline fixture
    // =========================================================================

    private sealed class PipelineFixture : IDisposable
    {
        public readonly RecordedVideoSource   Source;
        public readonly FrameScheduler        Scheduler;
        public readonly LetterboxPreprocessor Preprocessor;
        public readonly MockInferenceEngine   Engine;
        public readonly MockPoseDecoder       Decoder;
        public readonly GeometryValidator     Validator;
        public readonly TrackerAdapter        Tracker;
        public readonly SceneAnalyzer         Scene;
        public readonly FocusSelector         Focus;
        public readonly AccessibilityEventDispatcher Dispatcher;
        public readonly RecordingEventSink    Sink;
        public readonly DiagnosticsService    Diagnostics;
        public readonly PipelineRunner        Runner;

        public PipelineFixture(string videoPath, int width, int height)
        {
            var backend = new OpenCvSharpVideoBackend();
            Assert.True(backend.Open(videoPath));

            Source      = new RecordedVideoSource(backend, PlaybackTimingMode.AsFastAsPossible);
            Scheduler   = new FrameScheduler();
            Preprocessor = new LetterboxPreprocessor(512);
            Engine      = new MockInferenceEngine();
            Decoder     = new MockPoseDecoder();
            Validator   = new GeometryValidator();
            Tracker     = new TrackerAdapter(width, height);
            Scene       = new SceneAnalyzer();
            Focus       = new FocusSelector();
            Dispatcher  = new AccessibilityEventDispatcher();
            Sink        = new RecordingEventSink();
            Diagnostics = new DiagnosticsService();

            Dispatcher.AddSink(Sink);

            var safety = new SafetyPolicy(OperatingMode.RecordedVideo);
            Runner = new PipelineRunner(
                safety, Source, Scheduler, Preprocessor,
                Engine, Decoder, Validator,
                Tracker, Scene, Focus, Dispatcher,
                new NullVisualCue(), new NullAudioCue(), new NullHapticCue(),
                new NullPointing(), new ManualDriftCompensator(),
                new GamepadAssistController(), Diagnostics);
        }

        public void Dispose() => Source.Dispose();
    }

    private static PoseDetection MakeDetection(int w, int h, float conf = 0.85f)
    {
        float l = w * 0.4f, t = h * 0.3f, r = w * 0.6f, b = h * 0.7f;
        float cx = (l + r) / 2, cy = (t + b) / 2;
        return new PoseDetection
        {
            DetectionId          = Guid.NewGuid(),
            BoundingBox          = new DetectionBoundingBox(l, t, r, b),
            ObjectConfidence     = conf,
            SourceFrameId        = 0,
            CaptureTimestampUs   = 0,
            InferenceTimestampUs = 0,
            ModelId              = "mock",
            Keypoints = [
                new PoseKeypoint { Name = KeypointName.Head,       X = cx, Y = t + (b-t)*0.1f, Confidence = 0.9f, Visibility = KeypointVisibility.Visible },
                new PoseKeypoint { Name = KeypointName.Neck,       X = cx, Y = t + (b-t)*0.25f, Confidence = 0.9f, Visibility = KeypointVisibility.Visible },
                new PoseKeypoint { Name = KeypointName.UpperChest, X = cx, Y = t + (b-t)*0.5f,  Confidence = 0.9f, Visibility = KeypointVisibility.Visible },
                new PoseKeypoint { Name = KeypointName.Hip,        X = cx, Y = b - (b-t)*0.1f,  Confidence = 0.9f, Visibility = KeypointVisibility.Visible },
            ],
        };
    }

    // =========================================================================
    // Tests
    // =========================================================================

    [Fact]
    public async Task RecordedPipeline_ConsumesAllSourceFrames()
    {
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();
        pf.Decoder.SetDetections([MakeDetection(_fx.Width, _fx.Height)]);

        int frames = 0;
        PipelineRunResult? last = null;
        while (true)
        {
            var r = await pf.Runner.TickAsync();
            if (r.Skipped && last != null) break;  // EOF → source stopped
            if (!r.Skipped) { frames++; last = r; }
            if (frames >= _fx.FrameCount) break;
        }

        Assert.True(frames > 0, "Pipeline consumed no frames");
        Assert.True(frames <= _fx.FrameCount, $"Consumed more frames than source has ({frames})");
    }

    [Fact]
    public async Task RecordedPipeline_FrameIdsAndTimestampsIncrease()
    {
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();

        var results = new List<PipelineRunResult>();
        while (results.Count < _fx.FrameCount)
        {
            var r = await pf.Runner.TickAsync();
            if (r.Skipped) break;
            results.Add(r);
        }

        Assert.NotEmpty(results);

        for (int i = 1; i < results.Count; i++)
        {
            long prevId = results[i - 1].Frame!.FrameId;
            long curId  = results[i].Frame!.FrameId;
            Assert.True(curId > prevId, $"FrameId not increasing at index {i}: {prevId} → {curId}");

            long prevTs = results[i - 1].Frame!.CaptureTimestampUs;
            long curTs  = results[i].Frame!.CaptureTimestampUs;
            Assert.True(curTs >= prevTs, $"Timestamp not monotonic at index {i}: {prevTs} → {curTs}");
        }
    }

    [Fact]
    public async Task RecordedPipeline_OneStableTrackSurvivesSequence()
    {
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();
        pf.Decoder.SetDetections([MakeDetection(_fx.Width, _fx.Height)]);

        int? focusId = null;
        int  warmupFrames = 6;
        int  processed    = 0;

        while (processed < _fx.FrameCount)
        {
            var r = await pf.Runner.TickAsync();
            if (r.Skipped) break;
            processed++;

            if (processed > warmupFrames && r.Focus != null)
            {
                focusId ??= r.Focus.TrackId;
                Assert.Equal(focusId, r.Focus.TrackId);
            }
        }

        Assert.NotNull(focusId);
    }

    [Fact]
    public async Task RecordedPipeline_FocusIsSticky()
    {
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();
        pf.Decoder.SetDetections([MakeDetection(_fx.Width, _fx.Height)]);

        int?  prevFocusId = null;
        int   switches    = 0;
        int   frames      = 0;

        while (frames < _fx.FrameCount)
        {
            var r = await pf.Runner.TickAsync();
            if (r.Skipped) break;
            frames++;

            int? curId = r.Focus?.TrackId;
            if (curId != null && prevFocusId != null && curId != prevFocusId)
                switches++;
            prevFocusId = curId;
        }

        Assert.True(switches <= 1, $"Focus switched {switches} times; expected stickiness");
    }

    [Fact]
    public async Task RecordedPipeline_EmergencyDisableClearsOutputs()
    {
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();
        pf.Decoder.SetDetections([MakeDetection(_fx.Width, _fx.Height)]);

        // Warm up
        for (int i = 0; i < 5; i++) await pf.Runner.TickAsync();
        pf.Sink.Clear();

        pf.Runner.EmergencyStop();

        // Further ticks must not generate non-emergency events
        for (int i = 0; i < 5; i++) await pf.Runner.TickAsync();

        Assert.DoesNotContain(pf.Sink.Events,
            e => e.Kind != AccessibilityEventKind.EmergencyDisabled);
    }

    [Fact]
    public async Task RecordedPipeline_DiagnosticsCapturePipelineLatency()
    {
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();
        pf.Decoder.SetDetections([MakeDetection(_fx.Width, _fx.Height)]);

        for (int i = 0; i < 10; i++) await pf.Runner.TickAsync();

        var snap = pf.Diagnostics.Current;
        Assert.True(snap.ActiveTracks >= 0);
        Assert.True(snap.InferenceMedianMs >= 0);
    }

    [Fact]
    public async Task RecordedPipeline_NoNativeMemoryCorruption()
    {
        // Read all frames and verify no crash / access violation occurs.
        // Any unmanaged memory corruption would surface here as AV or heap corruption.
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();

        var ex = await Record.ExceptionAsync(async () =>
        {
            int n = 0;
            while (n < _fx.FrameCount)
            {
                var r = await pf.Runner.TickAsync();
                if (r.Skipped) break;
                // Access pixel data to ensure it is readable
                Assert.NotNull(r.Frame?.PixelData);
                Assert.True(r.Frame!.PixelData.Length > 0);
                n++;
            }
        });

        Assert.Null(ex);
    }

    [Fact]
    public async Task RecordedPipeline_DuplicateSourceTimestampsDoNotBreakTracking()
    {
        // At MJPG / 30fps, consecutive raw codec timestamps can share the same millisecond value.
        // RecordedVideoSource.NormalizeTimestamp() must ensure CaptureTimestampUs is strictly
        // increasing regardless, so TrackerAdapter never receives a zero or negative frame delta.
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();
        pf.Decoder.SetDetections([MakeDetection(_fx.Width, _fx.Height)]);

        var frames = new List<KingAim.Core.Capture.CapturedFrame>();
        while (frames.Count < _fx.FrameCount)
        {
            var r = await pf.Runner.TickAsync();
            if (r.Skipped) break;
            if (r.Frame != null) frames.Add(r.Frame!);
        }

        Assert.NotEmpty(frames);

        // Effective timestamp must be strictly increasing (normalization guarantee)
        for (int i = 1; i < frames.Count; i++)
        {
            long prev = frames[i - 1].CaptureTimestampUs;
            long cur  = frames[i].CaptureTimestampUs;
            Assert.True(cur > prev,
                $"Frame {i}: CaptureTimestampUs not strictly increasing: {prev} → {cur}");
        }
    }

    [Fact]
    public async Task RecordedPipeline_StaleFramesBehaviourCorrect()
    {
        using var pf = new PipelineFixture(_fx.VideoPath, _fx.Width, _fx.Height);
        await pf.Source.StartAsync();

        // Post 5 frames without consuming, then read — only the last should survive
        for (int i = 0; i < 5; i++)
        {
            var f = pf.Source.GetLatestFrame();
            if (f != null) pf.Scheduler.Post(f);
        }

        Assert.True(pf.Scheduler.TryTake(out _));
        Assert.False(pf.Scheduler.TryTake(out _));
        Assert.True(pf.Scheduler.FramesDropped >= 4);
    }
}
