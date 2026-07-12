using KingAim.Core.Accessibility;
using KingAim.Core.Accessibility.Cues;
using KingAim.Core.Accessibility.Input;
using KingAim.Core.Accessibility.Pointing;
using KingAim.Core.Capture;
using KingAim.Core.Decoding;
using KingAim.Core.Diagnostics;
using KingAim.Core.Focus;
using KingAim.Core.Inference;
using KingAim.Core.Perception;
using KingAim.Core.Preprocessing;
using KingAim.Core.Safety;
using KingAim.Core.Scene;
using KingAim.Core.Scheduling;
using KingAim.Core.Tracking;
using KingAim.Core.Validation;

namespace KingAim.Core.Pipeline;

/// <summary>Result returned by a single pipeline tick.</summary>
public sealed class PipelineRunResult
{
    public static readonly PipelineRunResult NoFrame = new() { Skipped = true };

    public bool   Skipped    { get; init; }
    public CapturedFrame?             Frame       { get; init; }
    public IReadOnlyList<PoseDetection> Detections { get; init; } = [];
    public IReadOnlyList<TrackState>    Tracks      { get; init; } = [];
    public SceneState?                  Scene       { get; init; }
    public TrackState?                  Focus       { get; init; }
}

/// <summary>
/// Wires the full capture → preprocess → infer → decode → validate → track →
/// scene → focus → dispatch → cue cycle into a single testable unit.
/// </summary>
public sealed class PipelineRunner
{
    private readonly SafetyPolicy                _safety;
    private readonly IFrameSource                _source;
    private readonly FrameScheduler              _scheduler;
    private readonly IFramePreprocessor          _preprocessor;
    private readonly IInferenceEngine            _engine;
    private readonly IModelDecoder               _decoder;
    private readonly IGeometryValidator          _validator;
    private readonly ITrackerService             _tracker;
    private readonly ISceneAnalyzer              _scene;
    private readonly IFocusSelector              _focus;
    private readonly AccessibilityEventDispatcher _dispatcher;
    private readonly IVisualCueProvider          _visual;
    private readonly IAudioCueProvider           _audio;
    private readonly IHapticCueProvider          _haptic;
    private readonly IPointingAssistController   _pointing;
    private readonly IDriftCompensator           _drift;
    private readonly IGamepadAssistController    _gamepadAssist;
    private readonly DiagnosticsService          _diagnostics;

    public PipelineRunner(
        SafetyPolicy                 safety,
        IFrameSource                 source,
        FrameScheduler               scheduler,
        IFramePreprocessor           preprocessor,
        IInferenceEngine             engine,
        IModelDecoder                decoder,
        IGeometryValidator           validator,
        ITrackerService              tracker,
        ISceneAnalyzer               scene,
        IFocusSelector               focus,
        AccessibilityEventDispatcher dispatcher,
        IVisualCueProvider           visual,
        IAudioCueProvider            audio,
        IHapticCueProvider           haptic,
        IPointingAssistController    pointing,
        IDriftCompensator            drift,
        IGamepadAssistController     gamepadAssist,
        DiagnosticsService           diagnostics)
    {
        _safety      = safety;
        _source      = source;
        _scheduler   = scheduler;
        _preprocessor = preprocessor;
        _engine      = engine;
        _decoder     = decoder;
        _validator   = validator;
        _tracker     = tracker;
        _scene       = scene;
        _focus       = focus;
        _dispatcher  = dispatcher;
        _visual      = visual;
        _audio       = audio;
        _haptic      = haptic;
        _pointing      = pointing;
        _drift         = drift;
        _gamepadAssist = gamepadAssist;
        _diagnostics   = diagnostics;
    }

    /// <summary>
    /// Runs one complete pipeline tick synchronously (for deterministic integration tests).
    /// Returns <see cref="PipelineRunResult.NoFrame"/> when no frame is available.
    /// </summary>
    public async Task<PipelineRunResult> TickAsync(CancellationToken ct = default)
    {
        if (_safety.IsEmergencyDisabled)
            return PipelineRunResult.NoFrame;

        // 1. Capture
        var raw = _source.GetLatestFrame();
        if (raw == null) return PipelineRunResult.NoFrame;

        // 2. Schedule (drops stale)
        _scheduler.Post(raw);
        if (!_scheduler.TryTake(out var frame) || frame == null)
            return PipelineRunResult.NoFrame;

        // 3. Preprocess
        var preprocessed = _preprocessor.Preprocess(frame);

        // 4. Infer
        var input  = new InferenceInput
        {
            Tensor  = preprocessed.Tensor,
            FrameId = preprocessed.FrameId,
            Meta    = preprocessed.Meta,
        };
        var output = await _engine.RunAsync(input, ct);

        // 5. Decode
        var detections = _decoder.Decode(output, preprocessed.Meta);

        // 6. Validate — filter rejected
        var valid = new List<PoseDetection>(detections.Count);
        foreach (var d in detections)
        {
            var r = _validator.Validate(d, frame.SourceWidth, frame.SourceHeight);
            if (r.Quality != DetectionQuality.Rejected)
                valid.Add(d);
        }

        // 7. Track
        var tracks = _tracker.Update(valid, frame.FrameId, frame.CaptureTimestampUs);

        // 8. Scene
        var sceneState = _scene.Analyze(tracks, frame.FrameId, frame.SourceWidth, frame.SourceHeight);

        // 9. Focus
        var focusTrack = _focus.SelectFocus(sceneState, frame.SourceWidth, frame.SourceHeight);

        // Attach focus back onto scene for dispatcher
        var finalScene = new SceneState
        {
            FrameId               = sceneState.FrameId,
            ActiveTracks          = sceneState.ActiveTracks,
            PrimaryTrack          = focusTrack,
            VisibleEnemyCount     = sceneState.VisibleEnemyCount,
            NearestCentreTrack    = sceneState.NearestCentreTrack,
            HighestConfidenceTrack = sceneState.HighestConfidenceTrack,
            SceneStability        = sceneState.SceneStability,
        };

        // 10. Dispatch events
        _dispatcher.Dispatch(finalScene, frame.FrameId, frame.SourceWidth, frame.SourceHeight);

        // 11. Cues
        if (!_safety.IsEmergencyDisabled)
        {
            _visual.Update(finalScene, focusTrack);
            _audio.Update(finalScene, focusTrack);
            _haptic.Update(finalScene, focusTrack);
            _pointing.SetFocusTarget(focusTrack, finalScene);
        }

        // 12. Diagnostics
        if (output.InferenceUs > 0)
            _diagnostics.RecordInferenceLatency(output.InferenceUs / 1000.0);
        if (_scheduler.FramesDropped > 0)
            _diagnostics.RecordDroppedFrame();

        _diagnostics.UpdateTrackStats(
            activeTracks:    tracks.Count,
            focusTrackId:    focusTrack?.TrackId,
            modelConfidence: focusTrack?.DetectionConfidence ?? 0f);

        return new PipelineRunResult
        {
            Frame      = frame,
            Detections = valid,
            Tracks     = tracks,
            Scene      = finalScene,
            Focus      = focusTrack,
        };
    }

    /// <summary>
    /// Returns the combined pointer correction for this polling interval.
    /// Call this from the mouse output loop (not from TickAsync).
    /// Pointing assist contributes only when engaged; drift runs whenever enabled.
    /// Both components return fractional screen units; the caller converts to pixels.
    /// </summary>
    public (float DeltaX, float DeltaY) GetPointerDelta(double deltaSeconds)
    {
        if (_safety.IsEmergencyDisabled) return (0f, 0f);
        var (px, py) = _pointing.Update();
        var (dx, dy) = _drift.Compensate(deltaSeconds);
        return (px + dx, py + dy);
    }

    /// <summary>
    /// Returns the combined gamepad right-stick assist delta for this polling interval.
    /// Call this from the gamepad output loop (not from TickAsync).
    /// The returned values are in stick space (-1..1). Blend with physical input via
    /// <c>StickBlender.Blend</c> before passing to <c>IGamepadOutput.SetFullState</c>.
    /// </summary>
    /// <param name="focus">Current focus track from the most recent TickAsync result.</param>
    /// <param name="screenW">Capture width in pixels.</param>
    /// <param name="screenH">Capture height in pixels.</param>
    /// <param name="deltaSeconds">Elapsed time since the last gamepad update call.</param>
    public (float AssistRx, float AssistRy) GetGamepadAssistDelta(
        TrackState? focus, int screenW, int screenH, double deltaSeconds)
    {
        if (_safety.IsEmergencyDisabled) return (0f, 0f);

        bool hasTarget = focus != null && !focus.IsLost;
        float errorX = 0f, errorY = 0f, velX = 0f, velY = 0f;
        float confidence = 0f;
        double observationAgeMs = 0.0;
        int? trackId = null;

        if (hasTarget)
        {
            float cx = focus!.Box.CentreX;
            float cy = focus.Box.CentreY;
            errorX = screenW > 0 ? (cx / screenW) - 0.5f : 0f;
            errorY = screenH > 0 ? (cy / screenH) - 0.5f : 0f;
            velX = screenW > 0 ? focus.VelocityX / screenW : 0f;
            velY = screenH > 0 ? focus.VelocityY / screenH : 0f;
            confidence = focus.DetectionConfidence;
            observationAgeMs = focus.MissingFrames * Math.Max(deltaSeconds, 1.0 / 240.0) * 1000.0;
            trackId = focus.TrackId;
        }

        var (assistRx, assistRy) = _gamepadAssist.Update(
            hasTarget, errorX, errorY, velX, velY,
            confidence, observationAgeMs, deltaSeconds, trackId);

        var (driftX, driftY) = _drift.Compensate(deltaSeconds);

        return (
            Math.Clamp(assistRx + driftX, -1f, 1f),
            Math.Clamp(assistRy + driftY, -1f, 1f));
    }

    public void EmergencyStop()
    {
        _safety.EmergencyDisable();
        _dispatcher.EmergencyDisable();
    }
}
