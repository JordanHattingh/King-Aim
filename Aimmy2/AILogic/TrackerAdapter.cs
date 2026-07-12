using System.Drawing;
using KingAim.Core.Perception;
using KingAim.Core.Tracking;

namespace Aimmy2.AILogic;

/// <summary>
/// Bridges KingAim.Core.Tracking.ITrackerService to the frozen TrackManager.
/// All conversion logic is contained here. TrackManager internals are not modified.
/// </summary>
public sealed class TrackerAdapter : ITrackerService
{
    private TrackManager _tm;
    private readonly int _screenLeft;
    private readonly int _screenTop;
    private readonly int _screenWidth;
    private readonly int _screenHeight;

    // Epoch anchors the pipeline's microsecond timestamps to a stable UTC DateTime.
    // This prevents wall-clock drift, DST jumps, and NTP corrections from affecting
    // frame ordering inside the frozen TrackManager (which uses DateTime internally).
    private readonly DateTime _sessionEpochUtc;

    private long _prevTimestampUs = -1;
    private readonly Dictionary<int, long> _trackFirstFrameId = [];
    private IReadOnlyList<TrackState> _last = [];

    public TrackerAdapter(
        int      screenWidth,
        int      screenHeight,
        int      screenLeft  = 0,
        int      screenTop   = 0,
        DateTime sessionEpoch = default)
    {
        if (screenWidth  <= 0) throw new ArgumentOutOfRangeException(nameof(screenWidth));
        if (screenHeight <= 0) throw new ArgumentOutOfRangeException(nameof(screenHeight));
        _screenWidth     = screenWidth;
        _screenHeight    = screenHeight;
        _screenLeft      = screenLeft;
        _screenTop       = screenTop;
        _sessionEpochUtc = sessionEpoch == default ? DateTime.UtcNow : sessionEpoch;
        _tm = BuildTrackManager();
    }

    public IReadOnlyList<TrackState> ActiveTracks => _last;

    public IReadOnlyList<TrackState> Update(
        IReadOnlyList<PoseDetection> frameDetections,
        long frameId,
        long captureTimestampUs)
    {
        // Compute actual frame delta from pipeline timestamps (avoids fixed-FPS assumption)
        double frameDeltaSeconds = _prevTimestampUs >= 0
            ? Math.Max(0.0, (captureTimestampUs - _prevTimestampUs) / 1_000_000.0)
            : 1.0 / 30.0;  // first frame: assume 30 fps until we have a second sample
        _prevTimestampUs = captureTimestampUs;

        var predictions = frameDetections.Select(ToPrediction).ToList();
        var classRoles  = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };

        // Epoch-relative DateTime — stable regardless of wall-clock drift
        DateTime frameTime = _sessionEpochUtc +
            TimeSpan.FromTicks((long)(captureTimestampUs * (TimeSpan.TicksPerSecond / 1_000_000.0)));

        var tracks = _tm.Update(predictions, classRoles, frameTime);

        var seen   = new HashSet<int>(tracks.Count);
        var result = new List<TrackState>(tracks.Count);

        foreach (var t in tracks)
        {
            seen.Add(t.TrackId);
            if (!_trackFirstFrameId.ContainsKey(t.TrackId))
                _trackFirstFrameId[t.TrackId] = frameId;

            int age = (int)(frameId - _trackFirstFrameId[t.TrackId] + 1);
            result.Add(ToTrackState(t, age, frameDeltaSeconds));
        }

        foreach (var id in _trackFirstFrameId.Keys.Except(seen).ToList())
            _trackFirstFrameId.Remove(id);

        _last = result;
        return _last;
    }

    public void Reset()
    {
        _tm = BuildTrackManager();
        _trackFirstFrameId.Clear();
        _prevTimestampUs = -1;
        _last = [];
    }

    // -------------------------------------------------------------------------
    // PoseDetection → Prediction
    // -------------------------------------------------------------------------

    private Prediction ToPrediction(PoseDetection d)
    {
        var box    = d.BoundingBox;
        var rect   = new RectangleF(box.Left, box.Top, box.Width, box.Height);
        var screen = new RectangleF(
            _screenLeft + box.Left,
            _screenTop  + box.Top,
            box.Width,
            box.Height);

        return new Prediction
        {
            Rectangle       = rect,
            ScreenRectangle = screen,
            Confidence      = d.ObjectConfidence,
            ClassId         = 0,
            ClassName       = "enemy",
            ScreenCenterX   = screen.Left + screen.Width  / 2f,
            ScreenCenterY   = screen.Top  + screen.Height / 2f,
            Keypoints       = ToLegacyKeypoints(d),
        };
    }

    private PlayerKeypoints? ToLegacyKeypoints(PoseDetection d)
    {
        if (d.Keypoints.Count == 0) return null;
        return new PlayerKeypoints
        {
            Head  = ToLegacyKeypoint(d.Head),
            Neck  = ToLegacyKeypoint(d.Neck),
            Chest = ToLegacyKeypoint(d.UpperChest),
            Hip   = ToLegacyKeypoint(d.Hip),
        };
    }

    private Keypoint ToLegacyKeypoint(PoseKeypoint? kp)
    {
        if (kp == null) return Keypoint.Empty;
        return new Keypoint
        {
            X          = _screenLeft + kp.Value.X,
            Y          = _screenTop  + kp.Value.Y,
            Visibility = kp.Value.Visibility switch
            {
                KingAim.Core.Perception.KeypointVisibility.Visible  => 0.90f,
                KingAim.Core.Perception.KeypointVisibility.Occluded => 0.35f,
                _                                                    => 0.00f,
            },
        };
    }

    // -------------------------------------------------------------------------
    // Track → TrackState
    // -------------------------------------------------------------------------

    private TrackState ToTrackState(Track t, int age, double frameDeltaSeconds)
    {
        // Box in desktop-pixel space → source-frame pixels, clamped to frame bounds
        var box = t.BoundingBox;
        var srcBox = new DetectionBoundingBox(
            Math.Max(0f,           box.Left   - _screenLeft),
            Math.Max(0f,           box.Top    - _screenTop),
            Math.Min(_screenWidth,  box.Right  - _screenLeft),
            Math.Min(_screenHeight, box.Bottom - _screenTop));

        // Velocity: TrackManager stores normalized-per-second; convert to pixels/frame
        float vx = t.Velocity.X * _screenWidth  * (float)frameDeltaSeconds;
        float vy = t.Velocity.Y * _screenHeight * (float)frameDeltaSeconds;

        float stability = ComputeStability(t);

        return new TrackState
        {
            TrackId             = t.TrackId,
            Box                 = srcBox,
            Keypoints           = ToCorePoseKeypoints(t.Keypoints),
            DetectionConfidence = t.Confidence,
            Age                 = age,
            VisibleFrames       = t.ObservationCount,
            MissingFrames       = t.FramesSinceLastSeen,
            VelocityX           = vx,
            VelocityY           = vy,
            StabilityScore      = stability,
        };
    }

    private static float ComputeStability(Track t)
    {
        if (t.FramesSinceLastSeen > 0) return 0f;
        return Math.Min(1f, t.ObservationCount / 15f);
    }

    private static IReadOnlyList<PoseKeypoint> ToCorePoseKeypoints(PlayerKeypoints? kps)
    {
        if (kps == null) return [];
        return
        [
            ToCoreKeypoint(KeypointName.Head,       kps.Head),
            ToCoreKeypoint(KeypointName.Neck,       kps.Neck),
            ToCoreKeypoint(KeypointName.UpperChest, kps.Chest),
            ToCoreKeypoint(KeypointName.Hip,        kps.Hip),
        ];
    }

    private static PoseKeypoint ToCoreKeypoint(KeypointName name, Keypoint kp) =>
        new(name, kp.X, kp.Y, kp.Visibility,
            kp.Visibility >= 0.5f  ? KingAim.Core.Perception.KeypointVisibility.Visible  :
            kp.Visibility >= 0.25f ? KingAim.Core.Perception.KeypointVisibility.Occluded :
                                     KingAim.Core.Perception.KeypointVisibility.Absent);

    private TrackManager BuildTrackManager() => new()
    {
        ScreenWidth  = _screenWidth,
        ScreenHeight = _screenHeight,
        ScreenLeft   = _screenLeft,
        ScreenTop    = _screenTop,
    };
}
