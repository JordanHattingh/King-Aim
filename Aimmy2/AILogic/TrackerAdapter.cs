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
    private readonly float _assumedFps;

    private readonly Dictionary<int, long> _trackFirstFrameId = [];
    private IReadOnlyList<TrackState> _last = [];

    public TrackerAdapter(
        int screenWidth,
        int screenHeight,
        int screenLeft  = 0,
        int screenTop   = 0,
        float assumedFps = 30f)
    {
        if (screenWidth  <= 0) throw new ArgumentOutOfRangeException(nameof(screenWidth));
        if (screenHeight <= 0) throw new ArgumentOutOfRangeException(nameof(screenHeight));
        _screenWidth  = screenWidth;
        _screenHeight = screenHeight;
        _screenLeft   = screenLeft;
        _screenTop    = screenTop;
        _assumedFps   = assumedFps > 0 ? assumedFps : 30f;
        _tm = BuildTrackManager();
    }

    public IReadOnlyList<TrackState> ActiveTracks => _last;

    public IReadOnlyList<TrackState> Update(
        IReadOnlyList<PoseDetection> frameDetections,
        long frameId,
        long captureTimestampUs)
    {
        var predictions = frameDetections
            .Select(d => ToPrediction(d))
            .ToList();

        var classRoles = new Dictionary<int, SemanticRole> { [0] = SemanticRole.Enemy };
        var frameTime  = DateTimeOffset.FromUnixTimeMilliseconds(captureTimestampUs / 1000L)
                                       .UtcDateTime;

        var tracks = _tm.Update(predictions, classRoles, frameTime);

        var seen   = new HashSet<int>(tracks.Count);
        var result = new List<TrackState>(tracks.Count);

        foreach (var t in tracks)
        {
            seen.Add(t.TrackId);
            if (!_trackFirstFrameId.ContainsKey(t.TrackId))
                _trackFirstFrameId[t.TrackId] = frameId;

            int age = (int)(frameId - _trackFirstFrameId[t.TrackId] + 1);
            result.Add(ToTrackState(t, age));
        }

        // Clean up entries for expired tracks
        foreach (var id in _trackFirstFrameId.Keys.Except(seen).ToList())
            _trackFirstFrameId.Remove(id);

        _last = result;
        return _last;
    }

    public void Reset()
    {
        _tm = BuildTrackManager();
        _trackFirstFrameId.Clear();
        _last = [];
    }

    // -------------------------------------------------------------------------
    // PoseDetection → Prediction
    // -------------------------------------------------------------------------

    private Prediction ToPrediction(PoseDetection d)
    {
        var box    = d.BoundingBox;
        // Source-frame pixel space (capture local, 0,0 = top-left of capture region)
        var rect   = new RectangleF(box.Left, box.Top, box.Width, box.Height);
        // Desktop-pixel space (offset by screen origin)
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

    private TrackState ToTrackState(Track t, int age)
    {
        // Track.BoundingBox is in desktop-pixel space; convert back to source-frame pixels.
        var box    = t.BoundingBox;
        var srcBox = new DetectionBoundingBox(
            box.Left   - _screenLeft,
            box.Top    - _screenTop,
            box.Right  - _screenLeft,
            box.Bottom - _screenTop);

        float stabilityScore = ComputeStability(t);

        return new TrackState
        {
            TrackId             = t.TrackId,
            Box                 = srcBox,
            Keypoints           = ToCorePoseKeypoints(t.Keypoints),
            DetectionConfidence = t.Confidence,
            Age                 = age,
            VisibleFrames       = t.ObservationCount,
            MissingFrames       = t.FramesSinceLastSeen,
            VelocityX           = t.Velocity.X * _screenWidth  / _assumedFps,
            VelocityY           = t.Velocity.Y * _screenHeight / _assumedFps,
            StabilityScore      = stabilityScore,
        };
    }

    private static float ComputeStability(Track t)
    {
        if (t.FramesSinceLastSeen > 0) return 0f;
        float obs = Math.Min(1f, t.ObservationCount / 15f);
        return obs;
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
            kp.Visibility >= 0.5f ? KingAim.Core.Perception.KeypointVisibility.Visible  :
            kp.Visibility >= 0.25f ? KingAim.Core.Perception.KeypointVisibility.Occluded :
                                     KingAim.Core.Perception.KeypointVisibility.Absent);

    // -------------------------------------------------------------------------

    private TrackManager BuildTrackManager() => new()
    {
        ScreenWidth  = _screenWidth,
        ScreenHeight = _screenHeight,
        ScreenLeft   = _screenLeft,
        ScreenTop    = _screenTop,
    };
}
