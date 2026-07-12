using KingAim.Core.Perception;

namespace KingAim.Core.Tracking;

/// <summary>
/// IoU tracker with a per-track Kalman filter (constant-velocity model).
/// Each frame: predict all tracks → match predicted boxes against detections →
/// correct matched filters → use filtered velocity in TrackState.
/// </summary>
public sealed class KalmanTrackerService : ITrackerService
{
    private readonly int   _maxMissingFrames;
    private readonly float _iouThreshold;

    private readonly List<TrackEntry> _tracks = [];
    private readonly List<TrackState> _result = [];
    private int  _nextId          = 1;
    private long _lastTimestampUs = 0;

    public KalmanTrackerService(int maxMissingFrames = 5, float iouThreshold = 0.25f)
    {
        _maxMissingFrames = maxMissingFrames;
        _iouThreshold     = iouThreshold;
    }

    public IReadOnlyList<TrackState> ActiveTracks => _result;

    public IReadOnlyList<TrackState> Update(
        IReadOnlyList<PoseDetection> frameDetections,
        long frameId,
        long captureTimestampUs)
    {
        double dtSeconds = _lastTimestampUs == 0
            ? 1.0 / 30.0
            : Math.Clamp((captureTimestampUs - _lastTimestampUs) / 1_000_000.0, 1.0 / 240.0, 0.5);
        _lastTimestampUs = captureTimestampUs;

        // 1. Predict all tracks and store the predicted boxes for matching
        var predicted = new DetectionBoundingBox[_tracks.Count];
        for (int i = 0; i < _tracks.Count; i++)
            predicted[i] = _tracks[i].Filter.Predict(dtSeconds);

        // 2. Greedy IoU matching: tracks vs predicted boxes
        var trackMatched = new bool[_tracks.Count];
        var detMatched   = new bool[frameDetections.Count];

        for (int ti = 0; ti < _tracks.Count; ti++)
        {
            int   bestDet = -1;
            float bestIou = _iouThreshold;

            for (int di = 0; di < frameDetections.Count; di++)
            {
                if (detMatched[di]) continue;
                float iou = Iou(predicted[ti], frameDetections[di].BoundingBox);
                if (iou > bestIou) { bestIou = iou; bestDet = di; }
            }

            if (bestDet >= 0)
            {
                trackMatched[ti]    = true;
                detMatched[bestDet] = true;
                _tracks[ti].Filter.Correct(frameDetections[bestDet].BoundingBox);

                var e = _tracks[ti];
                e.LastConfidence = frameDetections[bestDet].ObjectConfidence;
                e.LastKeypoints  = frameDetections[bestDet].Keypoints;
                e.Age++;
                e.VisibleFrames++;
                e.MissingFrames = 0;
            }
        }

        // 3. Increment missing counter for unmatched tracks; cap covariance
        for (int ti = 0; ti < _tracks.Count; ti++)
        {
            if (!trackMatched[ti])
            {
                _tracks[ti].MissingFrames++;
                _tracks[ti].Filter.CapCovariance();
            }
        }

        // 4. Expire tracks that have been missing too long
        _tracks.RemoveAll(e => e.MissingFrames > _maxMissingFrames);

        // 5. Spawn new tracks for unmatched detections
        for (int di = 0; di < frameDetections.Count; di++)
        {
            if (detMatched[di]) continue;
            var filter = new KalmanTrackFilter();
            filter.Initialize(frameDetections[di].BoundingBox);
            _tracks.Add(new TrackEntry
            {
                TrackId        = _nextId++,
                Filter         = filter,
                LastConfidence = frameDetections[di].ObjectConfidence,
                LastKeypoints  = frameDetections[di].Keypoints,
                Age            = 1,
                VisibleFrames  = 1,
                MissingFrames  = 0,
            });
        }

        // 6. Emit TrackState list from current filter state
        _result.Clear();
        foreach (var e in _tracks)
        {
            float stability = MathF.Min(1f, e.VisibleFrames / 15f);
            if (e.MissingFrames > 0) stability *= MathF.Pow(0.8f, e.MissingFrames);

            _result.Add(new TrackState
            {
                TrackId             = e.TrackId,
                Box                 = DetectionBoundingBox.FromXywh(e.Filter.Cx, e.Filter.Cy, e.Filter.W, e.Filter.H),
                Keypoints           = e.LastKeypoints,
                DetectionConfidence = e.LastConfidence * (e.MissingFrames > 0 ? MathF.Pow(0.85f, e.MissingFrames) : 1f),
                Age                 = e.Age,
                VisibleFrames       = e.VisibleFrames,
                MissingFrames       = e.MissingFrames,
                VelocityX           = e.Filter.Vx,
                VelocityY           = e.Filter.Vy,
                StabilityScore      = stability,
            });
        }

        return _result;
    }

    public void Reset()
    {
        _tracks.Clear();
        _result.Clear();
        _nextId          = 1;
        _lastTimestampUs = 0;
    }

    private static float Iou(DetectionBoundingBox a, DetectionBoundingBox b)
    {
        float ix1 = MathF.Max(a.Left,   b.Left);
        float iy1 = MathF.Max(a.Top,    b.Top);
        float ix2 = MathF.Min(a.Right,  b.Right);
        float iy2 = MathF.Min(a.Bottom, b.Bottom);
        float inter = MathF.Max(0, ix2 - ix1) * MathF.Max(0, iy2 - iy1);
        float union = a.Area + b.Area - inter;
        return union > 0 ? inter / union : 0f;
    }

    private sealed class TrackEntry
    {
        public required int   TrackId;
        public required KalmanTrackFilter Filter;
        public required float LastConfidence;
        public required IReadOnlyList<PoseKeypoint> LastKeypoints;
        public required int   Age;
        public required int   VisibleFrames;
        public          int   MissingFrames;
    }
}
