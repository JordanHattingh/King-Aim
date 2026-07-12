using KingAim.Core.Perception;

namespace KingAim.Core.Tracking;

/// <summary>
/// Lightweight IoU-based tracker used by the synthetic integration pipeline and tests.
/// Assigns stable IDs across frames; removes tracks that have been missing for too long.
/// </summary>
public sealed class SimpleTrackerService : ITrackerService
{
    private readonly int _maxMissingFrames;
    private readonly float _iouThreshold;
    private readonly List<TrackState> _tracks = [];
    private int _nextId = 1;

    public SimpleTrackerService(int maxMissingFrames = 5, float iouThreshold = 0.3f)
    {
        _maxMissingFrames = maxMissingFrames;
        _iouThreshold     = iouThreshold;
    }

    public IReadOnlyList<TrackState> ActiveTracks => _tracks;

    public IReadOnlyList<TrackState> Update(
        IReadOnlyList<PoseDetection> frameDetections,
        long frameId,
        long captureTimestampUs)
    {
        var unmatched = new List<PoseDetection>(frameDetections);
        var updated   = new List<TrackState>();

        foreach (var track in _tracks)
        {
            // Try to find the best-matching detection
            int    bestIdx  = -1;
            float  bestIou  = _iouThreshold;
            for (int i = 0; i < unmatched.Count; i++)
            {
                float iou = Iou(track.Box, unmatched[i].BoundingBox);
                if (iou > bestIou) { bestIou = iou; bestIdx = i; }
            }

            if (bestIdx >= 0)
            {
                var det = unmatched[bestIdx];
                unmatched.RemoveAt(bestIdx);

                float stability = Math.Min(1f, (track.VisibleFrames + 1) / 15f);
                updated.Add(new TrackState
                {
                    TrackId             = track.TrackId,
                    Box                 = det.BoundingBox,
                    Keypoints           = det.Keypoints,
                    DetectionConfidence = det.ObjectConfidence,
                    Age                 = track.Age + 1,
                    VisibleFrames       = track.VisibleFrames + 1,
                    MissingFrames       = 0,
                    VelocityX           = det.BoundingBox.CentreX - track.Box.CentreX,
                    VelocityY           = det.BoundingBox.CentreY - track.Box.CentreY,
                    StabilityScore      = stability,
                });
            }
            else
            {
                int missing = track.MissingFrames + 1;
                if (missing <= _maxMissingFrames)
                {
                    updated.Add(new TrackState
                    {
                        TrackId             = track.TrackId,
                        Box                 = track.Box,
                        Keypoints           = track.Keypoints,
                        DetectionConfidence = track.DetectionConfidence * 0.85f,
                        Age                 = track.Age + 1,
                        VisibleFrames       = track.VisibleFrames,
                        MissingFrames       = missing,
                        StabilityScore      = track.StabilityScore * 0.8f,
                    });
                }
                // else: track expires
            }
        }

        // Unmatched detections spawn new tracks
        foreach (var det in unmatched)
        {
            updated.Add(new TrackState
            {
                TrackId             = _nextId++,
                Box                 = det.BoundingBox,
                Keypoints           = det.Keypoints,
                DetectionConfidence = det.ObjectConfidence,
                Age                 = 1,
                VisibleFrames       = 1,
                MissingFrames       = 0,
                StabilityScore      = 0f,
            });
        }

        _tracks.Clear();
        _tracks.AddRange(updated);
        return _tracks;
    }

    public void Reset()
    {
        _tracks.Clear();
        _nextId = 1;
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
}
