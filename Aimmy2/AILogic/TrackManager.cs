using System.Drawing;

namespace Aimmy2.AILogic
{
    public sealed class Track
    {
        public int TrackId { get; internal set; }
        public SemanticRole Role { get; internal set; }
        public int ClassId { get; internal set; }
        public string ClassName { get; internal set; } = "";
        public RectangleF BoundingBox { get; internal set; }
        public float Confidence { get; internal set; }
        public DateTime LastSeen { get; internal set; }
        public int FramesSinceLastSeen { get; internal set; }
        public PointF Velocity { get; internal set; }
        public int ObservationCount { get; internal set; }

        internal PointF Center => new(
            BoundingBox.X + BoundingBox.Width / 2f,
            BoundingBox.Y + BoundingBox.Height / 2f);
    }

    public sealed class TrackManager
    {
        private const float IoUWeight = 0.6f;
        private const float DistanceWeight = 0.4f;
        private const float MinMatchScore = 0.1f;

        private readonly List<Track> _tracks = new();
        private int _nextTrackId = 1;

        public int MaxFramesLost { get; set; } = 5;

        public IReadOnlyList<Track> ActiveTracks => _tracks;

        public int PlayerCount => _tracks.Count(t => t.Role == SemanticRole.Player);
        public int EnemyCount => _tracks.Count(t => t.Role == SemanticRole.Enemy);
        public int FriendlyCount => _tracks.Count(t => t.Role == SemanticRole.Friendly);

        public IReadOnlyList<Track> Update(
            IReadOnlyList<Prediction> detections,
            Dictionary<int, SemanticRole> classRoles,
            DateTime frameTime)
        {
            ArgumentNullException.ThrowIfNull(detections);
            ArgumentNullException.ThrowIfNull(classRoles);

            foreach (var track in _tracks)
            {
                double dt = Math.Max(0, (frameTime - track.LastSeen).TotalSeconds);
                if (dt > 0 && dt < 1.0)
                {
                    track.BoundingBox = new RectangleF(
                        track.BoundingBox.X + track.Velocity.X,
                        track.BoundingBox.Y + track.Velocity.Y,
                        track.BoundingBox.Width,
                        track.BoundingBox.Height);
                }
            }

            var unmatchedDetections = new List<int>(Enumerable.Range(0, detections.Count));
            var matchedTracks = new HashSet<Track>();

            var candidatePairs = new List<(Track Track, int DetectionIndex, float Score)>();
            foreach (var track in _tracks)
            {
                for (int i = 0; i < detections.Count; i++)
                {
                    var detection = detections[i];
                    SemanticRole detectionRole = classRoles.GetValueOrDefault(detection.ClassId, SemanticRole.Enemy);

                    if (detection.ClassId != track.ClassId)
                        continue;

                    float iou = ComputeIoU(track.BoundingBox, detection.Rectangle);
                    float centerDistance = Distance(TrackCenter(track.BoundingBox), DetectionCenter(detection));
                    float maxDim = Math.Max(track.BoundingBox.Width, track.BoundingBox.Height);
                    float normalizedDistance = maxDim > 0 ? Math.Min(1f, centerDistance / (maxDim * 3f)) : 1f;

                    float score = IoUWeight * iou + DistanceWeight * (1f - normalizedDistance);
                    if (score >= MinMatchScore)
                    {
                        candidatePairs.Add((track, i, score));
                    }
                }
            }

            foreach (var pair in candidatePairs.OrderByDescending(p => p.Score))
            {
                if (matchedTracks.Contains(pair.Track) || !unmatchedDetections.Contains(pair.DetectionIndex))
                    continue;

                var detection = detections[pair.DetectionIndex];
                UpdateTrackWithDetection(pair.Track, detection, frameTime);
                matchedTracks.Add(pair.Track);
                unmatchedDetections.Remove(pair.DetectionIndex);
            }

            foreach (var track in _tracks)
            {
                if (!matchedTracks.Contains(track))
                {
                    track.FramesSinceLastSeen++;
                }
            }

            foreach (int detectionIndex in unmatchedDetections)
            {
                var detection = detections[detectionIndex];
                SemanticRole role = classRoles.GetValueOrDefault(detection.ClassId, SemanticRole.Enemy);
                var newTrack = new Track
                {
                    TrackId = _nextTrackId++,
                    Role = role,
                    ClassId = detection.ClassId,
                    ClassName = detection.ClassName,
                    BoundingBox = detection.Rectangle,
                    Confidence = detection.Confidence,
                    LastSeen = frameTime,
                    FramesSinceLastSeen = 0,
                    Velocity = PointF.Empty,
                    ObservationCount = 1,
                };
                _tracks.Add(newTrack);
            }

            _tracks.RemoveAll(t => t.FramesSinceLastSeen > MaxFramesLost);

            return _tracks.ToList();
        }

        private static void UpdateTrackWithDetection(Track track, Prediction detection, DateTime frameTime)
        {
            PointF oldCenter = TrackCenter(track.BoundingBox);
            PointF newCenter = DetectionCenter(detection);

            float newVelX = newCenter.X - oldCenter.X;
            float newVelY = newCenter.Y - oldCenter.Y;

            const float smoothing = 0.7f;
            track.Velocity = new PointF(
                track.Velocity.X * smoothing + newVelX * (1 - smoothing),
                track.Velocity.Y * smoothing + newVelY * (1 - smoothing));

            track.BoundingBox = detection.Rectangle;
            track.Confidence = detection.Confidence;
            track.ClassName = detection.ClassName;
            track.LastSeen = frameTime;
            track.FramesSinceLastSeen = 0;
            track.ObservationCount++;
        }

        private static PointF TrackCenter(RectangleF box) =>
            new(box.X + box.Width / 2f, box.Y + box.Height / 2f);

        private static PointF DetectionCenter(Prediction p) =>
            new(p.Rectangle.X + p.Rectangle.Width / 2f, p.Rectangle.Y + p.Rectangle.Height / 2f);

        private static float Distance(PointF a, PointF b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return MathF.Sqrt(dx * dx + dy * dy);
        }

        private static float ComputeIoU(RectangleF a, RectangleF b)
        {
            float x1 = Math.Max(a.Left, b.Left);
            float y1 = Math.Max(a.Top, b.Top);
            float x2 = Math.Min(a.Right, b.Right);
            float y2 = Math.Min(a.Bottom, b.Bottom);

            if (x2 <= x1 || y2 <= y1)
                return 0f;

            float intersection = (x2 - x1) * (y2 - y1);
            float union = a.Width * a.Height + b.Width * b.Height - intersection;
            return union > 0 ? intersection / union : 0f;
        }
    }
}
