using AILogic;
using System.Drawing;

namespace Aimmy2.AILogic
{
    public sealed class Track
    {
        public int TrackId { get; internal set; }
        public SemanticRole Role { get; internal set; }
        public int ClassId { get; internal set; }
        public string ClassName { get; internal set; } = "";

        /// <summary>Absolute desktop-pixel bounding box.</summary>
        public RectangleF BoundingBox { get; internal set; }

        public float Confidence { get; internal set; }
        public DateTime FirstSeen { get; internal set; }
        public DateTime LastSeen { get; internal set; }
        public int FramesSinceLastSeen { get; internal set; }

        /// <summary>
        /// Velocity in normalized display units per second. X=1 means one display-width per second;
        /// Y=1 means one display-height per second. This is intentionally frame-rate independent.
        /// </summary>
        public PointF Velocity { get; internal set; }

        public int ObservationCount { get; internal set; }

        internal DateTime LastBufferTimestamp { get; set; }
        internal KalmanPrediction Kalman { get; } = new();
        internal TrackRingBuffer RingBuffer { get; } = new();

        public PlayerKeypoints? Keypoints { get; internal set; }
        public (float X, float Y)? GruPredictedCenter { get; internal set; }

        internal PointF Center => new(
            BoundingBox.X + BoundingBox.Width / 2f,
            BoundingBox.Y + BoundingBox.Height / 2f);
    }

    public sealed class TrackManager
    {
        private readonly List<Track> _tracks = new();
        private int _nextTrackId = 1;

        public int ScreenWidth { get; set; } = 1920;
        public int ScreenHeight { get; set; } = 1080;
        public int ScreenLeft { get; set; }
        public int ScreenTop { get; set; }

        [Obsolete("Track expiry is time based. Use MaxLostSeconds.")]
        public int MaxFramesLost { get; set; } = 5;

        public double MaxLostSeconds { get; set; } = 0.6;

        /// <summary>First-order velocity filter response rate in 1/seconds.</summary>
        public float VelocityResponseRatePerSecond { get; set; } = 12f;

        public ITrackAssociationStrategy AssociationStrategy { get; set; } = new GlobalCostTrackAssociation();
        public TrackAssociationSettings AssociationSettings { get; } = new();

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

            var unmatchedDetections = new HashSet<int>(Enumerable.Range(0, detections.Count));
            var matchedTracks = new HashSet<Track>();

            IReadOnlyList<TrackDetectionMatch> matches = AssociationStrategy.Associate(
                _tracks,
                detections,
                frameTime,
                ScreenWidth,
                ScreenHeight,
                AssociationSettings);

            foreach (TrackDetectionMatch match in matches)
            {
                if (matchedTracks.Contains(match.Track) || !unmatchedDetections.Contains(match.DetectionIndex))
                    continue;

                UpdateTrackWithDetection(match.Track, detections[match.DetectionIndex], frameTime);
                matchedTracks.Add(match.Track);
                unmatchedDetections.Remove(match.DetectionIndex);
            }

            foreach (var track in _tracks)
            {
                if (matchedTracks.Contains(track))
                    continue;

                track.FramesSinceLastSeen++;

                double missingAgeSeconds = Math.Max(0, (frameTime - track.LastSeen).TotalSeconds);
                if (missingAgeSeconds > 0 && missingAgeSeconds < MaxLostSeconds)
                {
                    var predicted = track.Kalman.GetKalmanPosition(mouseSpeed: 0, timestamp: frameTime, applyLead: false);
                    float halfW = track.BoundingBox.Width / 2f;
                    float halfH = track.BoundingBox.Height / 2f;
                    track.BoundingBox = new RectangleF(
                        predicted.X - halfW,
                        predicted.Y - halfH,
                        track.BoundingBox.Width,
                        track.BoundingBox.Height);
                }

                if (track.RingBuffer.Count > 0 && frameTime > track.LastBufferTimestamp)
                {
                    float missDt = Math.Clamp(
                        (float)(frameTime - track.LastBufferTimestamp).TotalSeconds,
                        0f,
                        0.1f);

                    if (missDt > 0f)
                    {
                        track.RingBuffer.Push(TrackObservation.Missing(track.RingBuffer.Tail, missDt));
                        track.LastBufferTimestamp = frameTime;
                    }
                }
            }

            foreach (int detectionIndex in unmatchedDetections)
            {
                var detection = detections[detectionIndex];
                SemanticRole role = classRoles.GetValueOrDefault(detection.ClassId, SemanticRole.Unknown);
                RectangleF detectionBox = GetScreenRectangle(detection);
                PointF center = TrackCenter(detectionBox);

                var newTrack = new Track
                {
                    TrackId = _nextTrackId++,
                    Role = role,
                    ClassId = detection.ClassId,
                    ClassName = detection.ClassName,
                    BoundingBox = detectionBox,
                    Confidence = detection.Confidence,
                    FirstSeen = frameTime,
                    LastSeen = frameTime,
                    LastBufferTimestamp = frameTime,
                    FramesSinceLastSeen = 0,
                    Velocity = PointF.Empty,
                    ObservationCount = 1,
                    Keypoints = detection.Keypoints,
                };

                newTrack.Kalman.UpdateKalmanFilter(new KalmanPrediction.Detection
                {
                    X = (int)center.X,
                    Y = (int)center.Y,
                    Timestamp = frameTime,
                });

                newTrack.RingBuffer.Push(CreateObservedSample(newTrack, detection, center, dtSeconds: 0f));
                _tracks.Add(newTrack);
            }

            _tracks.RemoveAll(t => (frameTime - t.LastSeen).TotalSeconds > MaxLostSeconds);
            return _tracks.ToList();
        }

        private void UpdateTrackWithDetection(Track track, Prediction detection, DateTime frameTime)
        {
            RectangleF detectionBox = GetScreenRectangle(detection);
            PointF newCenter = TrackCenter(detectionBox);
            PointF oldCenter = TrackCenter(track.BoundingBox);
            float dt = Math.Clamp((float)(frameTime - track.LastSeen).TotalSeconds, 0f, 0.1f);

            track.Kalman.UpdateKalmanFilter(new KalmanPrediction.Detection
            {
                X = (int)newCenter.X,
                Y = (int)newCenter.Y,
                Timestamp = frameTime,
            });

            var smoothedDetection = track.Kalman.GetKalmanPosition(mouseSpeed: 0, timestamp: frameTime, applyLead: false);
            PointF smoothedCenter = new(smoothedDetection.X, smoothedDetection.Y);

            if (dt > 1e-5f)
            {
                float sw = Math.Max(ScreenWidth, 1);
                float sh = Math.Max(ScreenHeight, 1);
                float measuredVx = ((smoothedCenter.X - oldCenter.X) / sw) / dt;
                float measuredVy = ((smoothedCenter.Y - oldCenter.Y) / sh) / dt;
                float alpha = 1f - MathF.Exp(-Math.Max(VelocityResponseRatePerSecond, 0f) * dt);

                track.Velocity = new PointF(
                    track.Velocity.X + (measuredVx - track.Velocity.X) * alpha,
                    track.Velocity.Y + (measuredVy - track.Velocity.Y) * alpha);
            }

            float halfW = detectionBox.Width / 2f;
            float halfH = detectionBox.Height / 2f;
            track.BoundingBox = new RectangleF(
                smoothedCenter.X - halfW,
                smoothedCenter.Y - halfH,
                detectionBox.Width,
                detectionBox.Height);

            track.Confidence = detection.Confidence;
            track.ClassName = detection.ClassName;
            track.Keypoints = detection.Keypoints;
            track.LastSeen = frameTime;
            track.FramesSinceLastSeen = 0;
            track.ObservationCount++;

            track.RingBuffer.Push(CreateObservedSample(track, detection, smoothedCenter, dt));
            track.LastBufferTimestamp = frameTime;
        }

        private TrackObservation CreateObservedSample(
            Track track,
            Prediction detection,
            PointF center,
            float dtSeconds)
        {
            float sw = Math.Max(ScreenWidth, 1);
            float sh = Math.Max(ScreenHeight, 1);

            return new TrackObservation
            {
                CxNorm = Math.Clamp((center.X - ScreenLeft) / sw, 0f, 1f),
                CyNorm = Math.Clamp((center.Y - ScreenTop) / sh, 0f, 1f),
                WNorm = Math.Clamp(track.BoundingBox.Width / sw, 0f, 1f),
                HNorm = Math.Clamp(track.BoundingBox.Height / sh, 0f, 1f),
                Confidence = detection.Confidence,
                ObservedMask = 1f,
                DtSeconds = dtSeconds,
                AgeSeconds = 0f,
            };
        }

        private static RectangleF GetScreenRectangle(Prediction prediction) =>
            prediction.ScreenRectangle.Width > 0 && prediction.ScreenRectangle.Height > 0
                ? prediction.ScreenRectangle
                : prediction.Rectangle;

        private static PointF TrackCenter(RectangleF box) =>
            new(box.X + box.Width / 2f, box.Y + box.Height / 2f);

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
