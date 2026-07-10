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
        public RectangleF BoundingBox { get; internal set; }
        public float Confidence { get; internal set; }
        public DateTime FirstSeen { get; internal set; }
        public DateTime LastSeen { get; internal set; }
        public int FramesSinceLastSeen { get; internal set; }
        public PointF Velocity { get; internal set; }
        public int ObservationCount { get; internal set; }

        // Per-track Kalman filter: fallback when GRU buffer is not yet full.
        internal KalmanPrediction Kalman { get; } = new();

        // 8-frame ring buffer fed to the GRU temporal predictor.
        internal TrackRingBuffer RingBuffer { get; } = new();

        /// <summary>
        /// Most recent keypoints from the pose model. Null for plain detection models.
        /// </summary>
        public PlayerKeypoints? Keypoints { get; internal set; }

        /// <summary>
        /// GRU-predicted next position (screen pixels). Null until buffer has 8 frames.
        /// Used by AIManager instead of Kalman for the locked target.
        /// </summary>
        public (float X, float Y)? GruPredictedCenter { get; internal set; }

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

        // Screen dimensions needed to normalize ring buffer observations to [0,1].
        public int ScreenWidth  { get; set; } = 1920;
        public int ScreenHeight { get; set; } = 1080;

        public int MaxFramesLost { get; set; } = 5;

        /// <summary>
        /// Wall-clock time a track may go unmatched before being dropped. This is the actual
        /// expiry rule (not MaxFramesLost) so tracking survives short detection gaps consistently
        /// regardless of the AI loop's real FPS — a frame-count-only rule loses tracks almost
        /// instantly when FPS is low (e.g. 5 missed frames at 5 FPS is a full second gone).
        /// </summary>
        public double MaxLostSeconds { get; set; } = 0.6;

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
                if (dt > 0 && dt < MaxLostSeconds)
                {
                    // Use Kalman extrapolation for lost-track prediction: the filter accounts for
                    // elapsed time and velocity uncertainty, giving a better estimate than raw pixel velocity.
                    var predicted = track.Kalman.GetKalmanPosition(mouseSpeed: 0);
                    float halfW = track.BoundingBox.Width / 2f;
                    float halfH = track.BoundingBox.Height / 2f;
                    track.BoundingBox = new RectangleF(
                        predicted.X - halfW, predicted.Y - halfH,
                        track.BoundingBox.Width, track.BoundingBox.Height);
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
                    // Push a carry-forward Missing observation so the GRU buffer advances
                    // during detection gaps and doesn't stall on the last-seen frame.
                    if (track.RingBuffer.Count > 0)
                    {
                        float missDt = Math.Clamp((float)(frameTime - track.LastSeen).TotalSeconds, 0f, 0.1f);
                        track.RingBuffer.Push(TrackObservation.Missing(track.RingBuffer.Tail, missDt));
                    }
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
                    FirstSeen = frameTime,
                    LastSeen = frameTime,
                    FramesSinceLastSeen = 0,
                    Velocity = PointF.Empty,
                    ObservationCount = 1,
                };
                _tracks.Add(newTrack);
            }

            _tracks.RemoveAll(t => (frameTime - t.LastSeen).TotalSeconds > MaxLostSeconds);

            return _tracks.ToList();
        }

        private void UpdateTrackWithDetection(Track track, Prediction detection, DateTime frameTime)
        {
            PointF newCenter = DetectionCenter(detection);

            // Feed the raw detection into the per-track Kalman filter.
            track.Kalman.UpdateKalmanFilter(new KalmanPrediction.Detection
            {
                X = (int)newCenter.X,
                Y = (int)newCenter.Y,
                Timestamp = frameTime,
            });

            // Ask Kalman for its smoothed current estimate (lead=0).
            // This gives a noise-filtered position rather than the raw detection jitter.
            var smoothed = track.Kalman.GetKalmanPosition(mouseSpeed: 0);

            // Derive velocity from the Kalman state (smoothed estimate vs previous smoothed position).
            // Use the old bounding-box centre as the "previous" for velocity calculation.
            PointF oldCenter = TrackCenter(track.BoundingBox);
            float velX = smoothed.X - oldCenter.X;
            float velY = smoothed.Y - oldCenter.Y;

            // Light EMA on top of Kalman velocity to damp sudden spikes (Kalman velocity can
            // jump on first-observation frames when the filter is not yet converged).
            const float velSmoothing = 0.5f;
            track.Velocity = new PointF(
                track.Velocity.X * velSmoothing + velX * (1f - velSmoothing),
                track.Velocity.Y * velSmoothing + velY * (1f - velSmoothing));

            // Use the Kalman-smoothed centre to reposition the bounding box, keeping its size unchanged.
            float halfW = detection.Rectangle.Width / 2f;
            float halfH = detection.Rectangle.Height / 2f;
            track.BoundingBox = new RectangleF(smoothed.X - halfW, smoothed.Y - halfH,
                detection.Rectangle.Width, detection.Rectangle.Height);

            // Capture timing BEFORE updating LastSeen so DtSeconds is the real inter-frame gap.
            float dt  = Math.Clamp((float)(frameTime - track.LastSeen).TotalSeconds, 0f, 0.1f);
            float age = Math.Clamp((float)(frameTime - track.FirstSeen).TotalSeconds, 0f, 0.25f);

            track.Confidence = detection.Confidence;
            track.ClassName = detection.ClassName;
            track.Keypoints = detection.Keypoints;
            track.LastSeen = frameTime;
            track.FramesSinceLastSeen = 0;
            track.ObservationCount++;

            // Push real observation into the GRU ring buffer.
            float sw = Math.Max(ScreenWidth,  1);
            float sh = Math.Max(ScreenHeight, 1);
            var center = new PointF(
                smoothed.X / sw,
                smoothed.Y / sh);
            track.RingBuffer.Push(new TrackObservation
            {
                CxNorm       = center.X,
                CyNorm       = center.Y,
                WNorm        = detection.Rectangle.Width  / sw,
                HNorm        = detection.Rectangle.Height / sh,
                Confidence   = detection.Confidence,
                ObservedMask = 1f,
                DtSeconds    = dt,
                AgeSeconds   = age,
            });
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
