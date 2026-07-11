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

        /// <summary>Typed absolute desktop center for the current (possibly extrapolated) box.</summary>
        public DesktopPixel CenterDesktop { get; internal set; }

        /// <summary>Typed display-normalized center for the current (possibly extrapolated) box.</summary>
        public ScreenFraction CenterFraction { get; internal set; }

        public int ObservationCount { get; internal set; }

        internal DateTime LastBufferTimestamp { get; set; }
        internal KalmanPrediction Kalman { get; } = new();
        internal TrackRingBuffer RingBuffer { get; } = new();
        internal int KeypointFrameAge { get; set; }
        internal RectangleF LastObservedBoundingBox { get; set; }
        internal PointF LastVelocitySampleCenter { get; set; }
        internal DateTime LastVelocitySampleTimestamp { get; set; }
        internal bool HasVelocitySample { get; set; }

        public PlayerKeypoints? Keypoints { get; internal set; }
        public (float X, float Y)? GruPredictedCenter { get; internal set; }
        public bool BoundingBoxIsExtrapolated { get; internal set; }

        internal PointF Center => new(CenterDesktop.X, CenterDesktop.Y);
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

        /// <summary>Number of consecutive non-pose frames that may reuse the last pose sample.</summary>
        public int KeypointCarryForwardFrames { get; set; } = 3;

        /// <summary>When enabled, temporarily extrapolate a lost track's box before freezing it.</summary>
        public bool UsePredictiveBoundingBox { get; set; } = true;

        /// <summary>First-order velocity filter response rate in 1/seconds.</summary>
        public float VelocityResponseRatePerSecond { get; set; } = 12f;
        public double MinimumVelocitySampleSeconds { get; set; } = 1.0 / 120.0;
        public float RawVelocityMeasurementWeight { get; set; } = 0.9f;
        public long VelocityUpdatesAccepted { get; private set; }
        public long VelocityUpdatesSkippedSubFrame { get; private set; }
        public double MinimumObservedUpdateIntervalMs { get; private set; } = double.PositiveInfinity;
        public float MaximumAcceptedVelocityMagnitude { get; private set; }

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
            CoordinateContractGuards.ValidateDimensions(ScreenWidth, ScreenHeight);

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

            foreach (Track track in _tracks)
            {
                if (matchedTracks.Contains(track))
                    continue;

                track.FramesSinceLastSeen++;
                AgeKeypointsWithoutPose(track);

                double missingAgeSeconds = Math.Max(0, (frameTime - track.LastSeen).TotalSeconds);
                track.BoundingBoxIsExtrapolated = false;

                if (UsePredictiveBoundingBox)
                    ExtrapolateLostBoundingBox(track, missingAgeSeconds);

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
                Prediction detection = detections[detectionIndex];
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
                    LastObservedBoundingBox = detectionBox,
                    Confidence = detection.Confidence,
                    FirstSeen = frameTime,
                    LastSeen = frameTime,
                    LastBufferTimestamp = frameTime,
                    FramesSinceLastSeen = 0,
                    Velocity = PointF.Empty,
                    ObservationCount = 1,
                    Keypoints = detection.Keypoints,
                    KeypointFrameAge = detection.Keypoints == null ? Math.Max(0, KeypointCarryForwardFrames) : 0,
                    BoundingBoxIsExtrapolated = false,
                    LastVelocitySampleCenter = center,
                    LastVelocitySampleTimestamp = frameTime,
                    HasVelocitySample = true,
                };

                newTrack.Kalman.UpdateKalmanFilter(new KalmanPrediction.Detection
                {
                    X = (int)center.X,
                    Y = (int)center.Y,
                    Timestamp = frameTime,
                });

                newTrack.RingBuffer.Push(CreateObservedSample(newTrack, detection, center, dtSeconds: 0f));
                RefreshCoordinateContracts(newTrack);
                _tracks.Add(newTrack);
            }

            _tracks.RemoveAll(t => (frameTime - t.LastSeen).TotalSeconds > MaxLostSeconds);

            foreach (Track track in _tracks)
                RefreshCoordinateContracts(track);

            return _tracks.ToList();
        }

        private void UpdateTrackWithDetection(Track track, Prediction detection, DateTime frameTime)
        {
            RectangleF detectionBox = GetScreenRectangle(detection);
            PointF newCenter = TrackCenter(detectionBox);
            float dt = Math.Clamp((float)(frameTime - track.LastSeen).TotalSeconds, 0f, 0.1f);

            track.Kalman.UpdateKalmanFilter(new KalmanPrediction.Detection
            {
                X = (int)newCenter.X,
                Y = (int)newCenter.Y,
                Timestamp = frameTime,
            });

            KalmanPrediction.Detection smoothedDetection = track.Kalman.GetKalmanPosition(
                mouseSpeed: 0,
                timestamp: frameTime,
                applyLead: false);
            PointF smoothedCenter = new(smoothedDetection.X, smoothedDetection.Y);
            float rawWeight = Math.Clamp(RawVelocityMeasurementWeight, 0f, 1f);
            PointF velocitySampleCenter = new(
                smoothedCenter.X + (newCenter.X - smoothedCenter.X) * rawWeight,
                smoothedCenter.Y + (newCenter.Y - smoothedCenter.Y) * rawWeight);

            double velocityDt = track.HasVelocitySample
                ? (frameTime - track.LastVelocitySampleTimestamp).TotalSeconds
                : 0.0;

            if (track.HasVelocitySample && velocityDt >= 0.0)
                MinimumObservedUpdateIntervalMs = Math.Min(MinimumObservedUpdateIntervalMs, velocityDt * 1000.0);

            if (!track.HasVelocitySample)
            {
                track.LastVelocitySampleCenter = velocitySampleCenter;
                track.LastVelocitySampleTimestamp = frameTime;
                track.HasVelocitySample = true;
            }
            else if (double.IsFinite(velocityDt) && velocityDt >= MinimumVelocitySampleSeconds)
            {
                float stableDt = (float)Math.Min(velocityDt, 0.1);
                float sw = Math.Max(ScreenWidth, 1);
                float sh = Math.Max(ScreenHeight, 1);
                float measuredVx = ((velocitySampleCenter.X - track.LastVelocitySampleCenter.X) / sw) / stableDt;
                float measuredVy = ((velocitySampleCenter.Y - track.LastVelocitySampleCenter.Y) / sh) / stableDt;

                if (!float.IsFinite(measuredVx) || !float.IsFinite(measuredVy))
                {
                    measuredVx = track.Velocity.X;
                    measuredVy = track.Velocity.Y;
                }

                float alpha = 1f - MathF.Exp(-Math.Max(VelocityResponseRatePerSecond, 0f) * stableDt);

                track.Velocity = new PointF(
                    track.Velocity.X + (measuredVx - track.Velocity.X) * alpha,
                    track.Velocity.Y + (measuredVy - track.Velocity.Y) * alpha);

                if (!float.IsFinite(track.Velocity.X) || !float.IsFinite(track.Velocity.Y))
                    track.Velocity = PointF.Empty;

                track.LastVelocitySampleCenter = velocitySampleCenter;
                track.LastVelocitySampleTimestamp = frameTime;
                VelocityUpdatesAccepted++;
                MaximumAcceptedVelocityMagnitude = Math.Max(MaximumAcceptedVelocityMagnitude,
                    MathF.Sqrt(track.Velocity.X * track.Velocity.X + track.Velocity.Y * track.Velocity.Y));
            }
            else
            {
                VelocityUpdatesSkippedSubFrame++;
            }

            float halfW = detectionBox.Width / 2f;
            float halfH = detectionBox.Height / 2f;
            track.BoundingBox = new RectangleF(
                smoothedCenter.X - halfW,
                smoothedCenter.Y - halfH,
                detectionBox.Width,
                detectionBox.Height);
            track.LastObservedBoundingBox = track.BoundingBox;
            track.BoundingBoxIsExtrapolated = false;

            track.Confidence = detection.Confidence;
            track.ClassName = detection.ClassName;
            UpdateKeypoints(track, detection.Keypoints);
            track.LastSeen = frameTime;
            track.FramesSinceLastSeen = 0;
            track.ObservationCount++;

            track.RingBuffer.Push(CreateObservedSample(track, detection, smoothedCenter, dt));
            track.LastBufferTimestamp = frameTime;
            RefreshCoordinateContracts(track);
        }

        private void UpdateKeypoints(Track track, PlayerKeypoints? incoming)
        {
            if (incoming != null)
            {
                track.Keypoints = incoming;
                track.KeypointFrameAge = 0;
                return;
            }

            AgeKeypointsWithoutPose(track);
        }

        private void AgeKeypointsWithoutPose(Track track)
        {
            int carryLimit = Math.Max(0, KeypointCarryForwardFrames);
            if (track.Keypoints != null && track.KeypointFrameAge < carryLimit)
            {
                track.KeypointFrameAge++;
                return;
            }

            track.Keypoints = null;
            track.KeypointFrameAge = carryLimit;
        }

        private void ExtrapolateLostBoundingBox(Track track, double missingAgeSeconds)
        {
            double extrapolationLimit = Math.Max(0, MaxLostSeconds / 2.0);
            if (missingAgeSeconds <= 0 || missingAgeSeconds > extrapolationLimit)
                return;

            RectangleF observedBox = track.LastObservedBoundingBox.Width > 0 && track.LastObservedBoundingBox.Height > 0
                ? track.LastObservedBoundingBox
                : track.BoundingBox;
            PointF observedCenter = TrackCenter(observedBox);
            float predictedX;
            float predictedY;

            if (track.GruPredictedCenter is { } gruCenter)
            {
                float gruDeltaX = gruCenter.X - observedCenter.X;
                float gruDeltaY = gruCenter.Y - observedCenter.Y;
                float referenceDt = track.RingBuffer.Count > 0
                    ? Math.Clamp(track.RingBuffer.Tail.DtSeconds, 0.001f, 0.1f)
                    : 1f / 60f;
                float stepCount = Math.Max(1f, (float)missingAgeSeconds / referenceDt);
                predictedX = observedCenter.X + gruDeltaX * stepCount;
                predictedY = observedCenter.Y + gruDeltaY * stepCount;
            }
            else
            {
                predictedX = observedCenter.X
                    + track.Velocity.X * Math.Max(ScreenWidth, 1) * (float)missingAgeSeconds;
                predictedY = observedCenter.Y
                    + track.Velocity.Y * Math.Max(ScreenHeight, 1) * (float)missingAgeSeconds;
            }

            track.BoundingBox = new RectangleF(
                predictedX - observedBox.Width / 2f,
                predictedY - observedBox.Height / 2f,
                observedBox.Width,
                observedBox.Height);
            track.BoundingBoxIsExtrapolated = true;
        }

        private TrackObservation CreateObservedSample(
            Track track,
            Prediction detection,
            PointF center,
            float dtSeconds)
        {
            ScreenFraction centerFraction = new DesktopPixel(center.X, center.Y)
                .ToScreenFraction(ScreenLeft, ScreenTop, ScreenWidth, ScreenHeight);

            return new TrackObservation
            {
                CxNorm = Math.Clamp(centerFraction.X, 0f, 1f),
                CyNorm = Math.Clamp(centerFraction.Y, 0f, 1f),
                WNorm = Math.Clamp(track.BoundingBox.Width / Math.Max(ScreenWidth, 1), 0f, 1f),
                HNorm = Math.Clamp(track.BoundingBox.Height / Math.Max(ScreenHeight, 1), 0f, 1f),
                Confidence = detection.Confidence,
                ObservedMask = 1f,
                DtSeconds = dtSeconds,
                AgeSeconds = 0f,
            };
        }

        private void RefreshCoordinateContracts(Track track)
        {
            PointF center = TrackCenter(track.BoundingBox);
            track.CenterDesktop = new DesktopPixel(center.X, center.Y);
            track.CenterFraction = track.CenterDesktop.ToScreenFraction(
                ScreenLeft,
                ScreenTop,
                ScreenWidth,
                ScreenHeight);
        }

        private static RectangleF GetScreenRectangle(Prediction prediction) =>
            prediction.ScreenRectangle.Width > 0 && prediction.ScreenRectangle.Height > 0
                ? prediction.ScreenRectangle
                : prediction.Rectangle;

        private static PointF TrackCenter(RectangleF box) =>
            new(box.X + box.Width / 2f, box.Y + box.Height / 2f);
    }
}
