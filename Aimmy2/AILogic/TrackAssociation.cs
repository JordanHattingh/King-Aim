using AILogic;
using System.Drawing;

namespace Aimmy2.AILogic
{
    public readonly record struct TrackDetectionMatch(Track Track, int DetectionIndex, float Cost);

    public sealed class TrackAssociationSettings
    {
        public float IoUWeight { get; set; } = 0.45f;
        public float CenterDistanceWeight { get; set; } = 0.30f;
        public float MotionErrorWeight { get; set; } = 0.25f;

        /// <summary>
        /// Blend amount between the timestamped Kalman motion prior and the previous GRU centre hint.
        /// Zero is Kalman-only; one is GRU-only. Ignored when the track has no GRU prediction.
        /// </summary>
        public float GruAssociationWeight { get; set; } = 0.6f;

        /// <summary>
        /// Optional pose-geometry contribution. Disabled by default until a pose bundle opts in.
        /// </summary>
        public float PoseSimilarityWeight { get; set; } = 0.0f;

        public float MaximumAcceptedCost { get; set; } = 0.75f;
        public float UnmatchedCost { get; set; } = 1.25f;
        public float MaximumCenterDistanceScreenFraction { get; set; } = 0.15f;
    }

    public interface ITrackAssociationStrategy
    {
        IReadOnlyList<TrackDetectionMatch> Associate(
            IReadOnlyList<Track> tracks,
            IReadOnlyList<Prediction> detections,
            DateTime observationTime,
            int screenWidth,
            int screenHeight,
            TrackAssociationSettings settings);
    }

    /// <summary>
    /// Global minimum-cost assignment using a padded Hungarian matrix. Cost combines overlap,
    /// screen-normalized center distance, timestamped Kalman/GRU motion error, and optional
    /// translation/scale-normalized pose geometry.
    /// </summary>
    public sealed class GlobalCostTrackAssociation : ITrackAssociationStrategy
    {
        public IReadOnlyList<TrackDetectionMatch> Associate(
            IReadOnlyList<Track> tracks,
            IReadOnlyList<Prediction> detections,
            DateTime observationTime,
            int screenWidth,
            int screenHeight,
            TrackAssociationSettings settings)
        {
            if (tracks.Count == 0 || detections.Count == 0)
                return Array.Empty<TrackDetectionMatch>();

            int size = Math.Max(tracks.Count, detections.Count);
            double[,] cost = new double[size, size];
            for (int row = 0; row < size; row++)
            {
                for (int column = 0; column < size; column++)
                    cost[row, column] = settings.UnmatchedCost;
            }

            for (int trackIndex = 0; trackIndex < tracks.Count; trackIndex++)
            {
                Track track = tracks[trackIndex];
                for (int detectionIndex = 0; detectionIndex < detections.Count; detectionIndex++)
                {
                    cost[trackIndex, detectionIndex] = ComputeCost(
                        track,
                        detections[detectionIndex],
                        observationTime,
                        screenWidth,
                        screenHeight,
                        settings);
                }
            }

            int[] assignment = HungarianSolver.Solve(cost);
            var matches = new List<TrackDetectionMatch>();
            for (int trackIndex = 0; trackIndex < tracks.Count; trackIndex++)
            {
                int detectionIndex = assignment[trackIndex];
                if (detectionIndex < 0 || detectionIndex >= detections.Count)
                    continue;

                float pairCost = (float)cost[trackIndex, detectionIndex];
                if (pairCost <= settings.MaximumAcceptedCost)
                    matches.Add(new TrackDetectionMatch(tracks[trackIndex], detectionIndex, pairCost));
            }
            return matches;
        }

        private static double ComputeCost(
            Track track,
            Prediction detection,
            DateTime observationTime,
            int screenWidth,
            int screenHeight,
            TrackAssociationSettings settings)
        {
            if (track.ClassId != detection.ClassId)
                return settings.UnmatchedCost;

            RectangleF detectionBox = GetScreenRectangle(detection);
            PointF trackCenter = Center(track.BoundingBox);
            PointF detectionCenter = Center(detectionBox);
            double diagonal = Math.Max(
                1.0,
                Math.Sqrt((double)screenWidth * screenWidth + (double)screenHeight * screenHeight));

            double centerDistance = Distance(trackCenter, detectionCenter) / diagonal;
            if (centerDistance > settings.MaximumCenterDistanceScreenFraction
                && ComputeIoU(track.BoundingBox, detectionBox) <= 0)
            {
                return settings.UnmatchedCost;
            }

            KalmanPrediction.Detection kalman = track.Kalman.GetKalmanPosition(
                mouseSpeed: 0,
                timestamp: observationTime,
                applyLead: false);
            PointF kalmanCenter = new(kalman.X, kalman.Y);
            PointF predictedCenter = kalmanCenter;

            if (track.GruPredictedCenter is { } gruCenter)
            {
                float gruWeight = Math.Clamp(settings.GruAssociationWeight, 0f, 1f);
                predictedCenter = new PointF(
                    kalmanCenter.X + (gruCenter.X - kalmanCenter.X) * gruWeight,
                    kalmanCenter.Y + (gruCenter.Y - kalmanCenter.Y) * gruWeight);
            }

            double motionError = Distance(predictedCenter, detectionCenter) / diagonal;
            double iouCost = 1.0 - ComputeIoU(track.BoundingBox, detectionBox);

            double baseWeightSum = Math.Max(
                1e-6,
                settings.IoUWeight + settings.CenterDistanceWeight + settings.MotionErrorWeight);
            double cost = (
                settings.IoUWeight * iouCost
                + settings.CenterDistanceWeight * centerDistance
                + settings.MotionErrorWeight * motionError) / baseWeightSum;

            double? poseSimilarity = ComputePoseSimilarity(
                track.Keypoints,
                track.BoundingBox,
                detection.Keypoints,
                detectionBox);
            float poseWeight = Math.Max(0f, settings.PoseSimilarityWeight);
            if (poseWeight > 0f && poseSimilarity.HasValue)
                cost += poseWeight * (1.0 - poseSimilarity.Value);

            return cost;
        }

        private static double? ComputePoseSimilarity(
            PlayerKeypoints? trackPose,
            RectangleF trackBox,
            PlayerKeypoints? detectionPose,
            RectangleF detectionBox)
        {
            if (trackPose == null || detectionPose == null
                || trackBox.Width <= 0 || trackBox.Height <= 0
                || detectionBox.Width <= 0 || detectionBox.Height <= 0)
            {
                return null;
            }

            var trackPoints = new List<PointF>(PlayerKeypoints.KeypointCount);
            var detectionPoints = new List<PointF>(PlayerKeypoints.KeypointCount);
            for (int i = 0; i < PlayerKeypoints.KeypointCount; i++)
            {
                Keypoint a = trackPose[i];
                Keypoint b = detectionPose[i];
                if (!a.IsUsable || !b.IsUsable)
                    continue;

                trackPoints.Add(new PointF(
                    (a.X - trackBox.Left) / trackBox.Width,
                    (a.Y - trackBox.Top) / trackBox.Height));
                detectionPoints.Add(new PointF(
                    (b.X - detectionBox.Left) / detectionBox.Width,
                    (b.Y - detectionBox.Top) / detectionBox.Height));
            }

            if (trackPoints.Count < 2)
                return null;

            PointF trackCentroid = Centroid(trackPoints);
            PointF detectionCentroid = Centroid(detectionPoints);
            double dot = 0;
            double trackNorm = 0;
            double detectionNorm = 0;

            for (int i = 0; i < trackPoints.Count; i++)
            {
                double ax = trackPoints[i].X - trackCentroid.X;
                double ay = trackPoints[i].Y - trackCentroid.Y;
                double bx = detectionPoints[i].X - detectionCentroid.X;
                double by = detectionPoints[i].Y - detectionCentroid.Y;

                dot += ax * bx + ay * by;
                trackNorm += ax * ax + ay * ay;
                detectionNorm += bx * bx + by * by;
            }

            if (trackNorm <= 1e-9 || detectionNorm <= 1e-9)
                return null;

            double cosine = dot / Math.Sqrt(trackNorm * detectionNorm);
            return Math.Clamp((cosine + 1.0) * 0.5, 0.0, 1.0);
        }

        private static PointF Centroid(IReadOnlyList<PointF> points)
        {
            float x = 0;
            float y = 0;
            foreach (PointF point in points)
            {
                x += point.X;
                y += point.Y;
            }

            return new PointF(x / points.Count, y / points.Count);
        }

        private static RectangleF GetScreenRectangle(Prediction prediction) =>
            prediction.ScreenRectangle.Width > 0 && prediction.ScreenRectangle.Height > 0
                ? prediction.ScreenRectangle
                : prediction.Rectangle;

        private static PointF Center(RectangleF box) =>
            new(box.X + box.Width / 2f, box.Y + box.Height / 2f);

        private static double Distance(PointF a, PointF b)
        {
            double dx = a.X - b.X;
            double dy = a.Y - b.Y;
            return Math.Sqrt(dx * dx + dy * dy);
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

    internal static class HungarianSolver
    {
        /// <summary>Returns assigned column per row for a square cost matrix.</summary>
        public static int[] Solve(double[,] cost)
        {
            int n = cost.GetLength(0);
            if (n != cost.GetLength(1))
                throw new ArgumentException("Hungarian solver expects a square matrix.", nameof(cost));

            // 1-indexed implementation of the O(n^3) shortest augmenting path algorithm.
            var u = new double[n + 1];
            var v = new double[n + 1];
            var p = new int[n + 1];
            var way = new int[n + 1];

            for (int i = 1; i <= n; i++)
            {
                p[0] = i;
                int j0 = 0;
                var minv = Enumerable.Repeat(double.PositiveInfinity, n + 1).ToArray();
                var used = new bool[n + 1];

                do
                {
                    used[j0] = true;
                    int i0 = p[j0];
                    double delta = double.PositiveInfinity;
                    int j1 = 0;
                    for (int j = 1; j <= n; j++)
                    {
                        if (used[j])
                            continue;
                        double current = cost[i0 - 1, j - 1] - u[i0] - v[j];
                        if (current < minv[j])
                        {
                            minv[j] = current;
                            way[j] = j0;
                        }
                        if (minv[j] < delta)
                        {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                    for (int j = 0; j <= n; j++)
                    {
                        if (used[j])
                        {
                            u[p[j]] += delta;
                            v[j] -= delta;
                        }
                        else
                        {
                            minv[j] -= delta;
                        }
                    }
                    j0 = j1;
                }
                while (p[j0] != 0);

                do
                {
                    int j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                }
                while (j0 != 0);
            }

            var assignment = Enumerable.Repeat(-1, n).ToArray();
            for (int j = 1; j <= n; j++)
            {
                if (p[j] > 0)
                    assignment[p[j] - 1] = j - 1;
            }
            return assignment;
        }
    }
}
