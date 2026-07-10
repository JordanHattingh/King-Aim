using System.Drawing;

namespace Aimmy2.AILogic
{
    public enum ObservationPointSource
    {
        Neck,
        Chest,
        Head,
        Hip,
        BoxFallback
    }

    public readonly record struct ResolvedObservationPoint(
        PointF Point,
        ObservationPointSource Source,
        float Confidence);

    public static class ObservationPointResolver
    {
        public static ResolvedObservationPoint Resolve(Track track, float boxFallbackFraction = 0.25f)
        {
            ArgumentNullException.ThrowIfNull(track);

            PlayerKeypoints? keypoints = track.Keypoints;
            if (keypoints != null)
            {
                if (keypoints.Neck.IsUsable)
                    return FromKeypoint(keypoints.Neck, ObservationPointSource.Neck);
                if (keypoints.Chest.IsUsable)
                    return FromKeypoint(keypoints.Chest, ObservationPointSource.Chest);
                if (keypoints.Head.IsUsable)
                    return FromKeypoint(keypoints.Head, ObservationPointSource.Head);
                if (keypoints.Hip.IsUsable)
                    return FromKeypoint(keypoints.Hip, ObservationPointSource.Hip);
            }

            float fraction = Math.Clamp(boxFallbackFraction, 0f, 1f);
            return new ResolvedObservationPoint(
                new PointF(
                    track.BoundingBox.Left + track.BoundingBox.Width * 0.5f,
                    track.BoundingBox.Top + track.BoundingBox.Height * fraction),
                ObservationPointSource.BoxFallback,
                track.Confidence);
        }

        private static ResolvedObservationPoint FromKeypoint(Keypoint keypoint, ObservationPointSource source) =>
            new(new PointF(keypoint.X, keypoint.Y), source, keypoint.Visibility);
    }
}
