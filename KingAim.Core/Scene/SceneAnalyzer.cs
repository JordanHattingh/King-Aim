using KingAim.Core.Tracking;

namespace KingAim.Core.Scene;

public sealed class SceneAnalyzer : ISceneAnalyzer
{
    public SceneState Analyze(
        IReadOnlyList<TrackState> tracks,
        long frameId,
        int sourceWidth,
        int sourceHeight)
    {
        if (tracks.Count == 0)
            return new SceneState { FrameId = frameId };

        float cx = sourceWidth  / 2f;
        float cy = sourceHeight / 2f;

        TrackState? nearestCentre = null;
        float       nearestDist   = float.MaxValue;
        TrackState? highestConf   = null;
        float       bestConf      = -1f;

        foreach (var t in tracks)
        {
            float dist = MathF.Sqrt(
                MathF.Pow(t.Box.CentreX - cx, 2) +
                MathF.Pow(t.Box.CentreY - cy, 2));

            if (dist < nearestDist) { nearestDist = dist; nearestCentre = t; }
            if (t.DetectionConfidence > bestConf) { bestConf = t.DetectionConfidence; highestConf = t; }
        }

        float sceneStability = tracks.Count > 0
            ? tracks.Average(t => t.StabilityScore)
            : 0f;

        return new SceneState
        {
            FrameId                  = frameId,
            ActiveTracks             = tracks,
            PrimaryTrack             = nearestCentre,
            VisibleEnemyCount        = tracks.Count,
            NearestCentreTrack       = nearestCentre,
            HighestConfidenceTrack   = highestConf,
            SceneStability           = sceneStability,
        };
    }
}
