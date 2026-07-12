using KingAim.Core.Scene;
using KingAim.Core.Tracking;

namespace KingAim.Core.Focus;

public sealed class FocusSelector : IFocusSelector
{
    private readonly FocusWeights _weights;
    private TrackState? _current;

    public FocusSelector(FocusWeights? weights = null)
    {
        _weights = weights ?? FocusWeights.Default;
    }

    public TrackState? CurrentFocus => _current;

    public TrackState? SelectFocus(SceneState scene, int sourceWidth, int sourceHeight)
    {
        if (scene.ActiveTracks.Count == 0)
        {
            _current = null;
            return null;
        }

        float cx = sourceWidth  / 2f;
        float cy = sourceHeight / 2f;
        float maxDist = MathF.Sqrt(cx * cx + cy * cy);

        TrackState? best     = null;
        float       bestScore = -1f;

        foreach (var t in scene.ActiveTracks)
        {
            float dist = MathF.Sqrt(
                MathF.Pow(t.Box.CentreX - cx, 2) +
                MathF.Pow(t.Box.CentreY - cy, 2));

            float centreScore   = maxDist > 0 ? 1f - dist / maxDist : 1f;
            float poseScore     = t.Keypoints.Count > 0
                ? t.Keypoints.Average(kp => kp.Confidence)
                : 0f;
            float persistScore  = Math.Min(1f, t.VisibleFrames / 30f);

            float score = _weights.Confidence     * t.DetectionConfidence
                        + _weights.Stability      * t.StabilityScore
                        + _weights.CentreProximity * centreScore
                        + _weights.PoseQuality    * poseScore
                        + _weights.Persistence    * persistScore;

            // Stickiness bonus for current focus target.
            if (_current != null && t.TrackId == _current.TrackId)
                score += _weights.StickinessThreshold;

            if (score > bestScore) { bestScore = score; best = t; }
        }

        _current = best;
        return best;
    }
}
