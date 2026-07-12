using KingAim.Core.Tracking;

namespace KingAim.Core.Scene;

/// <summary>
/// Converts a set of active tracks into a coherent SceneState.
/// </summary>
public interface ISceneAnalyzer
{
    SceneState Analyze(
        IReadOnlyList<TrackState> tracks,
        long frameId,
        int sourceWidth,
        int sourceHeight);
}
