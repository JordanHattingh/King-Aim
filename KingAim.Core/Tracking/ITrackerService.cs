using KingAim.Core.Perception;

namespace KingAim.Core.Tracking;

/// <summary>
/// Adapter that places the existing frozen tracker behind a stable interface.
/// The tracker regression suite (97/97 tests) must remain green at all times.
/// </summary>
public interface ITrackerService
{
    /// <summary>
    /// Updates tracker state with the detections from one frame.
    /// Returns the full set of currently active tracks.
    /// </summary>
    IReadOnlyList<TrackState> Update(
        IReadOnlyList<PoseDetection> frameDetections,
        long frameId,
        long captureTimestampUs);

    /// <summary>All currently active tracks (same as last Update result).</summary>
    IReadOnlyList<TrackState> ActiveTracks { get; }

    /// <summary>Clears all tracks. Use on scene change or source reset.</summary>
    void Reset();
}
