using KingAim.Core.Capture;

namespace KingAim.Core.Scheduling;

/// <summary>
/// Decouples the capture rate from the inference rate.
/// Maintains a latest-frame slot: stale frames are dropped rather than queued.
/// </summary>
public interface IFrameScheduler : IDisposable
{
    /// <summary>Post a captured frame. If a previous frame is waiting, it is dropped.</summary>
    void Post(CapturedFrame frame);

    /// <summary>Take the latest frame. Returns false if none is available.</summary>
    bool TryTake(out CapturedFrame? frame);

    long FramesCaptured  { get; }
    long FramesConsumed  { get; }
    long FramesDropped   { get; }
    double AverageFrameAgeMs { get; }
    double MaximumFrameAgeMs { get; }
}
