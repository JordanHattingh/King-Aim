using KingAim.Core.Capture;

namespace KingAim.Core.Scheduling;

/// <summary>
/// Single-slot frame mailbox. A newly posted frame overwrites any unconsumed one,
/// ensuring the pipeline always works on the most recent data.
/// </summary>
public sealed class FrameScheduler
{
    private CapturedFrame? _slot;
    private readonly object _lock = new();

    public long FramesCaptured  { get; private set; }
    public long FramesConsumed  { get; private set; }
    public long FramesDropped   { get; private set; }

    /// <summary>
    /// Stores a new frame. If a previous frame was waiting it is silently dropped.
    /// </summary>
    public void Post(CapturedFrame frame)
    {
        lock (_lock)
        {
            if (_slot != null)
                FramesDropped++;
            _slot = frame;
            FramesCaptured++;
        }
    }

    /// <summary>
    /// Takes the waiting frame (if any) and returns it. Returns false when empty.
    /// </summary>
    public bool TryTake(out CapturedFrame? frame)
    {
        lock (_lock)
        {
            frame = _slot;
            _slot = null;
            if (frame != null)
            {
                FramesConsumed++;
                return true;
            }
            return false;
        }
    }

    public void Reset()
    {
        lock (_lock)
        {
            _slot = null;
        }
    }
}
