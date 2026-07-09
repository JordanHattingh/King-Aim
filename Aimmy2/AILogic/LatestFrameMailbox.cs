using System.Drawing;

namespace Aimmy2.AILogic
{
    public sealed record CapturedFrame(
        long FrameId,
        DateTime CaptureStartedAt,
        DateTime CaptureCompletedAt,
        int Width,
        int Height,
        Rectangle CaptureRegion,
        Bitmap Image,
        float CaptureToModelScale = 1f);

    public sealed class LatestFrameMailbox : IDisposable
    {
        private readonly object _lock = new();
        private CapturedFrame? _slot;
        private bool _disposed;

        private long _framesCaptured;
        private long _framesConsumed;
        private long _framesReplaced;
        private double _totalFrameAgeMs;
        private double _maximumFrameAgeMs;

        public long FramesCaptured => Interlocked.Read(ref _framesCaptured);
        public long FramesConsumed => Interlocked.Read(ref _framesConsumed);
        public long FramesReplaced => Interlocked.Read(ref _framesReplaced);

        public double AverageFrameAge
        {
            get
            {
                lock (_lock)
                {
                    return _framesConsumed > 0 ? _totalFrameAgeMs / _framesConsumed : 0;
                }
            }
        }

        public double MaximumFrameAge
        {
            get
            {
                lock (_lock)
                {
                    return _maximumFrameAgeMs;
                }
            }
        }

        public void Post(CapturedFrame frame)
        {
            ArgumentNullException.ThrowIfNull(frame);

            CapturedFrame? replaced = null;

            lock (_lock)
            {
                if (_disposed)
                {
                    frame.Image.Dispose();
                    return;
                }

                if (_slot != null)
                {
                    replaced = _slot;
                    Interlocked.Increment(ref _framesReplaced);
                }

                _slot = frame;
                Interlocked.Increment(ref _framesCaptured);
            }

            replaced?.Image.Dispose();
        }

        public bool TryTake(out CapturedFrame? frame)
        {
            lock (_lock)
            {
                if (_slot == null)
                {
                    frame = null;
                    return false;
                }

                frame = _slot;
                _slot = null;
            }

            double ageMs = (DateTime.UtcNow - frame.CaptureCompletedAt).TotalMilliseconds;
            lock (_lock)
            {
                _totalFrameAgeMs += ageMs;
                _maximumFrameAgeMs = Math.Max(_maximumFrameAgeMs, ageMs);
            }
            Interlocked.Increment(ref _framesConsumed);

            return true;
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed)
                    return;

                _disposed = true;
                _slot?.Image.Dispose();
                _slot = null;
            }
        }
    }
}
