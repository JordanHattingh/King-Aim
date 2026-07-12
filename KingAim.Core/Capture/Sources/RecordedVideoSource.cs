namespace KingAim.Core.Capture.Sources;

/// <summary>Controls how the source paces frame delivery.</summary>
public enum PlaybackTimingMode
{
    /// <summary>Sleeps between frames to match the video's native frame rate.</summary>
    RealTime,

    /// <summary>Delivers frames as fast as the consumer calls GetLatestFrame().</summary>
    AsFastAsPossible,

    /// <summary>Delivers at a fixed caller-supplied frame rate regardless of the video's.</summary>
    FixedFrameRate,

    /// <summary>
    /// Advances exactly one frame per <see cref="RecordedVideoSource.AdvanceFrame"/> call.
    /// Ideal for deterministic integration tests.
    /// </summary>
    ManualStep,
}

/// <summary>
/// Replays a recorded video file for pipeline testing and accessibility development.
/// Injects a video backend so OpenCvSharp or any other reader can be swapped in.
/// </summary>
public sealed class RecordedVideoSource : IFrameSource, IDisposable
{
    private readonly IVideoBackend _backend;
    private readonly PlaybackTimingMode _timing;
    private readonly double _fixedFps;

    private bool  _running;
    private bool  _disposed;
    private long  _nextFrameId;
    private bool  _pendingAdvance;      // for ManualStep
    private byte[]? _currentPixels;
    private long  _currentPresentationUs;
    private long  _lastDeliveryUs;
    private CapturedFrame? _cachedFrame; // returned until next AdvanceFrame()

    public RecordedVideoSource(
        IVideoBackend backend,
        PlaybackTimingMode timing = PlaybackTimingMode.AsFastAsPossible,
        double fixedFps = 30.0)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _backend  = backend;
        _timing   = timing;
        _fixedFps = fixedFps > 0 ? fixedFps : 30.0;
    }

    public FrameSourceType SourceType => FrameSourceType.RecordedVideo;
    public int    Width     => _backend.Width;
    public int    Height    => _backend.Height;
    public double FrameRate => _timing == PlaybackTimingMode.FixedFrameRate ? _fixedFps : _backend.FrameRate;
    public bool   IsRunning => _running;
    public int    FrameCount => _backend.FrameCount;

    public Task StartAsync(CancellationToken ct = default)
    {
        ThrowIfDisposed();
        if (!_backend.IsOpen)
            throw new InvalidOperationException("Video backend is not open. Call IVideoBackend.Open() first.");
        _running        = true;
        _pendingAdvance = false;
        _currentPixels  = null;
        _lastDeliveryUs = 0;
        _cachedFrame    = null;
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken ct = default)
    {
        _running = false;
        return Task.CompletedTask;
    }

    /// <summary>
    /// Signals the source to advance one frame on the next GetLatestFrame() call.
    /// Only meaningful in <see cref="PlaybackTimingMode.ManualStep"/>.
    /// </summary>
    public void AdvanceFrame() => _pendingAdvance = true;

    /// <summary>Seek to the given zero-based frame index and clear buffered state.</summary>
    public bool Seek(int frameIndex)
    {
        bool ok = _backend.Seek(frameIndex);
        if (ok)
        {
            _currentPixels   = null;
            _pendingAdvance  = false;
            _lastDeliveryUs  = 0;
        }
        return ok;
    }

    /// <summary>Restart playback from frame zero.</summary>
    public bool Restart() => Seek(0);

    public CapturedFrame? GetLatestFrame()
    {
        if (!_running) return null;

        switch (_timing)
        {
            case PlaybackTimingMode.ManualStep:
                if (!_pendingAdvance) return _cachedFrame;
                _pendingAdvance = false;
                if (!_backend.TryReadFrame(out var mpx, out var mts)) { _running = false; return null; }
                _currentPixels = mpx;
                _currentPresentationUs = mts;
                _cachedFrame = BuildFrame();
                return _cachedFrame;

            case PlaybackTimingMode.RealTime:
            {
                long nowUs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
                long intervalUs = (long)(1_000_000.0 / _backend.FrameRate);
                if (nowUs - _lastDeliveryUs < intervalUs) return _currentPixels == null ? null : BuildFrame();
                goto case PlaybackTimingMode.AsFastAsPossible;
            }
            case PlaybackTimingMode.FixedFrameRate:
            {
                long nowUs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
                long intervalUs = (long)(1_000_000.0 / _fixedFps);
                if (nowUs - _lastDeliveryUs < intervalUs) return _currentPixels == null ? null : BuildFrame();
                goto case PlaybackTimingMode.AsFastAsPossible;
            }
            case PlaybackTimingMode.AsFastAsPossible:
            default:
                if (!_backend.TryReadFrame(out var px, out var ts)) { _running = false; return null; }
                _currentPixels = px;
                _currentPresentationUs = ts;
                _lastDeliveryUs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
                return BuildFrame();
        }
    }

    private CapturedFrame BuildFrame()
    {
        long id = Interlocked.Increment(ref _nextFrameId);
        return new CapturedFrame
        {
            FrameId            = id,
            CaptureTimestampUs = _currentPresentationUs,
            SourceWidth        = _backend.Width,
            SourceHeight       = _backend.Height,
            PixelFormat        = "BGR24",
            SourceId           = "recorded_video",
            PixelData          = _currentPixels ?? [],
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _running  = false;
        _disposed = true;
        _backend.Dispose();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(RecordedVideoSource));
    }
}
