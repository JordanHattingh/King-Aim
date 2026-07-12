namespace KingAim.Core.Capture.Sources;

/// <summary>
/// Produces deterministic synthetic frames for pipeline integration tests.
/// No real game pixels. No pointing assistance available from this source.
/// </summary>
public sealed class SyntheticPatternSource : IFrameSource
{
    private readonly int    _width;
    private readonly int    _height;
    private readonly double _frameRate;
    private long _nextFrameId;
    private bool _running;
    private bool _disposed;

    public SyntheticPatternSource(int width = 1920, int height = 1080, double frameRate = 30.0)
    {
        _width     = width;
        _height    = height;
        _frameRate = frameRate;
    }

    public FrameSourceType SourceType => FrameSourceType.SyntheticTestPattern;
    public int    Width     => _width;
    public int    Height    => _height;
    public double FrameRate => _frameRate;
    public bool   IsRunning => _running;

    public Task StartAsync(CancellationToken ct = default)
    {
        ThrowIfDisposed();
        _running = true;
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken ct = default)
    {
        _running = false;
        return Task.CompletedTask;
    }

    public CapturedFrame? GetLatestFrame()
    {
        if (!_running) return null;

        long id      = Interlocked.Increment(ref _nextFrameId);
        int  stride  = _width * 3;
        var  pixels  = new byte[stride * _height];

        // Checkerboard pattern for visual identification in tests.
        for (int y = 0; y < _height; y++)
        for (int x = 0; x < _width;  x++)
        {
            byte v = ((x / 32 + y / 32) % 2 == 0) ? (byte)200 : (byte)50;
            int  i = y * stride + x * 3;
            pixels[i] = pixels[i + 1] = pixels[i + 2] = v;
        }

        return new CapturedFrame
        {
            FrameId            = id,
            CaptureTimestampUs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L,
            SourceWidth        = _width,
            SourceHeight       = _height,
            PixelFormat        = "BGR24",
            SourceId           = "synthetic",
            PixelData          = pixels,
        };
    }

    public void Dispose()
    {
        _running  = false;
        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(SyntheticPatternSource));
    }
}
