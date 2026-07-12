namespace KingAim.Core.Capture.Sources;

/// <summary>
/// Replays a recorded video file at its native frame rate.
/// Provides deterministic input for pipeline testing and accessibility development.
/// </summary>
public sealed class RecordedVideoSource : IFrameSource
{
    private readonly string _videoPath;
    private bool   _running;
    private bool   _disposed;

    public RecordedVideoSource(string videoPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(videoPath);
        _videoPath = videoPath;
    }

    public FrameSourceType SourceType => FrameSourceType.RecordedVideo;
    public int    Width     { get; private set; }
    public int    Height    { get; private set; }
    public double FrameRate { get; private set; }
    public bool   IsRunning => _running;

    public Task StartAsync(CancellationToken ct = default)
    {
        ThrowIfDisposed();
        _running = true;
        // Concrete implementation: open video with OpenCvSharp or MediaFoundation.
        // Scaffold only — finalize when capture layer is implemented.
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken ct = default)
    {
        _running = false;
        return Task.CompletedTask;
    }

    public CapturedFrame? GetLatestFrame()
    {
        if (!_running)
            return null;

        // Scaffold: returns null until concrete video reader is wired.
        return null;
    }

    public void Dispose()
    {
        _running  = false;
        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(RecordedVideoSource));
    }
}
