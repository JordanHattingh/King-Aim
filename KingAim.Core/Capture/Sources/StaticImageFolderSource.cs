namespace KingAim.Core.Capture.Sources;

/// <summary>
/// Iterates over image files in a directory, yielding each as a frame.
/// Used for model evaluation and reproducible testing.
/// </summary>
public sealed class StaticImageFolderSource : IFrameSource
{
    private static readonly string[] SupportedExtensions = [".png", ".jpg", ".jpeg", ".bmp"];

    private readonly string _folderPath;
    private readonly IReadOnlyList<string> _files;
    private int  _index;
    private bool _running;
    private bool _disposed;

    public StaticImageFolderSource(string folderPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folderPath);
        _folderPath = folderPath;
        _files = Directory.Exists(folderPath)
            ? Directory.GetFiles(folderPath)
                       .Where(f => SupportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                       .OrderBy(f => f)
                       .ToList()
            : [];
    }

    public FrameSourceType SourceType => FrameSourceType.StaticImageFolder;
    public int    Width     { get; private set; } = 1920;
    public int    Height    { get; private set; } = 1080;
    public double FrameRate => 1.0;
    public bool   IsRunning => _running;

    public int  TotalFrames   => _files.Count;
    public int  CurrentIndex  => _index;
    public bool IsExhausted   => _index >= _files.Count;

    public Task StartAsync(CancellationToken ct = default)
    {
        ThrowIfDisposed();
        _index   = 0;
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
        if (!_running || _index >= _files.Count)
            return null;

        var path = _files[_index++];
        // Concrete implementation: load pixel data with OpenCvSharp or similar.
        // Scaffold only — returns a thin stub with the path encoded in SourceId.
        return new CapturedFrame
        {
            FrameId            = _index - 1,
            CaptureTimestampUs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L,
            SourceWidth        = Width,
            SourceHeight       = Height,
            PixelFormat        = "BGR24",
            SourceId           = path,
            PixelData          = [],
        };
    }

    public void Dispose()
    {
        _running  = false;
        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(StaticImageFolderSource));
    }
}
