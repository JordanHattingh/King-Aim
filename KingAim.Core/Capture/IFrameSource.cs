namespace KingAim.Core.Capture;

/// <summary>
/// Common interface for all frame producers.
/// The pipeline depends on this, not on any concrete capture method.
/// </summary>
public interface IFrameSource : IDisposable
{
    FrameSourceType SourceType { get; }
    int    Width    { get; }
    int    Height   { get; }
    double FrameRate { get; }
    bool   IsRunning { get; }

    /// <summary>
    /// Starts producing frames. Safe to call multiple times (idempotent).
    /// </summary>
    Task StartAsync(CancellationToken ct = default);

    /// <summary>Stops producing frames. Waits for the current frame to complete.</summary>
    Task StopAsync(CancellationToken ct = default);

    /// <summary>
    /// Returns the latest available frame, or null if none is ready.
    /// The caller is responsible for disposing the returned frame.
    /// </summary>
    CapturedFrame? GetLatestFrame();
}
