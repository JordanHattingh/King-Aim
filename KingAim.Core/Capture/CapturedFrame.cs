namespace KingAim.Core.Capture;

/// <summary>
/// A single frame produced by a frame source.
/// The pixel data is owned by this record; callers must dispose when done.
/// </summary>
public sealed class CapturedFrame : IDisposable
{
    public long   FrameId           { get; init; }
    public long   CaptureTimestampUs { get; init; }  // microseconds since epoch (UTC)
    public int    SourceWidth        { get; init; }
    public int    SourceHeight       { get; init; }
    public string PixelFormat        { get; init; } = "BGR24";
    public string SourceId           { get; init; } = "";

    /// <summary>Raw pixel bytes. Layout described by PixelFormat.</summary>
    public byte[] PixelData { get; init; } = [];

    public void Dispose()
    {
        // Pixel data is a plain array; nothing to unmanage.
        // Dispose exists so consumers can use using() without caring about the implementation.
        GC.SuppressFinalize(this);
    }
}
