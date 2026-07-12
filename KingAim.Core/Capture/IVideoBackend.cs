namespace KingAim.Core.Capture;

/// <summary>
/// Low-level video reader abstraction. Implementations may wrap OpenCvSharp,
/// MediaFoundation, or a synthetic fake for testing.
/// </summary>
public interface IVideoBackend : IDisposable
{
    bool   IsOpen    { get; }
    int    Width     { get; }
    int    Height    { get; }
    double FrameRate { get; }
    int    FrameCount { get; }   // -1 if unknown

    /// <summary>Open the video source. Returns false if the path is unreadable.</summary>
    bool Open(string path);

    /// <summary>
    /// Decode the next frame into a BGR24 byte array and return its presentation
    /// timestamp in microseconds. Returns false when the stream ends.
    /// </summary>
    bool TryReadFrame(out byte[] pixels, out long presentationUs);

    /// <summary>Seek to the given zero-based frame index.</summary>
    bool Seek(int frameIndex);

    void Close();
}
