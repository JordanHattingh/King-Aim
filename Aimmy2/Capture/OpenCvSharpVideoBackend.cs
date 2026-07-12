using KingAim.Core.Capture;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace Aimmy2.Capture;

/// <summary>
/// IVideoBackend implementation backed by OpenCvSharp VideoCapture.
/// Always returns owned BGR24 byte arrays — callers may store them freely
/// without risk of the next Read() overwriting the buffer.
/// Timestamps are sourced from the video's presentation clock (PosMsec),
/// falling back to frame-index × FPS if the codec does not report position.
/// </summary>
public sealed class OpenCvSharpVideoBackend : IVideoBackend
{
    private VideoCapture? _capture;
    private readonly Mat  _mat = new();
    private long   _currentFrameIndex;
    private bool   _atEof;
    private bool   _disposed;

    // -------------------------------------------------------------------------
    // IVideoBackend properties
    // -------------------------------------------------------------------------

    public bool   IsOpen     => !_disposed && (_capture?.IsOpened() ?? false);
    public int    Width      => (int)(_capture?.Get(VideoCaptureProperties.FrameWidth)  ?? 0);
    public int    Height     => (int)(_capture?.Get(VideoCaptureProperties.FrameHeight) ?? 0);
    public double FrameRate  => _capture?.Get(VideoCaptureProperties.Fps)               ?? 0.0;
    public int    FrameCount => (int)(_capture?.Get(VideoCaptureProperties.FrameCount)  ?? -1);

    // -------------------------------------------------------------------------
    // IVideoBackend methods
    // -------------------------------------------------------------------------

    public bool Open(string path)
    {
        ThrowIfDisposed();
        _capture?.Release();
        _capture           = new VideoCapture(path);
        _currentFrameIndex = 0;
        _atEof             = false;
        return _capture.IsOpened();
    }

    /// <summary>
    /// Reads the next frame and copies the BGR24 pixels into a freshly allocated array.
    /// The returned array is fully owned by the caller; subsequent calls do not alias it.
    /// Timestamp priority: codec PosMsec → frame-index / FPS → synthetic 30fps fallback.
    /// </summary>
    public bool TryReadFrame(out byte[] pixels, out long presentationUs)
    {
        ThrowIfDisposed();
        pixels        = [];
        presentationUs = 0;

        if (_capture == null || !_capture.IsOpened() || _atEof)
            return false;

        // Read PosMsec BEFORE advancing (gives timestamp of the frame we are about to read)
        double posMs = _capture.Get(VideoCaptureProperties.PosMsec);

        if (!_capture.Read(_mat) || _mat.Empty())
        {
            _atEof = true;
            return false;
        }

        // Timestamp resolution
        if (posMs > 0.0)
            presentationUs = (long)(posMs * 1_000.0);
        else if (FrameRate > 0.0)
            presentationUs = (long)(_currentFrameIndex * (1_000_000.0 / FrameRate));
        else
            presentationUs = _currentFrameIndex * 33_333L;

        _currentFrameIndex++;

        pixels = ExtractBgr24(_mat);
        return true;
    }

    public bool Seek(int frameIndex)
    {
        ThrowIfDisposed();
        if (_capture == null || !_capture.IsOpened()) return false;

        bool ok = _capture.Set(VideoCaptureProperties.PosFrames, frameIndex);
        if (ok)
        {
            _currentFrameIndex = frameIndex;
            _atEof             = false;
        }
        return ok;
    }

    public void Close()
    {
        _capture?.Release();
        _capture = null;
        _atEof   = false;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _mat.Dispose();
        _capture?.Release();
        _capture = null;
    }

    // -------------------------------------------------------------------------
    // Pixel conversion helpers
    // -------------------------------------------------------------------------

    private static byte[] ExtractBgr24(Mat src)
    {
        Mat bgr;
        bool own = false;

        switch (src.Channels())
        {
            case 3:
                bgr = src;
                break;
            case 4:
                bgr = new Mat(); own = true;
                Cv2.CvtColor(src, bgr, ColorConversionCodes.BGRA2BGR);
                break;
            case 1:
                bgr = new Mat(); own = true;
                Cv2.CvtColor(src, bgr, ColorConversionCodes.GRAY2BGR);
                break;
            default:
                bgr = new Mat(); own = true;
                Cv2.CvtColor(src, bgr, ColorConversionCodes.BGRA2BGR);
                break;
        }

        try
        {
            int bytes = bgr.Rows * bgr.Cols * 3;
            var buf   = new byte[bytes];
            Marshal.Copy(bgr.Data, buf, 0, bytes);
            return buf;
        }
        finally
        {
            if (own) bgr.Dispose();
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenCvSharpVideoBackend));
    }
}
