using System.IO;
using System.Security.Cryptography;
using Aimmy2.Capture;
using KingAim.Core.Capture;
using KingAim.Core.Capture.Sources;
using OpenCvSharp;
using Xunit;

namespace Aimmy2.Tests;

/// <summary>
/// Integration tests for OpenCvSharpVideoBackend.
/// All tests share one fixture video generated in TempPath.
/// </summary>
public sealed class OpenCvBackendTests : IClassFixture<OpenCvBackendTests.VideoFixture>
{
    private readonly VideoFixture _fx;

    public OpenCvBackendTests(VideoFixture fx) => _fx = fx;

    // =========================================================================
    // Fixture — generate a tiny synthetic AVI once per test run
    // =========================================================================

    public sealed class VideoFixture : IDisposable
    {
        public string VideoPath   { get; }
        public int    FrameCount  { get; } = 15;
        public int    Width       { get; } = 64;
        public int    Height      { get; } = 48;
        public double FrameRate   { get; } = 30.0;

        public VideoFixture()
        {
            VideoPath = Path.Combine(Path.GetTempPath(), "king_aim_test_fixture.avi");
            GenerateIfMissing();
        }

        private void GenerateIfMissing()
        {
            if (File.Exists(VideoPath)) return;

            using var writer = new VideoWriter(
                VideoPath,
                VideoWriter.FourCC('M', 'J', 'P', 'G'),
                FrameRate,
                new Size(Width, Height));

            if (!writer.IsOpened())
                throw new InvalidOperationException(
                    $"VideoWriter could not open '{VideoPath}'. Check MJPG codec availability.");

            for (int i = 0; i < FrameCount; i++)
            {
                // Each frame has a unique background colour so frames are distinguishable
                byte b = (byte)((i * 17) & 0xFF);
                byte g = (byte)((i *  7) & 0xFF);
                byte r = (byte)((i * 31) & 0xFF);
                using var frame = new Mat(Height, Width, MatType.CV_8UC3, new Scalar(b, g, r));
                writer.Write(frame);
            }
        }

        /// <summary>Open a fresh backend pointing at the fixture video.</summary>
        public OpenCvSharpVideoBackend OpenBackend()
        {
            var b = new OpenCvSharpVideoBackend();
            Assert.True(b.Open(VideoPath), $"Failed to open fixture video at {VideoPath}");
            return b;
        }

        public void Dispose() { /* temp file cleaned up by OS */ }
    }

    // =========================================================================
    // Tests
    // =========================================================================

    [Fact]
    public void OpenCvBackend_OpensValidVideo()
    {
        using var b = new OpenCvSharpVideoBackend();
        bool ok = b.Open(_fx.VideoPath);
        Assert.True(ok);
        Assert.True(b.IsOpen);
    }

    [Fact]
    public void OpenCvBackend_ReportsDimensionsAndFps()
    {
        using var b = _fx.OpenBackend();
        Assert.Equal(_fx.Width,  b.Width);
        Assert.Equal(_fx.Height, b.Height);
        Assert.True(b.FrameRate > 0, "FrameRate should be positive");
        // FrameCount may be -1 for some codecs — just check it's reported
        Assert.True(b.FrameCount > 0 || b.FrameCount == -1);
    }

    [Fact]
    public void OpenCvBackend_ReadsAllFrames()
    {
        using var b = _fx.OpenBackend();
        int count = 0;
        while (b.TryReadFrame(out _, out _)) count++;

        Assert.Equal(_fx.FrameCount, count);
    }

    [Fact]
    public void OpenCvBackend_ReturnsOwnedFrameMemory()
    {
        using var b = _fx.OpenBackend();

        // Read frame 0 and keep a reference
        Assert.True(b.TryReadFrame(out byte[] frame0, out _));
        byte[] snapshot = (byte[])frame0.Clone();
        string hashBefore = Sha256Hex(frame0);

        // Read several more frames — they must NOT alias frame0's buffer
        for (int i = 0; i < 5; i++)
            b.TryReadFrame(out _, out _);

        string hashAfter = Sha256Hex(frame0);
        Assert.Equal(hashBefore, hashAfter);
        Assert.True(frame0.SequenceEqual(snapshot), "frame0 buffer was mutated by later reads");
    }

    [Fact]
    public void OpenCvBackend_TimestampsAreMonotonic()
    {
        using var b = _fx.OpenBackend();
        long prev = -1;
        while (b.TryReadFrame(out _, out long ts))
        {
            Assert.True(ts >= prev, $"Timestamp not monotonic: {ts} < {prev}");
            prev = ts;
        }
        Assert.True(prev >= 0, "No frames were read");
    }

    [Fact]
    public void OpenCvBackend_SeekReturnsRequestedFrame()
    {
        using var b = _fx.OpenBackend();

        // Read several frames to advance past the start
        for (int i = 0; i < 7; i++)
            b.TryReadFrame(out _, out _);

        // Seek back to frame 0 — next timestamp should be near zero
        bool seekOk = b.Seek(0);
        Assert.True(seekOk);

        Assert.True(b.TryReadFrame(out _, out long ts));
        // Allow up to one frame's worth of jitter
        long oneFrameUs = (long)(1_000_000.0 / _fx.FrameRate) + 1;
        Assert.True(ts < oneFrameUs, $"After Seek(0) expected ts~0 but got {ts}");
    }

    [Fact]
    public void OpenCvBackend_RestartReturnsFirstFrame()
    {
        using var b = _fx.OpenBackend();

        // Read a few frames
        for (int i = 0; i < 5; i++) b.TryReadFrame(out _, out _);

        // Restart via Seek(0)
        b.Seek(0);
        Assert.True(b.TryReadFrame(out _, out long ts));
        long oneFrameUs = (long)(1_000_000.0 / _fx.FrameRate) + 1;
        Assert.True(ts < oneFrameUs, $"After Seek(0) expected ts~0 but got {ts}");
    }

    [Fact]
    public void OpenCvBackend_EndOfStreamDoesNotThrow()
    {
        using var b = _fx.OpenBackend();

        // Drain all frames
        while (b.TryReadFrame(out _, out _)) { }

        // Extra reads must return false cleanly, not throw
        var ex = Record.Exception(() =>
        {
            for (int i = 0; i < 5; i++)
                b.TryReadFrame(out _, out _);
        });
        Assert.Null(ex);
    }

    [Fact]
    public void OpenCvBackend_RejectsMissingFile()
    {
        using var b = new OpenCvSharpVideoBackend();
        bool ok = b.Open(@"C:\does\not\exist\missing.mp4");
        Assert.False(ok);
        Assert.False(b.IsOpen);
    }

    [Fact]
    public void OpenCvBackend_DisposeIsIdempotent()
    {
        var b = _fx.OpenBackend();
        var ex = Record.Exception(() =>
        {
            b.Dispose();
            b.Dispose();
        });
        Assert.Null(ex);
    }

    [Fact]
    public void OpenCvBackend_RejectsReadAfterDispose()
    {
        var b = _fx.OpenBackend();
        b.Dispose();
        Assert.Throws<ObjectDisposedException>(() => b.TryReadFrame(out _, out _));
    }

    [Fact]
    public async Task OpenCvBackend_EffectiveTimestampsAreStrictlyIncreasing()
    {
        // RecordedVideoSource normalizes raw codec timestamps (which may duplicate at
        // MJPG's millisecond resolution) into a strictly-increasing pipeline timestamp.
        using var backend = _fx.OpenBackend();
        var source = new RecordedVideoSource(backend, PlaybackTimingMode.ManualStep);
        await source.StartAsync();

        long prevEffective = -1;
        for (int i = 0; i < _fx.FrameCount; i++)
        {
            source.AdvanceFrame();
            var frame = source.GetLatestFrame();
            if (frame == null) break;

            Assert.True(frame.CaptureTimestampUs > prevEffective,
                $"Frame {i}: effective ts {frame.CaptureTimestampUs} not > prev {prevEffective}");
            prevEffective = frame.CaptureTimestampUs;
        }
        Assert.True(prevEffective > 0, "No frames produced");
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static string Sha256Hex(byte[] data)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(data, hash);
        return Convert.ToHexString(hash);
    }
}
