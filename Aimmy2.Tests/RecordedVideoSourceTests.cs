using KingAim.Core.Capture;
using KingAim.Core.Capture.Sources;
using Xunit;

namespace Aimmy2.Tests;

public sealed class RecordedVideoSourceTests
{
    // ---------------------------------------------------------------------------
    // Fake backend — returns synthetic 1x1 BGR frames with incrementing timestamps
    // ---------------------------------------------------------------------------

    private sealed class FakeVideoBackend : IVideoBackend
    {
        private readonly int _frameCount;
        private int _position;
        public bool   IsOpen     { get; private set; }
        public int    Width      => 320;
        public int    Height     => 240;
        public double FrameRate  => 30.0;
        public int    FrameCount => _frameCount;
        public int    DisposeCalls { get; private set; }
        public bool   IsDisposed   => DisposeCalls > 0;

        public FakeVideoBackend(int frameCount = 10)
            => _frameCount = frameCount;

        public bool Open(string path)
        {
            IsOpen    = true;
            _position = 0;
            return true;
        }

        public bool TryReadFrame(out byte[] pixels, out long presentationUs)
        {
            if (_position >= _frameCount)
            {
                pixels = [];
                presentationUs = 0;
                return false;
            }

            // Minimal 1-pixel BGR frame; timestamp increments at 30fps
            pixels         = new byte[Width * Height * 3];
            presentationUs = (long)(_position * (1_000_000.0 / FrameRate));
            _position++;
            return true;
        }

        public bool Seek(int frameIndex)
        {
            if (frameIndex < 0 || frameIndex >= _frameCount) return false;
            _position = frameIndex;
            return true;
        }

        public void Close() => IsOpen = false;

        public void Dispose()
        {
            DisposeCalls++;
            Close();
        }

        public int Position => _position;
    }

    // ---------------------------------------------------------------------------
    // Test 1 — All frames are delivered
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task RecordedVideoSource_ReadsAllFrames()
    {
        var backend = new FakeVideoBackend(frameCount: 10);
        backend.Open("fake.mp4");
        using var src = new RecordedVideoSource(backend, PlaybackTimingMode.AsFastAsPossible);
        await src.StartAsync();

        int count = 0;
        while (src.GetLatestFrame() != null) count++;

        Assert.Equal(10, count);
    }

    // ---------------------------------------------------------------------------
    // Test 2 — Frame IDs are monotonically increasing
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task RecordedVideoSource_FrameIdsAreMonotonic()
    {
        var backend = new FakeVideoBackend(frameCount: 5);
        backend.Open("fake.mp4");
        using var src = new RecordedVideoSource(backend, PlaybackTimingMode.AsFastAsPossible);
        await src.StartAsync();

        var ids  = new List<long>();
        CapturedFrame? f;
        while ((f = src.GetLatestFrame()) != null) ids.Add(f.FrameId);

        Assert.Equal(5, ids.Count);
        for (int i = 1; i < ids.Count; i++)
            Assert.True(ids[i] > ids[i - 1], $"FrameId not monotonic at index {i}");
    }

    // ---------------------------------------------------------------------------
    // Test 3 — Timestamps increase across frames
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task RecordedVideoSource_TimestampsIncrease()
    {
        var backend = new FakeVideoBackend(frameCount: 5);
        backend.Open("fake.mp4");
        using var src = new RecordedVideoSource(backend, PlaybackTimingMode.AsFastAsPossible);
        await src.StartAsync();

        var ts = new List<long>();
        CapturedFrame? f;
        while ((f = src.GetLatestFrame()) != null) ts.Add(f.CaptureTimestampUs);

        Assert.Equal(5, ts.Count);
        for (int i = 1; i < ts.Count; i++)
            Assert.True(ts[i] > ts[i - 1], $"Timestamp not increasing at index {i}");
    }

    // ---------------------------------------------------------------------------
    // Test 4 — ManualStep is deterministic: same frame until AdvanceFrame is called
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task RecordedVideoSource_ManualStepIsDeterministic()
    {
        var backend = new FakeVideoBackend(frameCount: 5);
        backend.Open("fake.mp4");
        using var src = new RecordedVideoSource(backend, PlaybackTimingMode.ManualStep);
        await src.StartAsync();

        // No advance yet — should return null (nothing decoded yet)
        Assert.Null(src.GetLatestFrame());

        // Advance once
        src.AdvanceFrame();
        var f1a = src.GetLatestFrame();
        var f1b = src.GetLatestFrame();   // same frame, no second advance

        Assert.NotNull(f1a);
        Assert.Equal(f1a!.FrameId, f1b!.FrameId);

        // Advance again — different frame
        src.AdvanceFrame();
        var f2 = src.GetLatestFrame();
        Assert.NotNull(f2);
        Assert.NotEqual(f1a.FrameId, f2!.FrameId);
    }

    // ---------------------------------------------------------------------------
    // Test 5 — Restart returns to the first frame
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task RecordedVideoSource_RestartReturnsToFirstFrame()
    {
        var backend = new FakeVideoBackend(frameCount: 5);
        backend.Open("fake.mp4");
        using var src = new RecordedVideoSource(backend, PlaybackTimingMode.AsFastAsPossible);
        await src.StartAsync();

        var f1 = src.GetLatestFrame();
        var f2 = src.GetLatestFrame();
        Assert.NotNull(f1);
        Assert.NotNull(f2);
        Assert.NotEqual(f1!.FrameId, f2!.FrameId);

        src.Restart();

        var fAfterRestart = src.GetLatestFrame();
        Assert.NotNull(fAfterRestart);
        // Timestamp should be back to 0 (start of video)
        Assert.Equal(0L, fAfterRestart!.CaptureTimestampUs);
    }

    // ---------------------------------------------------------------------------
    // Test 6 — Dispose calls the backend's Dispose exactly once
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task RecordedVideoSource_DisposesCapture()
    {
        var backend = new FakeVideoBackend(frameCount: 3);
        backend.Open("fake.mp4");
        var src = new RecordedVideoSource(backend, PlaybackTimingMode.AsFastAsPossible);
        await src.StartAsync();
        src.GetLatestFrame();

        src.Dispose();
        src.Dispose();  // second call must be safe

        Assert.Equal(1, backend.DisposeCalls);
        Assert.False(src.IsRunning);
    }
}
