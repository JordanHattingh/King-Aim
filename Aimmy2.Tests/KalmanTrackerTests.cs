using KingAim.Core.Perception;
using KingAim.Core.Tracking;
using Xunit;

namespace Aimmy2.Tests;

public sealed class KalmanTrackerTests
{
    private const long FrameUs = 33_333L; // ~30 fps

    private static PoseDetection MakeDet(float cx, float cy, float halfSize = 50f, float conf = 0.9f)
    {
        float l = cx - halfSize, t = cy - halfSize;
        float r = cx + halfSize, b = cy + halfSize;
        float dy = b - t;
        return new PoseDetection
        {
            BoundingBox          = new DetectionBoundingBox(l, t, r, b),
            ObjectConfidence     = conf,
            SourceFrameId        = 0,
            CaptureTimestampUs   = 0,
            InferenceTimestampUs = 0,
            ModelId              = "test",
            Keypoints            =
            [
                new PoseKeypoint(KeypointName.Head,       cx, t + dy * 0.05f, 0.9f, KeypointVisibility.Visible),
                new PoseKeypoint(KeypointName.Neck,       cx, t + dy * 0.20f, 0.9f, KeypointVisibility.Visible),
                new PoseKeypoint(KeypointName.UpperChest, cx, t + dy * 0.45f, 0.9f, KeypointVisibility.Visible),
                new PoseKeypoint(KeypointName.Hip,        cx, t + dy * 0.75f, 0.9f, KeypointVisibility.Visible),
            ],
        };
    }

    // ── Basic sanity ──────────────────────────────────────────────────────────

    [Fact]
    public void NewDetection_CreatesTrack_WithCorrectBounds()
    {
        var tracker = new KalmanTrackerService();
        var result  = tracker.Update([MakeDet(300f, 200f)], 0, FrameUs);

        Assert.Single(result);
        Assert.Equal(1, result[0].TrackId);
        Assert.Equal(1, result[0].Age);
        Assert.Equal(0, result[0].MissingFrames);
        // Box should be near the input detection (filter may adjust slightly)
        Assert.InRange(result[0].Box.CentreX, 290f, 310f);
        Assert.InRange(result[0].Box.CentreY, 190f, 210f);
    }

    [Fact]
    public void TrackId_IsPreserved_AcrossConsecutiveFrames()
    {
        var tracker = new KalmanTrackerService();
        var r0 = tracker.Update([MakeDet(300f, 200f)], 0, 1 * FrameUs);
        var r1 = tracker.Update([MakeDet(310f, 200f)], 1, 2 * FrameUs);
        var r2 = tracker.Update([MakeDet(320f, 200f)], 2, 3 * FrameUs);

        Assert.Equal(r0[0].TrackId, r1[0].TrackId);
        Assert.Equal(r0[0].TrackId, r2[0].TrackId);
    }

    [Fact]
    public void MultipleTargets_GetSeparateIds()
    {
        var tracker = new KalmanTrackerService();
        var result  = tracker.Update([MakeDet(100f, 100f), MakeDet(800f, 600f)], 0, FrameUs);

        Assert.Equal(2, result.Count);
        Assert.NotEqual(result[0].TrackId, result[1].TrackId);
    }

    // ── Kalman velocity benefit ───────────────────────────────────────────────

    [Fact]
    public void TrackSurvives_FastMovement_WithEstablishedVelocity()
    {
        // Without velocity prediction a 70 px jump on a 100 px box gives IoU ≈ 0.18,
        // below the 0.25 threshold — the track is lost.
        // With Kalman prediction the predicted box advances ~13+ px (for vx ≥ 400 px/s),
        // reducing the effective gap to ≤ 57 px and restoring IoU to ≥ 0.27.
        var tracker = new KalmanTrackerService(iouThreshold: 0.25f);
        const float BoxHalf  = 50f;   // 100×100 px box
        const float NormalDx = 30f;   // px/frame at 30 fps → 900 px/s
        const int   Warmup   = 10;

        int   trackId = -1;
        float lastCx  = 200f;

        // Warm up: establish velocity over several frames at constant motion
        for (int frame = 0; frame < Warmup; frame++)
        {
            float cx = 200f + frame * NormalDx;
            var result = tracker.Update([MakeDet(cx, 300f, BoxHalf)], frame, (frame + 1) * FrameUs);
            Assert.Single(result);
            if (frame == 0) trackId = result[0].TrackId;
            else            Assert.Equal(trackId, result[0].TrackId);
            lastCx = cx;
        }

        // 70 px jump from the LAST MATCHED position; without Kalman: IoU ≈ 0.176 < 0.25.
        float fastCx = lastCx + 70f;
        var fastResult = tracker.Update([MakeDet(fastCx, 300f, BoxHalf)], Warmup, (Warmup + 1) * FrameUs);

        Assert.Single(fastResult);
        Assert.Equal(trackId, fastResult[0].TrackId);
    }

    [Fact]
    public void KalmanVelocity_ConvergesNear_TrueVelocity_AfterMultipleFrames()
    {
        // True velocity: 30 px per frame at 30 fps = 900 px/s
        var tracker = new KalmanTrackerService();
        const float DxPerFrame = 30f;

        for (int frame = 0; frame < 25; frame++)
            tracker.Update([MakeDet(100f + frame * DxPerFrame, 200f)], frame, (frame + 1) * FrameUs);

        float estimatedVx = tracker.ActiveTracks[0].VelocityX; // px/s
        float trueVx = DxPerFrame / (FrameUs / 1_000_000f);    // 900 px/s

        // Converged to within 25% of true velocity
        Assert.InRange(estimatedVx, trueVx * 0.75f, trueVx * 1.25f);
    }

    // ── Missing-frame handling ────────────────────────────────────────────────

    [Fact]
    public void TrackExpires_AfterMaxMissingFrames()
    {
        var tracker = new KalmanTrackerService(maxMissingFrames: 3);
        tracker.Update([MakeDet(300f, 300f)], 0, 1 * FrameUs);

        // No detections for 4 frames — track should expire after 3 missing
        for (int frame = 1; frame <= 4; frame++)
        {
            var result = tracker.Update([], frame, (frame + 1) * FrameUs);
            if (frame <= 3)
                Assert.Single(result);
            else
                Assert.Empty(result);
        }
    }

    [Fact]
    public void MissedTrack_BoxContinuesToPredict_NotFrozen()
    {
        // When a fast-moving track misses a frame, its predicted box should advance
        var tracker = new KalmanTrackerService(maxMissingFrames: 5);
        const float DxPerFrame = 40f;

        // Establish velocity over 5 frames
        float lastCx = 0f;
        for (int frame = 0; frame < 5; frame++)
        {
            float cx = 100f + frame * DxPerFrame;
            lastCx = cx;
            tracker.Update([MakeDet(cx, 200f)], frame, (frame + 1) * FrameUs);
        }

        // One missed frame — predicted box should move forward
        var missed = tracker.Update([], 5, 6 * FrameUs);
        Assert.Single(missed);
        // Predicted cx should be greater than last matched cx
        Assert.True(missed[0].Box.CentreX > lastCx,
            $"Expected predicted cx > {lastCx}, got {missed[0].Box.CentreX}");
    }

    // ── Numerical robustness ─────────────────────────────────────────────────

    [Fact]
    public void NoNaN_WhenTimestampUnchanged()
    {
        var tracker = new KalmanTrackerService();
        tracker.Update([MakeDet(200f, 200f)], 0, 100_000L);
        // Same timestamp → dt clamped to 1/240
        var result = tracker.Update([MakeDet(200f, 200f)], 1, 100_000L);

        Assert.Single(result);
        Assert.True(float.IsFinite(result[0].Box.CentreX));
        Assert.True(float.IsFinite(result[0].Box.CentreY));
        Assert.True(float.IsFinite(result[0].VelocityX));
        Assert.True(float.IsFinite(result[0].VelocityY));
    }

    [Fact]
    public void NoNaN_WhenTimestampJumpsLarge()
    {
        var tracker = new KalmanTrackerService();
        tracker.Update([MakeDet(200f, 200f)], 0, 1L);
        // Huge timestamp gap → dt clamped to 0.5 s
        var result = tracker.Update([MakeDet(200f, 200f)], 1, long.MaxValue / 2);

        Assert.Single(result);
        Assert.True(float.IsFinite(result[0].Box.CentreX));
        Assert.True(float.IsFinite(result[0].VelocityX));
    }

    [Fact]
    public void NoNaN_OnFirstFrameWithNoTimestampHistory()
    {
        var tracker = new KalmanTrackerService();
        // captureTimestampUs = 0 means _lastTimestampUs is also 0 → use default 1/30 dt
        var result = tracker.Update([MakeDet(300f, 300f)], 0, 0L);

        Assert.Single(result);
        Assert.True(float.IsFinite(result[0].Box.CentreX));
    }

    // ── Reset ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Reset_ClearsAllState()
    {
        var tracker = new KalmanTrackerService();
        tracker.Update([MakeDet(100f, 100f), MakeDet(500f, 500f)], 0, FrameUs);
        tracker.Reset();

        Assert.Empty(tracker.ActiveTracks);

        // After reset, new detections get fresh IDs starting from 1
        var result = tracker.Update([MakeDet(200f, 200f)], 0, FrameUs);
        Assert.Single(result);
        Assert.Equal(1, result[0].TrackId);
    }
}
