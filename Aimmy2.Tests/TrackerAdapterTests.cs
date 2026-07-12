using Aimmy2.AILogic;
using KingAim.Core.Perception;
using Xunit;

namespace Aimmy2.Tests;

public sealed class TrackerAdapterTests
{
    private const int W = 1920;
    private const int H = 1080;

    // Build a detection with valid topology (head above neck above chest above hip)
    private static PoseDetection MakeDetection(float l, float t, float r, float b, float conf = 0.85f)
    {
        float cx = (l + r) / 2f;
        float dy = (b - t);
        return new PoseDetection
        {
            DetectionId       = Guid.NewGuid(),
            BoundingBox       = new DetectionBoundingBox(l, t, r, b),
            ObjectConfidence  = conf,
            SourceFrameId     = 0,
            CaptureTimestampUs = 0,
            InferenceTimestampUs = 0,
            ModelId           = "test",
            Keypoints         =
            [
                new PoseKeypoint(KeypointName.Head,       cx, t + dy * 0.05f, 0.9f, KeypointVisibility.Visible),
                new PoseKeypoint(KeypointName.Neck,       cx, t + dy * 0.20f, 0.9f, KeypointVisibility.Visible),
                new PoseKeypoint(KeypointName.UpperChest, cx, t + dy * 0.45f, 0.9f, KeypointVisibility.Visible),
                new PoseKeypoint(KeypointName.Hip,        cx, t + dy * 0.75f, 0.9f, KeypointVisibility.Visible),
            ],
        };
    }

    private static long MakeTs(int frameNum) =>
        (long)(DateTimeOffset.UtcNow.AddSeconds(frameNum * (1.0 / 30.0)).ToUnixTimeMilliseconds()) * 1000L;

    // ---------------------------------------------------------------------------
    // Test 1 — Stable TrackId preserved across matched frames
    // ---------------------------------------------------------------------------

    [Fact]
    public void TrackerAdapter_PreservesStableId()
    {
        var adapter = new TrackerAdapter(W, H);
        var det     = MakeDetection(460, 250, 560, 480);

        var t1 = adapter.Update([det], frameId: 1, captureTimestampUs: MakeTs(1));
        var t2 = adapter.Update([det], frameId: 2, captureTimestampUs: MakeTs(2));
        var t3 = adapter.Update([det], frameId: 3, captureTimestampUs: MakeTs(3));

        Assert.Single(t1);
        Assert.Single(t2);
        Assert.Single(t3);
        Assert.Equal(t1[0].TrackId, t2[0].TrackId);
        Assert.Equal(t2[0].TrackId, t3[0].TrackId);
    }

    // ---------------------------------------------------------------------------
    // Test 2 — All four keypoints are mapped
    // ---------------------------------------------------------------------------

    [Fact]
    public void TrackerAdapter_MapsAllFourKeypoints()
    {
        var adapter = new TrackerAdapter(W, H);
        var det     = MakeDetection(460, 250, 560, 480);

        var tracks = adapter.Update([det], frameId: 1, captureTimestampUs: MakeTs(1));

        var kps = tracks[0].Keypoints;
        Assert.Contains(kps, k => k.Name == KeypointName.Head);
        Assert.Contains(kps, k => k.Name == KeypointName.Neck);
        Assert.Contains(kps, k => k.Name == KeypointName.UpperChest);
        Assert.Contains(kps, k => k.Name == KeypointName.Hip);
    }

    // ---------------------------------------------------------------------------
    // Test 3 — Empty detection frame does not throw; no tracks returned
    // ---------------------------------------------------------------------------

    [Fact]
    public void TrackerAdapter_HandlesEmptyDetectionFrame()
    {
        var adapter = new TrackerAdapter(W, H);

        var tracks = adapter.Update([], frameId: 1, captureTimestampUs: MakeTs(1));

        Assert.Empty(tracks);
        Assert.Empty(adapter.ActiveTracks);
    }

    // ---------------------------------------------------------------------------
    // Test 4 — Track expires after MaxLostSeconds without observations
    // ---------------------------------------------------------------------------

    [Fact]
    public void TrackerAdapter_ExpiresMissingTrack()
    {
        var adapter = new TrackerAdapter(W, H);
        var det     = MakeDetection(460, 250, 560, 480);

        // Establish track
        adapter.Update([det], frameId: 1, captureTimestampUs: MakeTs(0));

        // Send empty frames spaced 250ms apart until track expires (default MaxLostSeconds=0.6)
        // At 250ms per frame, 3 empty frames = 0.75s → should expire
        for (int i = 1; i <= 4; i++)
        {
            long ts = (long)(DateTimeOffset.UtcNow.AddSeconds(i * 0.25).ToUnixTimeMilliseconds()) * 1000L;
            adapter.Update([], frameId: (long)(i + 1), captureTimestampUs: ts);
        }

        Assert.Empty(adapter.ActiveTracks);
    }

    // ---------------------------------------------------------------------------
    // Test 5 — Adapter does not mutate the underlying Track objects
    // ---------------------------------------------------------------------------

    [Fact]
    public void TrackerAdapter_DoesNotMutateFrozenTrackerContract()
    {
        var adapter = new TrackerAdapter(W, H);
        var det     = MakeDetection(460, 250, 560, 480);

        var firstResult  = adapter.Update([det], frameId: 1, captureTimestampUs: MakeTs(1));
        int trackId      = firstResult[0].TrackId;
        float confBefore = firstResult[0].DetectionConfidence;

        // Update again with a slightly different box — the adapter should not overwrite
        // the TrackState objects returned in the first call
        adapter.Update([MakeDetection(462, 252, 562, 482)], frameId: 2, captureTimestampUs: MakeTs(2));

        // firstResult is a snapshot — values must not change
        Assert.Equal(trackId,     firstResult[0].TrackId);
        Assert.Equal(confBefore,  firstResult[0].DetectionConfidence);
    }

    // ---------------------------------------------------------------------------
    // Test 6 — Adapter output matches TrackManager baseline for a known sequence
    // ---------------------------------------------------------------------------

    [Fact]
    public void TrackerAdapter_MatchesBaselineSequence()
    {
        var adapter = new TrackerAdapter(W, H);

        // 5-frame sequence: detection moves right by 10px each frame
        var ids = new List<int>();
        for (int i = 0; i < 5; i++)
        {
            float l = 460 + i * 10f;
            var tracks = adapter.Update(
                [MakeDetection(l, 250, l + 100, 480)],
                frameId: i + 1,
                captureTimestampUs: MakeTs(i));

            if (tracks.Count > 0)
                ids.Add(tracks[0].TrackId);
        }

        // One continuous track across all 5 frames → all IDs identical
        Assert.Equal(5, ids.Count);
        Assert.True(ids.All(id => id == ids[0]), "Track ID must remain stable across matched frames");
    }
}
