using System.Drawing;
using System.IO;
using Aimmy2.TestArena;
using Xunit;

namespace Aimmy2.Tests;

public sealed class ScenarioMetricsTests
{
    [Fact]
    public void Recorder_CountsMatchesFalsePositivesAndMisses()
    {
        var recorder = new ScenarioMetricsRecorder("StaticEnemy", matchRadiusPixels: 30);
        DateTime now = DateTime.UtcNow;
        recorder.Record(
            now,
            [new ArenaGroundTruth("enemy1", new PointF(100, 100), true)],
            [new ArenaDetection(4, new PointF(105, 100), false, 10, null)],
            5, 8, 60);
        recorder.Record(
            now.AddMilliseconds(16),
            [new ArenaGroundTruth("enemy1", new PointF(100, 100), true)],
            [new ArenaDetection(8, new PointF(400, 400), false, 10, null)],
            7, 9, 58);

        ScenarioMetricsSummary summary = recorder.Summarize();
        Assert.Equal(2, summary.Frames);
        Assert.Equal(2, summary.DetectionCount);
        Assert.Equal(1, summary.FalsePositives);
        Assert.Equal(1, summary.MissedDetections);
        Assert.Equal(1, summary.TrackLosses);
    }

    [Fact]
    public void Recorder_CountsIdentitySwitchAndReacquisition()
    {
        var recorder = new ScenarioMetricsRecorder("Cross", matchRadiusPixels: 30);
        DateTime now = DateTime.UtcNow;
        ArenaGroundTruth[] truth = [new("enemy1", new PointF(100, 100), true)];
        recorder.Record(now, truth, [new ArenaDetection(1, new PointF(100, 100), false, 0, null)], 1, 1, 60);
        recorder.Record(now.AddMilliseconds(20), truth, [], 1, 1, 60);
        recorder.Record(now.AddMilliseconds(60), truth, [new ArenaDetection(2, new PointF(100, 100), true, 20, new PointF(102, 100))], 1, 1, 60);

        ScenarioMetricsSummary summary = recorder.Summarize();
        Assert.Equal(1, summary.IdentitySwitches);
        Assert.Equal(1, summary.TrackLosses);
        Assert.Equal(40, summary.MeanReacquisitionMs, 3);
        Assert.Equal(2, summary.MeanGruErrorPixels, 3);
    }

    // ── Hungarian evaluator edge-case tests ───────────────────────────────────

    [Fact]
    public void Hungarian_OneTarget_ZeroDetections_ReturnsMiss()
    {
        var recorder = new ScenarioMetricsRecorder("H_1T0D", matchRadiusPixels: 50);
        recorder.Record(
            DateTime.UtcNow,
            [new ArenaGroundTruth("t1", new PointF(100, 100), true)],
            [],
            0, 0, 60);
        var s = recorder.Summarize();
        Assert.Equal(0, s.FalsePositives);
        Assert.Equal(1, s.MissedDetections);
    }

    [Fact]
    public void Hungarian_OneTarget_DetectionOutsideRadius_ReturnsMissAndFP()
    {
        var recorder = new ScenarioMetricsRecorder("H_1T1D_Far", matchRadiusPixels: 50);
        recorder.Record(
            DateTime.UtcNow,
            [new ArenaGroundTruth("t1", new PointF(100, 100), true)],
            [new ArenaDetection(1, new PointF(500, 500), false, 0, null)],
            0, 0, 60);
        var s = recorder.Summarize();
        Assert.Equal(1, s.FalsePositives);
        Assert.Equal(1, s.MissedDetections);
    }

    [Fact]
    public void Hungarian_TwoTargets_OneDetection_GloballyAssignsNearest()
    {
        // T1 is 10px away, T2 is 200px away from the single detection.
        // Hungarian must assign the detection to T1 (the global minimum),
        // not greedily to whichever comes first.
        var recorder = new ScenarioMetricsRecorder("H_2T1D", matchRadiusPixels: 150);
        recorder.Record(
            DateTime.UtcNow,
            [
                new ArenaGroundTruth("t1", new PointF(100, 100), true),
                new ArenaGroundTruth("t2", new PointF(300, 100), true),
            ],
            [new ArenaDetection(1, new PointF(110, 100), false, 0, null)],
            0, 0, 60);
        var s = recorder.Summarize();
        // T1 matched (10px < 150 radius), T2 missed, detection not FP.
        Assert.Equal(0, s.FalsePositives);
        Assert.Equal(1, s.MissedDetections);
    }

    [Fact]
    public void Hungarian_OneTarget_TwoDetections_PicksNearest()
    {
        // One target, two detections — pick the nearer one; far one is FP.
        var recorder = new ScenarioMetricsRecorder("H_1T2D", matchRadiusPixels: 100);
        recorder.Record(
            DateTime.UtcNow,
            [new ArenaGroundTruth("t1", new PointF(100, 100), true)],
            [
                new ArenaDetection(1, new PointF(105, 100), false, 0, null),  // 5px
                new ArenaDetection(2, new PointF(180, 100), false, 0, null),  // 80px — still within radius but further
            ],
            0, 0, 60);
        var s = recorder.Summarize();
        Assert.Equal(1, s.FalsePositives);
        Assert.Equal(0, s.MissedDetections);
    }

    [Fact]
    public void Hungarian_AllCostsInf_NeitherMatchNorCrash()
    {
        // Every detection is far beyond radius — should produce miss + FP without crashing.
        var recorder = new ScenarioMetricsRecorder("H_AllInf", matchRadiusPixels: 10);
        recorder.Record(
            DateTime.UtcNow,
            [
                new ArenaGroundTruth("t1", new PointF(100, 100), true),
                new ArenaGroundTruth("t2", new PointF(200, 200), true),
            ],
            [
                new ArenaDetection(1, new PointF(500, 500), false, 0, null),
                new ArenaDetection(2, new PointF(600, 600), false, 0, null),
            ],
            0, 0, 60);
        var s = recorder.Summarize();
        Assert.Equal(2, s.FalsePositives);
        Assert.Equal(2, s.MissedDetections);
    }

    [Fact]
    public void Hungarian_RectangularMatrix_MoreDetectionsThanTargets()
    {
        // 1 target, 3 detections (padded matrix): must not crash and must report 2 FP.
        var recorder = new ScenarioMetricsRecorder("H_Rect", matchRadiusPixels: 50);
        recorder.Record(
            DateTime.UtcNow,
            [new ArenaGroundTruth("t1", new PointF(100, 100), true)],
            [
                new ArenaDetection(1, new PointF(105, 100), false, 0, null),
                new ArenaDetection(2, new PointF(400, 400), false, 0, null),
                new ArenaDetection(3, new PointF(600, 600), false, 0, null),
            ],
            0, 0, 60);
        var s = recorder.Summarize();
        Assert.Equal(2, s.FalsePositives);
        Assert.Equal(0, s.MissedDetections);
    }

    [Fact]
    public void ReportWriter_CreatesJsonAndCsv()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"kingaim-arena-{Guid.NewGuid():N}");
        try
        {
            var recorder = new ScenarioMetricsRecorder("NoTarget");
            recorder.Record(DateTime.UtcNow, [], [], 2, 3, 60);
            ScenarioReportWriter.Write(directory, recorder.Summarize());

            Assert.Single(Directory.GetFiles(directory, "*.json"));
            Assert.Single(Directory.GetFiles(directory, "*.csv"));
        }
        finally
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }
}
