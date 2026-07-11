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
        var recorder = new ScenarioMetricsRecorder("Cross", 30);
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
