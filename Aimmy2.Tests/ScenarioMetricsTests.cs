using System.Drawing;
using System.IO;
using Aimmy2.AILogic;
using Aimmy2.TestArena;
using Xunit;

namespace Aimmy2.Tests;

public sealed class ScenarioMetricsTests
{
    [Fact]
    public void FixedSimulationClock_UsesExactRationalFrameTimeAndResets()
    {
        var clock = new FixedSimulationClock();
        Assert.Equal(FixedSimulationClock.EpochUtc, clock.CurrentTimestamp);
        Assert.True(clock.Advance() > FixedSimulationClock.EpochUtc);
        while (clock.FrameIndex < 60) clock.Advance();
        Assert.Equal(FixedSimulationClock.EpochUtc.AddSeconds(1), clock.CurrentTimestamp);
        while (clock.FrameIndex < 600) clock.Advance();
        Assert.Equal(FixedSimulationClock.EpochUtc.AddSeconds(10), clock.CurrentTimestamp);
        clock.Reset();
        Assert.Equal(0, clock.FrameIndex);
        Assert.Equal(FixedSimulationClock.EpochUtc, clock.CurrentTimestamp);
    }

    [Fact]
    public void SyntheticProfiles_MapToNamedDeterministicConfigurations()
    {
        foreach (SyntheticProfile profile in Enum.GetValues<SyntheticProfile>())
        {
            SyntheticNoiseConfig config = profile.Configuration();
            Assert.Equal(profile.ToString(), config.ProfileName);
            Assert.Equal(42, config.Seed);
        }
    }

    [Fact]
    public void ContinuityMetrics_DistinguishOutsideGateFromConfirmedExpiry()
    {
        var recorder = new ScenarioMetricsRecorder("Continuity", matchRadiusPixels: 120);
        DateTime now = DateTime.UtcNow;
        ArenaGroundTruth[] target = [new("enemy1", new PointF(100, 100), true)];

        recorder.Record(now, target, [new(7, new PointF(100, 100), false, 0, null)], 0, 0, 60);
        recorder.Record(now.AddMilliseconds(16), target, [new(7, new PointF(230, 100), true, 16, null)], 0, 0, 60);
        recorder.Record(now.AddMilliseconds(32), target, [new(7, new PointF(235, 100), true, 32, null)], 0, 0, 60);
        recorder.Record(now.AddMilliseconds(48), target, [new(7, new PointF(100, 100), false, 0, null)], 0, 0, 60);
        recorder.Record(now.AddMilliseconds(64), target, [], 0, 0, 60);

        ScenarioMetricsSummary summary = recorder.Summarize();
        Assert.Equal(2, summary.PositionGateMissFrames);
        Assert.Equal(2, summary.SameTrackOutsideGateFrames);
        Assert.Equal(2, summary.AssociationLossEvents);
        Assert.Equal(1, summary.ConfirmedTrackExpiryEvents);
        Assert.Equal(135, summary.MaximumPositionErrorPixels, 3);
        Assert.Equal(2, summary.MaximumConsecutivePositionMissFrames);
    }

    private static ScenarioMetricsSummary ClassifyOcclusionDetection(int trackId, double ageMs, bool extrapolated, double elapsedMs)
    {
        var recorder = new ScenarioMetricsRecorder("OcclusionClassification", matchRadiusPixels: 30);
        DateTime now = DateTime.UtcNow;
        recorder.Record(now, [new("enemy1", new PointF(100, 100), true)],
            [new(7, new PointF(100, 100), false, 0, null)], 0, 0, 60);
        recorder.Record(now.AddMilliseconds(elapsedMs), [new("enemy1", new PointF(100, 100), false)],
            [new(trackId, new PointF(100, 100), extrapolated, ageMs, null)], 0, 0, 60);
        return recorder.Summarize();
    }

    [Theory]
    [InlineData(200, true)]
    [InlineData(450, false)]
    public void Occlusion_OldTrackWithinWindow_IsExpectedPersistence(double ageMs, bool extrapolated)
    {
        ScenarioMetricsSummary summary = ClassifyOcclusionDetection(7, ageMs, extrapolated, ageMs);
        Assert.Equal(1, summary.ExpectedOcclusionPersistenceFrames);
        Assert.Equal(0, summary.FalsePositives);
    }

    [Fact]
    public void Occlusion_OldTrackAfter601Ms_IsGhostAfterDeadline()
    {
        ScenarioMetricsSummary summary = ClassifyOcclusionDetection(7, 601, false, 601);
        Assert.Equal(1, summary.GhostAfterPersistenceDeadline);
        Assert.Equal(1, summary.FalsePositives);
        Assert.True(summary.OcclusionExceededPersistenceWindow);
        Assert.False(summary.OldTrackExpiredByDeadline);
    }

    [Fact]
    public void Occlusion_DifferentTrackWithinWindow_IsUnrelatedFalsePositive()
    {
        ScenarioMetricsSummary summary = ClassifyOcclusionDetection(8, 100, false, 100);
        Assert.Equal(1, summary.UnrelatedFalsePositive);
        Assert.Equal(1, summary.FalsePositives);
    }

    [Fact]
    public void Reappearance_OldAndNewTrackIds_CountsDuplicate()
    {
        var recorder = new ScenarioMetricsRecorder("DuplicateAfterReappearance", matchRadiusPixels: 30);
        DateTime now = DateTime.UtcNow;
        recorder.Record(now, [new("enemy1", new PointF(100, 100), true)],
            [new(7, new PointF(100, 100), false, 0, null)], 0, 0, 60);
        recorder.Record(now.AddMilliseconds(100), [new("enemy1", new PointF(100, 100), false)],
            [new(7, new PointF(100, 100), true, 100, null)], 0, 0, 60);
        recorder.Record(now.AddMilliseconds(200), [new("enemy1", new PointF(100, 100), true)],
            [new(8, new PointF(100, 100), false, 0, null), new(7, new PointF(200, 100), false, 200, null)], 0, 0, 60);

        ScenarioMetricsSummary summary = recorder.Summarize();
        Assert.Equal(1, summary.DuplicateAfterReappearance);
        Assert.Equal(1, summary.FalsePositives);
    }

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

    // ── Production HungarianSolver unit tests ────────────────────────────

    [Fact]
    public void ProdHungarian_1x1_AssignsOnly()
    {
        int[] result = HungarianSolver.Solve(new double[,] { { 5.0 } });
        Assert.Equal(new[] { 0 }, result);
    }

    [Fact]
    public void ProdHungarian_2x2_GlobalMinNotGreedy()
    {
        // Greedy (row-by-row) would assign row0→col0 (cost 1), forcing row1→col1 (cost 100).
        // Global optimum is row0→col1 (cost 2), row1→col0 (cost 2). Total 4 vs greedy 101.
        var cost = new double[,] { { 1, 2 }, { 2, 100 } };
        int[] result = HungarianSolver.Solve(cost);
        Assert.Equal(1, result[0]);  // row 0 → col 1
        Assert.Equal(0, result[1]);  // row 1 → col 0
    }

    [Fact]
    public void ProdHungarian_3x3_OptimalAssignment()
    {
        // All 6 permutations: min is (row0→col1, row1→col0, row2→col2) = 1+2+2 = 5.
        var cost = new double[,]
        {
            { 4, 1, 3 },
            { 2, 0, 5 },
            { 3, 2, 2 },
        };
        int[] result = HungarianSolver.Solve(cost);
        double totalCost = cost[0, result[0]] + cost[1, result[1]] + cost[2, result[2]];
        Assert.Equal(5.0, totalCost, 6);  // optimal = 1+2+2 = 5
    }

    [Fact]
    public void ProdHungarian_AllInfinity_NeitherCrashNorInfiniteLoop()
    {
        // Cost matrix padded entirely with UnmatchedCost (which is float, not +inf).
        // In production this never happens (UnmatchedCost = 1.25 is finite), but the
        // solver must not crash or loop when called with all-positive-infinity entries.
        const double INF = double.PositiveInfinity;
        var cost = new double[,] { { INF, INF }, { INF, INF } };
        int[] result = HungarianSolver.Solve(cost);
        // Solver may assign any valid permutation (all costs equal) — just must return
        // without crashing and produce a valid array of length 2.
        Assert.Equal(2, result.Length);
    }

    [Fact]
    public void ProdHungarian_SymmetricCosts_AssignmentIsValid()
    {
        // All off-diagonal entries are large; diagonal is cheap.
        // Expect identity assignment: row i → col i.
        var cost = new double[,]
        {
            { 0.1, 1.25, 1.25 },
            { 1.25, 0.1, 1.25 },
            { 1.25, 1.25, 0.1 },
        };
        int[] result = HungarianSolver.Solve(cost);
        Assert.Equal(0, result[0]);
        Assert.Equal(1, result[1]);
        Assert.Equal(2, result[2]);
    }

    [Fact]
    public void ReportWriter_CreatesJsonAndCsvAndHash()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"kingaim-arena-{Guid.NewGuid():N}");
        try
        {
            var recorder = new ScenarioMetricsRecorder("NoTarget");
            recorder.Record(DateTime.UtcNow, [], [], 2, 3, 60);
            ScenarioReportWriter.Write(directory, recorder.Summarize());

            Assert.Single(Directory.GetFiles(directory, "*.json"));
            Assert.Single(Directory.GetFiles(directory, "*.csv"));
            Assert.Single(Directory.GetFiles(directory, "*.hash"));
        }
        finally
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ReportWriter_WithRunId_UsesRunIdAsFilename()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"kingaim-arena-{Guid.NewGuid():N}");
        try
        {
            var recorder = new ScenarioMetricsRecorder("DirectionReversal", ScenarioKind.DirectionReversal,
                noiseConfig: SyntheticProfile.Noisy.Configuration())
            {
                RunId = "DirectionReversal_Noisy_R01_seed42_abc1234",
                Repetition = 1,
                GitCommit = "abc1234",
            };
            recorder.Record(FixedSimulationClock.EpochUtc, [], [], 0, 0, 0);
            ScenarioReportWriter.Write(directory, recorder.Summarize());

            Assert.Single(Directory.GetFiles(directory, "DirectionReversal_Noisy_R01_seed42_abc1234.json"));
        }
        finally
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }

    // ── Canonical hash / reproducibility tests ───────────────────────────────

    private static ScenarioMetricsSummary RunDeterministicSequence(int repetitionOffset = 0)
    {
        // Pure evaluator-level reproducibility: same fixed clock, same injected detections.
        // Does not drive TrackManager — tests that ScenarioMetricsRecorder is deterministic
        // when given identical input regardless of wall-clock time.
        _ = repetitionOffset; // all repetitions use the same inputs by design
        var recorder = new ScenarioMetricsRecorder("DirectionReversal", ScenarioKind.DirectionReversal,
            noiseConfig: SyntheticProfile.Noisy.Configuration())
        {
            RunId = $"DirectionReversal_Noisy_R0{repetitionOffset + 1}_seed42_test",
            Repetition = repetitionOffset + 1,
            GitCommit = "test",
        };
        var clock = new FixedSimulationClock();
        for (int i = 0; i < 600; i++)
        {
            DateTime ts = clock.Advance();
            bool occluded = i is >= 120 and < 240;
            var targets = new[] { new ArenaGroundTruth("enemy1", new System.Drawing.PointF(100 + i * 0.5f, 300), !occluded) };
            ArenaDetection[] detections = occluded
                ? []
                : [new ArenaDetection(1, new System.Drawing.PointF(100 + i * 0.5f, 300), false, i * (1000.0 / 60), null)];
            recorder.Record(ts, targets, detections, 0, 0, 0);
        }
        return recorder.Summarize();
    }

    [Fact]
    public void CanonicalHash_SameInputTwice_ProducesIdenticalHash()
    {
        string hash1 = ScenarioReportWriter.CanonicalHash(RunDeterministicSequence(0));
        string hash2 = ScenarioReportWriter.CanonicalHash(RunDeterministicSequence(0));
        Assert.Equal(hash1, hash2);
    }

    [Fact]
    public void CanonicalHash_FiveRepetitions_AllHashesMatch()
    {
        string[] hashes = Enumerable.Range(0, 5)
            .Select(i => ScenarioReportWriter.CanonicalHash(RunDeterministicSequence(i)))
            .ToArray();

        string first = hashes[0];
        foreach (string h in hashes)
            Assert.Equal(first, h);
    }

    [Fact]
    public void CanonicalHash_ExcludesRealElapsedMsAndRunId()
    {
        // Two summaries that differ only in wall-clock metadata must hash identically.
        // We can't control RealElapsedMs directly from outside, but two runs of the same
        // deterministic sequence at slightly different wall times will differ in RealElapsedMs
        // while producing the same canonical hash.
        ScenarioMetricsSummary s1 = RunDeterministicSequence(0);
        ScenarioMetricsSummary s2 = RunDeterministicSequence(1); // different RunId/Repetition

        // The canonical fields are identical regardless of RunId/Repetition/RealElapsedMs.
        Assert.Equal(ScenarioReportWriter.CanonicalHash(s1), ScenarioReportWriter.CanonicalHash(s2));
        // But the summaries themselves differ in RunId.
        Assert.NotEqual(s1.RunId, s2.RunId);
    }

    [Fact]
    public void CanonicalHash_DifferentDetectionCount_ProducesDifferentHash()
    {
        var r1 = new ScenarioMetricsRecorder("DirectionReversal", ScenarioKind.DirectionReversal,
            noiseConfig: SyntheticProfile.Noisy.Configuration());
        var r2 = new ScenarioMetricsRecorder("DirectionReversal", ScenarioKind.DirectionReversal,
            noiseConfig: SyntheticProfile.Noisy.Configuration());
        DateTime ts = FixedSimulationClock.EpochUtc;
        var target = new[] { new ArenaGroundTruth("e1", new System.Drawing.PointF(100, 100), true) };
        r1.Record(ts, target, [new ArenaDetection(1, new System.Drawing.PointF(100, 100), false, 0, null)], 0, 0, 0);
        r2.Record(ts, target, [], 0, 0, 0); // miss — different metrics

        Assert.NotEqual(
            ScenarioReportWriter.CanonicalHash(r1.Summarize()),
            ScenarioReportWriter.CanonicalHash(r2.Summarize()));
    }
}
