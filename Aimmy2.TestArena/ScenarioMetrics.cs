using System.Drawing;
using System.IO;
using System.Text;
using System.Text.Json;

namespace Aimmy2.TestArena;

public sealed record ArenaGroundTruth(string Id, PointF Center, bool Visible);

public sealed record ArenaDetection(
    int TrackId,
    PointF Center,
    bool IsExtrapolated,
    double ObservationAgeMs,
    PointF? GruPredictedCenter);

/// <summary>
/// Snapshot of all metrics recorded over one scenario run.
///
/// BenchmarkDomain clarifies what the numbers mean:
///
///   GameplayReplay — real gameplay frames were fed through the full YOLO + tracker
///     pipeline.  DetectorMetricsGated=true: detection count, FP, and miss counts
///     are actionable accuracy gates.
///
///   SyntheticTracking — coloured rectangles rendered in the arena.  YOLO detects
///     nothing (rectangles are out-of-distribution).  DetectorMetricsGated=false:
///     detection counts are expected to be zero and are not accuracy gates.
///     TrackerMetricsGated=false until direct observation injection is implemented
///     (see TODO in Scenario.cs ScenarioKindExtensions.TrackerMetricsGated).
/// </summary>
public sealed record ScenarioMetricsSummary(
    string Scenario,
    string BenchmarkDomain,
    bool   DetectorExecuted,
    bool   DetectorMetricsGated,
    bool   TrackerMetricsGated,
    // Injection configuration (populated for SyntheticTracking; "N/A" / 0 for GameplayReplay).
    string NoiseProfile,
    bool   InferenceMetricsApplicable,
    int    RandomSeed,
    string NoiseDistribution,
    double PositionNoiseMaxPx,
    double SizeNoisePercent,
    double DropProbabilityPct,
    int    FalsePositivesPerFrame,
    int    ContiguousOcclusionStartFrame,
    int    ContiguousOcclusionDurationFrames,
    // Metrics.
    int    Frames,
    int    DetectionCount,
    int    FalsePositives,
    int    MissedDetections,
    int    IdentitySwitches,
    int    TrackLosses,
    double MeanReacquisitionMs,
    double MeanPredictiveErrorPixels,
    double MeanGruErrorPixels,
    double ObservationAgeP95Ms,
    double InferenceP50Ms,
    double InferenceP95Ms,
    double InferenceP99Ms,
    double FrameAgeP95Ms,
    double CaptureFpsP50,
    // Occlusion-specific metrics (non-zero only for scenarios with Visible=false phases).
    int    OcclusionFrames,
    // FP classification during occlusion:
    //   ExpectedOcclusionPersistenceFrames — old track extrapolating within MaxLostSeconds; NOT an FP.
    //   GhostAfterPersistenceDeadline      — old track still emitted after MaxLostSeconds; IS an FP.
    //   DuplicateAfterReappearance         — old track emitted alongside new track after reappearance; IS an FP.
    //   UnrelatedFalsePositive             — detection unrelated to any occlusion lifecycle event; IS an FP.
    int    ExpectedOcclusionPersistenceFrames,
    int    GhostAfterPersistenceDeadline,
    int    DuplicateAfterReappearance,
    int    UnrelatedFalsePositive,
    // Occlusion transition diagnostics.
    bool   TrackIdPreservedAfterOcclusion,
    // Elapsed time proves the persistence window ended; only observed tracker output proves expiry.
    bool   OcclusionExceededPersistenceWindow,
    bool   OldTrackExpiredByDeadline,
    int    OldTrackIdAtOcclusionStart,
    int    NewTrackIdAtFirstReappearance,
    // Identity-switch classification.
    //   IdentitySwitches                        — total raw count (includes ambiguous).
    //   IdentitySwitchesOutsideAmbiguity        — raw count outside ambiguous frames.
    //   ExpectedTrackReinitializations          — new ID after confirmed expiry; not a tracker failure.
    //   UnexpectedIdentitySwitchesOutsideAmbiguity — true ID swap (not expiry-driven); gated failure.
    int    IdentitySwitchesOutsideAmbiguity,
    int    ExpectedTrackReinitializations,
    int    UnexpectedIdentitySwitchesOutsideAmbiguity,
    // Ambiguity tracking.
    int    AmbiguousIdentityFrames,
    // Per-cycle occlusion aggregates (meaningful for scenarios with repeated occlusion).
    int    OcclusionCycleCount,
    int    SuccessfulReacquisitions,
    int    FailedReacquisitions,
    int    MaxReacquisitionFrames,
    // Identity-switch trace diagnostics.
    int    SwitchTracesWritten);

public sealed class ScenarioMetricsRecorder
{
    private readonly string _scenario;
    private readonly BenchmarkDomain _domain;
    private readonly bool _detectorExecuted;
    private readonly bool _detectorMetricsGated;
    private readonly bool _trackerMetricsGated;
    private readonly SyntheticNoiseConfig? _noiseConfig;
    private readonly double _matchRadiusPixels;
    private readonly Dictionary<string, int> _lastTrackByTarget = new();
    private readonly Dictionary<string, DateTime> _lostAtByTarget = new();
    private readonly List<double> _reacquisitionMs = new();
    private readonly List<double> _predictiveErrors = new();
    private readonly List<double> _gruErrors = new();
    private readonly List<double> _observationAges = new();
    private readonly List<double> _inferenceMs = new();
    private readonly List<double> _frameAgeMs = new();
    private readonly List<double> _captureFps = new();
    private int _frames;
    private int _detections;
    private int _falsePositives;
    private int _misses;
    private int _identitySwitches;
    private int _trackLosses;
    // Occlusion tracking.
    private int _occlusionFrames;
    private bool _trackIdPreservedAfterOcclusion = true;
    private bool _wasOccluded;
    // Allowed persistence window (matches TrackManager.MaxLostSeconds default).
    private const double OcclusionPersistenceWindowSeconds = 0.6;
    private const double OcclusionPersistenceWindowMs      = OcclusionPersistenceWindowSeconds * 1000.0;
    private DateTime? _occlusionStartedAt;
    // Occlusion transition state.
    private int? _trackIdAtOcclusionStart;         // frozen when target first becomes hidden
    private int? _firstTrackIdAfterOcclusion;      // set on first match after reappearance
    private bool _awaitingFirstReappearanceMatch;  // true from hidden→visible until first match
    private bool _countingReacquisitionFrames;     // true from reappearance until first match
    private bool _occlusionExceededPersistenceWindow;
    private bool _oldTrackExpiredByDeadline;       // original ID observed absent after its deadline
    private DateTime? _reappearanceTimestamp;      // when target last became visible (duplicate window)
    // FP buckets (replaces position-based SpuriousFPDuringOcclusion).
    private int _expectedOcclusionPersistenceFrames;
    private int _ghostAfterPersistenceDeadline;
    private int _duplicateAfterReappearance;
    private int _unrelatedFalsePositive;
    // IS classification.
    private int _identitySwitchesOutsideAmbiguity;
    private int _expectedTrackReinitializations;
    private int _unexpectedIdentitySwitchesOutsideAmbiguity;
    // Ambiguity tracking.
    private int _ambiguousIdentityFrames;
    // Per-cycle occlusion aggregates.
    private int _occlusionCycleCount;
    private int _successfulReacquisitions;
    private int _failedReacquisitions;
    private int _currentCycleReacqFrames;
    private int _maxReacquisitionFrames;
    // Switch trace — ring buffer of recent frames for post-hoc CSV diagnostics.
    private const int SwitchWindowBefore = 10;
    private const int SwitchWindowAfter  = 10;
    private readonly Queue<FrameSnap> _snapBuffer = new();
    private readonly List<SwitchTrace> _pendingSwitchTraces = new();
    private int _switchTracesWritten;
    /// <summary>Directory to write per-switch diagnostic CSVs into. Null disables CSV output.</summary>
    public string? SwitchTracePath { get; set; }

    public ScenarioMetricsRecorder(
        string scenario,
        ScenarioKind kind,
        bool detectorExecuted = true,
        SyntheticNoiseConfig? noiseConfig = null,
        double matchRadiusPixels = 120)
    {
        _scenario = scenario;
        _domain = kind.Domain();
        _detectorExecuted = detectorExecuted;
        _detectorMetricsGated = kind.DetectorMetricsGated();
        _trackerMetricsGated = kind.TrackerMetricsGated();
        _noiseConfig = noiseConfig;
        _matchRadiusPixels = matchRadiusPixels;
    }

    /// <summary>Convenience overload for tests — uses GameplayReplay domain defaults.</summary>
    public ScenarioMetricsRecorder(string scenario, double matchRadiusPixels = 120)
        : this(scenario, ScenarioKind.GameplayReplay, matchRadiusPixels: matchRadiusPixels) { }

    public int Frames => _frames;

    public void Record(
        DateTime timestamp,
        IReadOnlyList<ArenaGroundTruth> targets,
        IReadOnlyList<ArenaDetection> detections,
        double inferenceMs,
        double frameAgeMs,
        double captureFps)
    {
        _frames++;
        _detections += detections.Count;
        if (_detectorExecuted)
        {
            AddFinite(_inferenceMs, inferenceMs);
            AddFinite(_frameAgeMs, frameAgeMs);
        }
        AddFinite(_captureFps, captureFps);
        foreach (ArenaDetection detection in detections)
            AddFinite(_observationAges, detection.ObservationAgeMs);

        ArenaGroundTruth[] visibleTargets   = targets.Where(t =>  t.Visible).ToArray();
        ArenaGroundTruth[] invisibleTargets = targets.Where(t => !t.Visible).ToArray();

        // ── Occlusion state transitions ────────────────────────────────────────
        bool prevWasOccluded = _wasOccluded;
        bool anyInvisible    = invisibleTargets.Length > 0;

        if (anyInvisible)
        {
            _occlusionFrames++;
            if (!prevWasOccluded)
            {
                // Entering a new occlusion cycle.
                _occlusionCycleCount++;
                // If the previous cycle's reappearance was never matched before a new
                // occlusion started, that cycle was a failed reacquisition.
                if (_awaitingFirstReappearanceMatch)
                    _failedReacquisitions++;

                _occlusionStartedAt       = timestamp;
                _occlusionExceededPersistenceWindow = false;
                _oldTrackExpiredByDeadline = false;
                _currentCycleReacqFrames  = 0;

                // Freeze the current track ID for the target going invisible.
                _trackIdAtOcclusionStart = null;
                foreach (ArenaGroundTruth t in invisibleTargets)
                    if (_lastTrackByTarget.TryGetValue(t.Id, out int id))
                        _trackIdAtOcclusionStart = id;
            }
            _wasOccluded = true;
        }
        else if (prevWasOccluded)
        {
            // Exiting occlusion.
            _wasOccluded           = false;
            _reappearanceTimestamp = timestamp;

            // If the gap exceeded the persistence window, the old track must have expired
            // before the target reappeared. A new ID on reappearance is an expected lifecycle
            // event, not a tracker failure.
            if (_occlusionStartedAt.HasValue &&
                (timestamp - _occlusionStartedAt.Value).TotalSeconds > OcclusionPersistenceWindowSeconds)
            {
                _occlusionExceededPersistenceWindow = true;
            }

            _occlusionStartedAt             = null;
            _awaitingFirstReappearanceMatch = true;
            _countingReacquisitionFrames    = true;
        }

        // ── Ambiguity detection ────────────────────────────────────────────────
        bool frameIsAmbiguous = false;
        if (visibleTargets.Length >= 2)
        {
            const float BoxWidthPx = 60f;
            for (int a = 0; a < visibleTargets.Length && !frameIsAmbiguous; a++)
                for (int b = a + 1; b < visibleTargets.Length && !frameIsAmbiguous; b++)
                    if (Distance(visibleTargets[a].Center, visibleTargets[b].Center) < BoxWidthPx)
                        frameIsAmbiguous = true;
        }
        if (frameIsAmbiguous)
            _ambiguousIdentityFrames++;

        // ── Global one-to-one assignment ─────────────────────────────────────
        int[] assignment = HungarianAssign(visibleTargets, detections, _matchRadiusPixels);

        FrameSnap snap = new(_frames, timestamp, visibleTargets, detections.ToArray(), assignment);

        var matchedDetections = new HashSet<int>();
        for (int ti = 0; ti < visibleTargets.Length; ti++)
        {
            ArenaGroundTruth target = visibleTargets[ti];
            int di = assignment[ti];

            if (di < 0)
            {
                _misses++;
                if (_lastTrackByTarget.ContainsKey(target.Id) && !_lostAtByTarget.ContainsKey(target.Id))
                {
                    _lostAtByTarget[target.Id] = timestamp;
                    _trackLosses++;
                }
                if (_countingReacquisitionFrames)
                {
                    _currentCycleReacqFrames++;
                }
                continue;
            }

            ArenaDetection match = detections[di];
            matchedDetections.Add(di);
            double bestDistance = Distance(target.Center, match.Center);

            if (_lastTrackByTarget.TryGetValue(target.Id, out int previousTrack) && previousTrack != match.TrackId)
            {
                _identitySwitches++;
                if (!frameIsAmbiguous)
                {
                    _identitySwitchesOutsideAmbiguity++;

                    // Classify: expected reinitialization after confirmed expiry, or a true IS?
                    // _awaitingFirstReappearanceMatch is still true here — it's cleared below.
                    if (_awaitingFirstReappearanceMatch && _oldTrackExpiredByDeadline)
                        _expectedTrackReinitializations++;
                    else
                        _unexpectedIdentitySwitchesOutsideAmbiguity++;

                    // Diagnostic CSV window regardless of IS type.
                    var trace = new SwitchTrace
                    {
                        TargetId       = target.Id,
                        OldTrackId     = previousTrack,
                        NewTrackId     = match.TrackId,
                        SwitchFrameIndex = _frames,
                    };
                    foreach (FrameSnap s in _snapBuffer)
                        trace.Rows.Add((s, "before"));
                    trace.Rows.Add((snap, "switch"));
                    _pendingSwitchTraces.Add(trace);
                }
            }
            _lastTrackByTarget[target.Id] = match.TrackId;

            // First match after the target reappeared from occlusion.
            if (_awaitingFirstReappearanceMatch)
            {
                _awaitingFirstReappearanceMatch = false;
                _countingReacquisitionFrames    = false;
                _firstTrackIdAfterOcclusion     = match.TrackId;
                _maxReacquisitionFrames = Math.Max(_maxReacquisitionFrames, _currentCycleReacqFrames);
                _successfulReacquisitions++;
                if (_trackIdAtOcclusionStart.HasValue && _trackIdAtOcclusionStart.Value != match.TrackId)
                    _trackIdPreservedAfterOcclusion = false;
            }

            if (_lostAtByTarget.Remove(target.Id, out DateTime lostAt))
                _reacquisitionMs.Add(Math.Max(0, (timestamp - lostAt).TotalMilliseconds));
            if (match.IsExtrapolated)
                _predictiveErrors.Add(bestDistance);
            if (match.GruPredictedCenter is PointF prediction)
                _gruErrors.Add(Distance(target.Center, prediction));
        }

        // ── False-positive classification ─────────────────────────────────────
        // Unmatched detections are classified into one of four mutually exclusive buckets.
        // Only GhostAfterPersistenceDeadline, DuplicateAfterReappearance, and
        // UnrelatedFalsePositive count toward the FalsePositives gate.
        // ExpectedOcclusionPersistenceFrames are intentional extrapolation and are NOT an FP.
        int unmatched = detections.Count - matchedDetections.Count;
        if (anyInvisible
            && _trackIdAtOcclusionStart.HasValue
            && detections.Any(det => det.TrackId == _trackIdAtOcclusionStart.Value
                && det.ObservationAgeMs > OcclusionPersistenceWindowMs))
        {
            _occlusionExceededPersistenceWindow = true;
        }
        if (anyInvisible
            && _trackIdAtOcclusionStart.HasValue
            && _occlusionStartedAt.HasValue
            && (timestamp - _occlusionStartedAt.Value).TotalSeconds > OcclusionPersistenceWindowSeconds)
        {
            _occlusionExceededPersistenceWindow = true;
            _oldTrackExpiredByDeadline = detections.All(det => det.TrackId != _trackIdAtOcclusionStart.Value);
        }
        if (unmatched > 0)
        {
            foreach (int di in Enumerable.Range(0, detections.Count).Where(i => !matchedDetections.Contains(i)))
            {
                ArenaDetection det = detections[di];

                if (anyInvisible)
                {
                    // During occlusion: classify by track identity and observation age.
                    bool isOldOccludedTrack = _trackIdAtOcclusionStart.HasValue
                        && det.TrackId == _trackIdAtOcclusionStart.Value;

                    if (isOldOccludedTrack
                        && det.ObservationAgeMs <= OcclusionPersistenceWindowMs)
                    {
                        // Old track alive within its allowed persistence window — expected, not an FP.
                        // Note: BoundingBoxIsExtrapolated is only true for the first MaxLostSeconds/2;
                        // the track stays alive but stops updating its box for the second half of the
                        // window. ObsAge alone is the reliable test for "track within lifetime".
                        _expectedOcclusionPersistenceFrames++;
                    }
                    else if (isOldOccludedTrack
                        && det.ObservationAgeMs > OcclusionPersistenceWindowMs)
                    {
                        // Old track still emitted after its deadline — ghost.
                        _ghostAfterPersistenceDeadline++;
                        _falsePositives++;
                    }
                    else
                    {
                        // Unrelated: not the known occluded track.
                        _unrelatedFalsePositive++;
                        _falsePositives++;
                    }
                }
                else
                {
                    // Visible phase: check if this is a lingering old track emitted alongside the
                    // new track shortly after reappearance (duplicate within MaxLostSeconds).
                    bool isDuplicate = _trackIdAtOcclusionStart.HasValue
                        && det.TrackId == _trackIdAtOcclusionStart.Value
                        && _reappearanceTimestamp.HasValue
                        && (timestamp - _reappearanceTimestamp.Value).TotalSeconds
                            <= OcclusionPersistenceWindowSeconds;

                    if (isDuplicate)
                    {
                        _duplicateAfterReappearance++;
                        _falsePositives++;
                    }
                    else
                    {
                        _unrelatedFalsePositive++;
                        _falsePositives++;
                    }
                }
            }
        }

        // ── Switch trace ring buffer ──────────────────────────────────────────
        _snapBuffer.Enqueue(snap);
        while (_snapBuffer.Count > SwitchWindowBefore) _snapBuffer.Dequeue();

        for (int pi = _pendingSwitchTraces.Count - 1; pi >= 0; pi--)
        {
            SwitchTrace pt = _pendingSwitchTraces[pi];
            if (snap.FrameIndex <= pt.SwitchFrameIndex) continue;
            pt.Rows.Add((snap, "after"));
            pt.AfterRemaining--;
            if (pt.AfterRemaining <= 0)
            {
                FlushSwitchTrace(pt);
                _switchTracesWritten++;
                _pendingSwitchTraces.RemoveAt(pi);
            }
        }
    }

    private void FlushSwitchTrace(SwitchTrace trace)
    {
        if (SwitchTracePath == null) return;
        try
        {
            Directory.CreateDirectory(SwitchTracePath);
            string safeScenario = string.Concat(_scenario.Select(c => char.IsLetterOrDigit(c) ? c : '_'));
            string file = Path.Combine(SwitchTracePath,
                $"{safeScenario}_{trace.TargetId}_sw{_switchTracesWritten + 1}_old{trace.OldTrackId}_new{trace.NewTrackId}.csv");
            var sb = new StringBuilder();
            sb.AppendLine("FrameIndex,Role,Timestamp,TargetId,TargetX,TargetY,AssignedTrackId,TrackX,TrackY,DistancePx,IsExtrapolated,ObsAgeMs,GruX,GruY,GruErrPx");
            foreach ((FrameSnap s, string role) in trace.Rows)
            {
                string ts = s.Timestamp.ToString("o", System.Globalization.CultureInfo.InvariantCulture);
                for (int ti = 0; ti < s.VisibleTargets.Length; ti++)
                {
                    ArenaGroundTruth gt = s.VisibleTargets[ti];
                    int di = s.Assignment[ti];
                    if (di >= 0 && di < s.Detections.Length)
                    {
                        ArenaDetection d = s.Detections[di];
                        double dist = Distance(gt.Center, d.Center);
                        sb.AppendLine(string.Join(',',
                            s.FrameIndex, role, ts, gt.Id,
                            Fmt(gt.Center.X), Fmt(gt.Center.Y),
                            d.TrackId,
                            Fmt(d.Center.X), Fmt(d.Center.Y),
                            Fmt(dist), d.IsExtrapolated, Fmt(d.ObservationAgeMs),
                            d.GruPredictedCenter.HasValue ? Fmt(d.GruPredictedCenter.Value.X) : "",
                            d.GruPredictedCenter.HasValue ? Fmt(d.GruPredictedCenter.Value.Y) : "",
                            d.GruPredictedCenter.HasValue ? Fmt(Distance(gt.Center, d.GruPredictedCenter.Value)) : ""));
                    }
                    else
                    {
                        sb.AppendLine(string.Join(',',
                            s.FrameIndex, role, ts, gt.Id,
                            Fmt(gt.Center.X), Fmt(gt.Center.Y),
                            "MISS", "", "", "", "", "", "", "", ""));
                    }
                }
                var matched = new HashSet<int>(s.Assignment.Where(x => x >= 0));
                for (int di = 0; di < s.Detections.Length; di++)
                {
                    if (matched.Contains(di)) continue;
                    ArenaDetection d = s.Detections[di];
                    sb.AppendLine(string.Join(',',
                        s.FrameIndex, role, ts, "FP", "", "",
                        d.TrackId, Fmt(d.Center.X), Fmt(d.Center.Y),
                        "", d.IsExtrapolated, Fmt(d.ObservationAgeMs),
                        d.GruPredictedCenter.HasValue ? Fmt(d.GruPredictedCenter.Value.X) : "",
                        d.GruPredictedCenter.HasValue ? Fmt(d.GruPredictedCenter.Value.Y) : "",
                        ""));
                }
            }
            File.WriteAllText(file, sb.ToString());
        }
        catch (IOException) { /* Non-fatal: diagnostics should not crash the benchmark. */ }
    }

    private static string Fmt(double v) => v.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);

    public ScenarioMetricsSummary Summarize() => new(
        _scenario,
        _domain.ToString(),
        _detectorExecuted,
        _detectorMetricsGated,
        _trackerMetricsGated,
        NoiseProfile:                      _noiseConfig?.ProfileName ?? "N/A",
        InferenceMetricsApplicable:        _detectorExecuted,
        RandomSeed:                        _noiseConfig?.Seed ?? 0,
        NoiseDistribution:                 _noiseConfig != null ? "Uniform" : "N/A",
        PositionNoiseMaxPx:                _noiseConfig?.PositionNoiseMaxPx ?? 0,
        SizeNoisePercent:                  (_noiseConfig?.SizeNoise ?? 0) * 100,
        DropProbabilityPct:                (_noiseConfig?.DropProbability ?? 0) * 100,
        FalsePositivesPerFrame:            _noiseConfig?.FalsePositiveCount ?? 0,
        ContiguousOcclusionStartFrame:     _noiseConfig?.ContiguousOcclusionStartFrame ?? 0,
        ContiguousOcclusionDurationFrames: _noiseConfig?.ContiguousOcclusionDurationFrames ?? 0,
        Frames:                   _frames,
        DetectionCount:           _detections,
        FalsePositives:           _falsePositives,
        MissedDetections:         _misses,
        IdentitySwitches:         _identitySwitches,
        TrackLosses:              _trackLosses,
        MeanReacquisitionMs:      Mean(_reacquisitionMs),
        MeanPredictiveErrorPixels: Mean(_predictiveErrors),
        MeanGruErrorPixels:       Mean(_gruErrors),
        ObservationAgeP95Ms:      Percentile(_observationAges, 0.95),
        InferenceP50Ms:           Percentile(_inferenceMs, 0.50),
        InferenceP95Ms:           Percentile(_inferenceMs, 0.95),
        InferenceP99Ms:           Percentile(_inferenceMs, 0.99),
        FrameAgeP95Ms:            Percentile(_frameAgeMs, 0.95),
        CaptureFpsP50:            Percentile(_captureFps, 0.50),
        OcclusionFrames:                        _occlusionFrames,
        ExpectedOcclusionPersistenceFrames:     _expectedOcclusionPersistenceFrames,
        GhostAfterPersistenceDeadline:          _ghostAfterPersistenceDeadline,
        DuplicateAfterReappearance:             _duplicateAfterReappearance,
        UnrelatedFalsePositive:                 _unrelatedFalsePositive,
        TrackIdPreservedAfterOcclusion:         _trackIdPreservedAfterOcclusion,
        OcclusionExceededPersistenceWindow:     _occlusionExceededPersistenceWindow,
        OldTrackExpiredByDeadline:              _oldTrackExpiredByDeadline,
        OldTrackIdAtOcclusionStart:             _trackIdAtOcclusionStart ?? 0,
        NewTrackIdAtFirstReappearance:          _firstTrackIdAfterOcclusion ?? 0,
        IdentitySwitchesOutsideAmbiguity:       _identitySwitchesOutsideAmbiguity,
        ExpectedTrackReinitializations:         _expectedTrackReinitializations,
        UnexpectedIdentitySwitchesOutsideAmbiguity: _unexpectedIdentitySwitchesOutsideAmbiguity,
        AmbiguousIdentityFrames:                _ambiguousIdentityFrames,
        OcclusionCycleCount:                    _occlusionCycleCount,
        SuccessfulReacquisitions:               _successfulReacquisitions,
        FailedReacquisitions:                   _failedReacquisitions,
        MaxReacquisitionFrames:                 _maxReacquisitionFrames,
        SwitchTracesWritten:                    _switchTracesWritten);

    private sealed record FrameSnap(
        int FrameIndex,
        DateTime Timestamp,
        ArenaGroundTruth[] VisibleTargets,
        ArenaDetection[] Detections,
        int[] Assignment);

    private sealed class SwitchTrace
    {
        public string TargetId = "";
        public int OldTrackId;
        public int NewTrackId;
        public int SwitchFrameIndex;
        public List<(FrameSnap Snap, string Role)> Rows = new();
        public int AfterRemaining = SwitchWindowAfter;
    }

    /// <summary>
    /// Munkres (Hungarian) algorithm — O(n³) square-matrix variant.
    /// Returns assignment[ti] = detection index for each visible target (ti), or -1 if
    /// no detection is within <paramref name="matchRadius"/>.
    /// Ground-truth IDs are never passed to the tracker; this lives only in the evaluator.
    /// </summary>
    private static int[] HungarianAssign(
        ArenaGroundTruth[] targets,
        IReadOnlyList<ArenaDetection> detections,
        double matchRadius)
    {
        int n = targets.Length;
        int m = detections.Count;
        int[] result = new int[n];

        if (n == 0 || m == 0)
        {
            Array.Fill(result, -1);
            return result;
        }

        const double INF = 1e9;
        int sz = Math.Max(n, m);
        double[,] cost = new double[sz, sz];
        for (int i = 0; i < sz; i++)
            for (int j = 0; j < sz; j++)
                cost[i, j] = INF;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
            {
                double d = Distance(targets[i].Center, detections[j].Center);
                cost[i, j] = d <= matchRadius ? d : INF;
            }

        double[] u = new double[sz + 1];
        double[] v = new double[sz + 1];
        int[] p = new int[sz + 1];
        int[] way = new int[sz + 1];

        for (int i = 1; i <= sz; i++)
        {
            p[0] = i;
            int j0 = 0;
            double[] minVal = new double[sz + 1];
            bool[] used = new bool[sz + 1];
            Array.Fill(minVal, INF);
            bool augmentingPathFound = false;

            do
            {
                used[j0] = true;
                int i0 = p[j0], j1 = -1;
                double delta = INF;

                for (int j = 1; j <= sz; j++)
                {
                    if (used[j]) continue;
                    double cur = cost[i0 - 1, j - 1] - u[i0] - v[j];
                    if (cur < minVal[j])
                    {
                        minVal[j] = cur;
                        way[j] = j0;
                    }
                    if (minVal[j] < delta)
                    {
                        delta = minVal[j];
                        j1 = j;
                    }
                }

                if (j1 < 0) break;

                for (int j = 0; j <= sz; j++)
                {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else minVal[j] -= delta;
                }
                j0 = j1;

                if (p[j0] == 0) augmentingPathFound = true;
            }
            while (p[j0] != 0);

            if (!augmentingPathFound) continue;

            do
            {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            }
            while (j0 != 0);
        }

        int[] colForRow = new int[sz + 1];
        for (int j = 1; j <= sz; j++)
            if (p[j] > 0 && p[j] <= sz)
                colForRow[p[j]] = j;

        for (int i = 0; i < n; i++)
        {
            int di = colForRow[i + 1] - 1;
            if (di < 0 || di >= m || cost[i, di] >= INF)
                result[i] = -1;
            else
                result[i] = di;
        }

        return result;
    }

    private static double Distance(PointF a, PointF b)
    {
        double dx = a.X - b.X;
        double dy = a.Y - b.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private static void AddFinite(List<double> values, double value)
    {
        if (double.IsFinite(value) && value >= 0)
            values.Add(value);
    }

    private static double Mean(List<double> values) => values.Count == 0 ? 0 : values.Average();

    private static double Percentile(List<double> values, double quantile)
    {
        if (values.Count == 0)
            return 0;
        double[] ordered = values.Order().ToArray();
        int index = (int)Math.Round((ordered.Length - 1) * quantile, MidpointRounding.AwayFromZero);
        return ordered[Math.Clamp(index, 0, ordered.Length - 1)];
    }
}

public static class ScenarioReportWriter
{
    public static void Write(string outputDirectory, ScenarioMetricsSummary summary)
    {
        Directory.CreateDirectory(outputDirectory);
        string safeScenario = string.Concat(summary.Scenario.Select(character => char.IsLetterOrDigit(character) ? character : '_'));
        string stamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff");
        string stem = Path.Combine(outputDirectory, $"{stamp}_{safeScenario}");
        File.WriteAllText(stem + ".json", JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));
        var csv = new StringBuilder();
        csv.AppendLine(string.Join(',', typeof(ScenarioMetricsSummary).GetProperties().Select(property => property.Name)));
        csv.AppendLine(string.Join(',', typeof(ScenarioMetricsSummary).GetProperties().Select(property => Convert.ToString(property.GetValue(summary), System.Globalization.CultureInfo.InvariantCulture))));
        File.WriteAllText(stem + ".csv", csv.ToString());
    }
}
