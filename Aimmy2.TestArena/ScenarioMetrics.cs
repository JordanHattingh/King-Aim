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

public sealed record ScenarioMetricsSummary(
    string Scenario,
    int Frames,
    int DetectionCount,
    int FalsePositives,
    int MissedDetections,
    int IdentitySwitches,
    int TrackLosses,
    double MeanReacquisitionMs,
    double MeanPredictiveErrorPixels,
    double MeanGruErrorPixels,
    double ObservationAgeP95Ms,
    double InferenceP50Ms,
    double InferenceP95Ms,
    double InferenceP99Ms,
    double FrameAgeP95Ms,
    double CaptureFpsP50);

public sealed class ScenarioMetricsRecorder
{
    private readonly string _scenario;
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

    public ScenarioMetricsRecorder(string scenario, double matchRadiusPixels = 120)
    {
        _scenario = scenario;
        _matchRadiusPixels = matchRadiusPixels;
    }

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
        AddFinite(_inferenceMs, inferenceMs);
        AddFinite(_frameAgeMs, frameAgeMs);
        AddFinite(_captureFps, captureFps);
        foreach (ArenaDetection detection in detections)
            AddFinite(_observationAges, detection.ObservationAgeMs);

        var available = new HashSet<int>(Enumerable.Range(0, detections.Count));
        foreach (ArenaGroundTruth target in targets.Where(target => target.Visible))
        {
            int bestIndex = -1;
            double bestDistance = double.MaxValue;
            foreach (int index in available)
            {
                double distance = Distance(target.Center, detections[index].Center);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestIndex = index;
                }
            }

            if (bestIndex < 0 || bestDistance > _matchRadiusPixels)
            {
                _misses++;
                if (_lastTrackByTarget.ContainsKey(target.Id) && !_lostAtByTarget.ContainsKey(target.Id))
                {
                    _lostAtByTarget[target.Id] = timestamp;
                    _trackLosses++;
                }
                continue;
            }

            ArenaDetection match = detections[bestIndex];
            available.Remove(bestIndex);
            if (_lastTrackByTarget.TryGetValue(target.Id, out int previousTrack) && previousTrack != match.TrackId)
                _identitySwitches++;
            _lastTrackByTarget[target.Id] = match.TrackId;
            if (_lostAtByTarget.Remove(target.Id, out DateTime lostAt))
                _reacquisitionMs.Add(Math.Max(0, (timestamp - lostAt).TotalMilliseconds));
            if (match.IsExtrapolated)
                _predictiveErrors.Add(bestDistance);
            if (match.GruPredictedCenter is PointF prediction)
                _gruErrors.Add(Distance(target.Center, prediction));
        }
        _falsePositives += available.Count;
    }

    public ScenarioMetricsSummary Summarize() => new(
        _scenario, _frames, _detections, _falsePositives, _misses, _identitySwitches, _trackLosses,
        Mean(_reacquisitionMs), Mean(_predictiveErrors), Mean(_gruErrors), Percentile(_observationAges, 0.95),
        Percentile(_inferenceMs, 0.50), Percentile(_inferenceMs, 0.95), Percentile(_inferenceMs, 0.99),
        Percentile(_frameAgeMs, 0.95), Percentile(_captureFps, 0.50));

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
