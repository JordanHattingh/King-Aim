namespace KingAim.Core.Diagnostics;

/// <summary>
/// Thread-safe diagnostics accumulator. Publishes an updated snapshot after each write.
/// </summary>
public sealed class DiagnosticsService : IDiagnosticsService
{
    private readonly object _lock = new();
    private readonly Queue<double> _latencies = new();
    private const int LatencyWindow = 60;

    private long   _droppedFrames;
    private int    _activeTracks;
    private int?   _focusTrackId;
    private float  _modelConfidence;
    private string _activeModelId = "";
    private string _runtimeProvider = "";
    private string? _lastError;
    private SystemHealthState _health = SystemHealthState.Healthy;

    public event Action<DiagnosticsSnapshot>? Updated;

    public DiagnosticsSnapshot Current { get; private set; } = new();

    public void RecordInferenceLatency(double ms)
    {
        lock (_lock)
        {
            _latencies.Enqueue(ms);
            if (_latencies.Count > LatencyWindow) _latencies.Dequeue();
        }
        Publish();
    }

    public void RecordDroppedFrame()
    {
        Interlocked.Increment(ref _droppedFrames);
        Publish();
    }

    public void RecordError(string message, Exception? ex = null)
    {
        lock (_lock) { _lastError = ex != null ? $"{message}: {ex.Message}" : message; }
        Publish();
    }

    public void SetHealthState(SystemHealthState state)
    {
        lock (_lock) { _health = state; }
        Publish();
    }

    /// <summary>Updates track summary fields (called by the pipeline after tracker.Update).</summary>
    public void UpdateTrackStats(int activeTracks, int? focusTrackId, float modelConfidence)
    {
        lock (_lock)
        {
            _activeTracks    = activeTracks;
            _focusTrackId    = focusTrackId;
            _modelConfidence = modelConfidence;
        }
        Publish();
    }

    public void SetActiveModel(string modelId, string provider)
    {
        lock (_lock) { _activeModelId = modelId; _runtimeProvider = provider; }
        Publish();
    }

    private void Publish()
    {
        DiagnosticsSnapshot snap;
        lock (_lock)
        {
            double median = 0, p95 = 0;
            if (_latencies.Count > 0)
            {
                var sorted = _latencies.OrderBy(x => x).ToList();
                median = sorted[sorted.Count / 2];
                p95    = sorted[(int)(sorted.Count * 0.95)];
            }

            snap = new DiagnosticsSnapshot
            {
                HealthState       = _health,
                ActiveModelId     = _activeModelId,
                RuntimeProvider   = _runtimeProvider,
                DroppedFrames     = _droppedFrames,
                ActiveTracks      = _activeTracks,
                FocusTrackId      = _focusTrackId,
                ModelConfidence   = _modelConfidence,
                InferenceMedianMs = median,
                InferenceP95Ms    = p95,
                LastError         = _lastError,
            };
        }
        Current = snap;
        ThreadPool.QueueUserWorkItem(_ => Updated?.Invoke(snap));
    }
}
