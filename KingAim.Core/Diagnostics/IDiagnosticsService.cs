namespace KingAim.Core.Diagnostics;

public interface IDiagnosticsService
{
    DiagnosticsSnapshot Current { get; }

    /// <summary>Fired on the thread-pool whenever the snapshot changes.</summary>
    event Action<DiagnosticsSnapshot>? Updated;

    void RecordInferenceLatency(double ms);
    void RecordDroppedFrame();
    void RecordError(string message, Exception? ex = null);
    void SetHealthState(SystemHealthState state);
    void UpdateTrackStats(int activeTracks, int? focusTrackId, float modelConfidence);
}
