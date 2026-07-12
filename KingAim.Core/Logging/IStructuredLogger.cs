namespace KingAim.Core.Logging;

/// <summary>
/// Offline, privacy-conscious structured logger.
/// Never saves screen frames by default.
/// All log data stays local.
/// </summary>
public interface IStructuredLogger
{
    void Log(LogEventKind kind, string message, IReadOnlyDictionary<string, object?>? fields = null);
    void LogError(string message, Exception? ex = null, IReadOnlyDictionary<string, object?>? fields = null);

    /// <summary>
    /// Starts a diagnostic recording session. Requires explicit user consent
    /// and records locally only. The recording indicator must be shown in the UI.
    /// </summary>
    Task StartDiagnosticRecordingAsync(string reason, CancellationToken ct = default);
    Task StopDiagnosticRecordingAsync();

    bool IsDiagnosticRecordingActive { get; }
}
