namespace KingAim.Core.Logging;

public enum LogEventKind
{
    AppStart,
    AppStop,
    ModelLoad,
    ModelChecksumVerified,
    HardwareIdentified,
    RuntimeProviderSelected,
    InferenceLatency,
    FrameDropped,
    TrackCreated,
    TrackLost,
    CueEmitted,
    ProfileChanged,
    Error,
    DiagnosticRecordingStarted,
    DiagnosticRecordingStopped,
}
