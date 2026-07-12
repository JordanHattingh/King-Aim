namespace KingAim.Core.Diagnostics;

/// <summary>
/// Point-in-time diagnostics. Safe to display in the UI on any thread.
/// </summary>
public sealed class DiagnosticsSnapshot
{
    public SystemHealthState HealthState     { get; init; }
    public string  ActiveModelId            { get; init; } = "";
    public string  RuntimeProvider          { get; init; } = "";
    public string  GpuAdapterName          { get; init; } = "";

    public double  CaptureFps              { get; init; }
    public double  InferenceFps            { get; init; }
    public double  InferenceMedianMs       { get; init; }
    public double  InferenceP95Ms          { get; init; }
    public double  CaptureToInferenceMs    { get; init; }
    public double  CaptureToQueMs          { get; init; }   // end-to-end cue latency

    public long    DroppedFrames           { get; init; }
    public int     ActiveTracks            { get; init; }
    public int?    FocusTrackId            { get; init; }
    public float   ModelConfidence         { get; init; }

    public long    VramUsedBytes           { get; init; }
    public string? LastError               { get; init; }

    public DateTimeOffset Timestamp        { get; init; } = DateTimeOffset.UtcNow;
}
