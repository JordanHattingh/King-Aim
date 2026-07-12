using KingAim.Core.Diagnostics;
using KingAim.Core.Safety;

namespace KingAim.Core.AppHost;

/// <summary>
/// Manages the full pipeline lifecycle. One instance per application run.
/// </summary>
public interface IApplicationHost : IAsyncDisposable
{
    SafetyPolicy Safety     { get; }
    SystemHealthState Health { get; }
    bool IsRunning          { get; }

    /// <summary>
    /// Initialises all services, verifies the model, warms up the runtime,
    /// and begins the capture → inference → cue loop.
    /// </summary>
    Task StartAsync(PipelineDescriptor pipeline, CancellationToken ct = default);

    /// <summary>
    /// Gracefully stops the pipeline, flushing logs and closing the model session.
    /// </summary>
    Task StopAsync(CancellationToken ct = default);

    /// <summary>Immediately halts all output. Safe to call from any thread.</summary>
    void EmergencyStop();

    /// <summary>
    /// Replaces the active model without stopping the pipeline.
    /// The pipeline pauses for the duration of the swap.
    /// </summary>
    Task SwapModelAsync(string modelId, CancellationToken ct = default);
}
