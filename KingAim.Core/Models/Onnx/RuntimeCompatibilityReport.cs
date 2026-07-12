namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Per-output statistics collected during a timed inference pass.
/// </summary>
public sealed record RuntimeTensorReport(
    string  Name,
    string  ElementType,
    IReadOnlyList<TensorDimension> Dimensions,
    long    ElementCount,
    float   MinValue,
    float   MaxValue,
    float   MeanValue,
    int     NanCount,
    int     InfinityCount);

/// <summary>
/// Latency summary over the timed inference runs.
/// </summary>
public sealed record RuntimeLatencyReport(
    double MedianMs,
    double P99Ms,
    int    RunCount);

/// <summary>
/// Full result of attempting to create an ORT session and run inference.
/// Separate from <see cref="DecoderCompatibilityReport"/>: this validates whether
/// ORT can actually execute the model on the target EP, not whether the decoder
/// understands its tensor shapes.
/// </summary>
public sealed record RuntimeCompatibilityReport(
    bool     SessionCreated,
    bool     WarmupSucceeded,
    bool     ExecutionSucceeded,
    string   RequestedProvider,
    string   ActualProvider,
    IReadOnlyList<TensorContract>      Inputs,
    IReadOnlyList<RuntimeTensorReport> Outputs,
    RuntimeLatencyReport?              Latency,
    IReadOnlyList<string>              Errors,
    IReadOnlyList<string>              Warnings)
{
    public bool IsCompatible
        => SessionCreated && WarmupSucceeded && ExecutionSucceeded && Errors.Count == 0;

    // -------------------------------------------------------------------------
    // Factory helpers
    // -------------------------------------------------------------------------

    public static RuntimeCompatibilityReport SessionFailed(
        string requestedProvider, string error)
        => new(false, false, false, requestedProvider, requestedProvider,
               [], [], null, [error], []);

    public static RuntimeCompatibilityReport WarmupFailed(
        string requestedProvider,
        IReadOnlyList<TensorContract> inputs,
        string error,
        IReadOnlyList<string>? warnings = null)
        => new(true, false, false, requestedProvider, requestedProvider,
               inputs, [], null, [error], warnings ?? []);
}
