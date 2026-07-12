namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Parameters for a single <see cref="IOnnxRuntimeValidator"/> validation run.
/// </summary>
public sealed record RuntimeValidationOptions
{
    public static readonly RuntimeValidationOptions DefaultCpu = new();
    public static readonly RuntimeValidationOptions DefaultDml = new() { RequestedProvider = "DirectML" };

    /// <summary>"CPU" or "DirectML".</summary>
    public string RequestedProvider { get; init; } = "CPU";

    /// <summary>Number of warm-up inference passes (not timed, not included in latency).</summary>
    public int WarmupRuns { get; init; } = 1;

    /// <summary>Number of timed inference passes used to compute latency percentiles.</summary>
    public int TimedRuns { get; init; } = 3;

    /// <summary>
    /// Maps symbolic dimension names (e.g. "batch_size") to concrete sizes used when
    /// building warm-up input tensors. Every dynamic dimension must be bound explicitly.
    /// </summary>
    public IReadOnlyDictionary<string, int>? DynamicDimensionBindings { get; init; }

    /// <summary>Maximum number of generated elements permitted across a single input tensor.</summary>
    public int MaximumGeneratedInputElements { get; init; } = 16_777_216;
}
