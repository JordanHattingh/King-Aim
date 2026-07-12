namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Result of attempting to create an ORT session and run a warm-up inference.
/// Distinct from <see cref="DecoderCompatibilityReport"/>: this validates whether
/// ORT can actually execute the model, not whether the decoder understands its shape.
/// </summary>
public sealed record RuntimeCompatibilityReport(
    bool     SessionCreated,
    bool     WarmupSucceeded,
    string   ExecutionProvider,
    IReadOnlyList<TensorContract> ActualInputs,
    IReadOnlyList<TensorContract> ActualOutputs,
    IReadOnlyList<string> Errors,
    IReadOnlyList<string> Warnings)
{
    public bool IsCompatible => SessionCreated && WarmupSucceeded && Errors.Count == 0;

    public static RuntimeCompatibilityReport SessionFailed(string provider, string error)
        => new(false, false, provider,
               [], [],
               [error], []);
}
