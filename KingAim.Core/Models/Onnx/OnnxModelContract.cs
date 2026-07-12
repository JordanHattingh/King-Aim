namespace KingAim.Core.Models.Onnx;

/// <summary>
/// The complete, inspected contract of a loaded ONNX model.
/// Captures everything needed to verify decoder compatibility and
/// to produce a deterministic, human-readable JSON report.
/// </summary>
public sealed record OnnxModelContract(
    string ModelPath,
    string Sha256,
    long   FileSizeBytes,
    int    OpsetVersion,
    IReadOnlyList<TensorContract> Inputs,
    IReadOnlyList<TensorContract> Outputs,
    IReadOnlyDictionary<string, string> Metadata)
{
    /// <summary>Primary input tensor (index 0). Null if the model has no inputs.</summary>
    public TensorContract? PrimaryInput  => Inputs.Count  > 0 ? Inputs[0]  : null;

    /// <summary>Primary output tensor (index 0). Null if the model has no outputs.</summary>
    public TensorContract? PrimaryOutput => Outputs.Count > 0 ? Outputs[0] : null;
}
