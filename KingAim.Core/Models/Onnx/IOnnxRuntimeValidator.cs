namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Validates that ORT can create a session, run warm-up inference, and collect
/// output statistics for a given model on a specified execution provider.
/// Separate from <see cref="IOnnxModelInspector"/>: the inspector reads graph
/// metadata from protobuf without loading the model; this validator proves the
/// model can actually execute on the target hardware.
/// </summary>
public interface IOnnxRuntimeValidator
{
    RuntimeCompatibilityReport Validate(
        string modelPath,
        RuntimeValidationOptions? options = null,
        CancellationToken cancellationToken = default);

    RuntimeCompatibilityReport ValidateBytes(
        byte[] modelBytes,
        RuntimeValidationOptions? options = null,
        string sourceName = "",
        CancellationToken cancellationToken = default);
}
