namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Validates that ORT can create a session for a given model and run a warm-up input.
/// Separate from <see cref="IOnnxModelInspector"/>: the inspector reads metadata without
/// loading the model; this validator proves the model can actually execute on the target EP.
/// </summary>
public interface IOnnxRuntimeValidator
{
    /// <summary>
    /// Attempt to open a session for <paramref name="modelPath"/> on the specified
    /// execution provider and run one warm-up pass with a zero-filled input tensor.
    /// </summary>
    RuntimeCompatibilityReport Validate(string modelPath, string executionProvider = "CPU");

    /// <summary>Same as <see cref="Validate(string,string)"/> but for in-memory bytes.</summary>
    RuntimeCompatibilityReport ValidateBytes(byte[] modelBytes, string executionProvider = "CPU",
        string sourceName = "");
}
