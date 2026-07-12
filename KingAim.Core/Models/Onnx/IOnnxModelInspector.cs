namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Reads ONNX session metadata and produces a verifiable, serialisable contract.
/// The inspector must not modify any file or run any inference.
/// </summary>
public interface IOnnxModelInspector
{
    /// <summary>Inspect a model file on disk.</summary>
    OnnxModelContract Inspect(string modelPath);

    /// <summary>Inspect a model from an in-memory byte array (e.g. for tests).</summary>
    OnnxModelContract InspectBytes(byte[] modelBytes, string sourceName = "");
}
