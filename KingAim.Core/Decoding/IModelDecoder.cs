using KingAim.Core.Inference;
using KingAim.Core.Perception;

namespace KingAim.Core.Decoding;

/// <summary>
/// Converts raw model output tensors into normalized PoseDetection objects.
/// Each model architecture gets its own implementation.
/// </summary>
public interface IModelDecoder
{
    /// <summary>Decoder identifier, must match ModelManifest.Decoder.</summary>
    string DecoderId { get; }

    /// <summary>
    /// Validates that the output tensor names and shapes are compatible with this decoder.
    /// Throws if the contract is violated.
    /// </summary>
    void ValidateOutputContract(InferenceOutput output);

    /// <summary>
    /// Decodes raw tensors into normalized detections in source-frame pixel space.
    /// </summary>
    IReadOnlyList<PoseDetection> Decode(InferenceOutput output, NmsParameters nms);
}

public sealed class NmsParameters
{
    public float ConfidenceThreshold { get; init; } = 0.25f;
    public float NmsIouThreshold     { get; init; } = 0.45f;
    public int   MaxDetections       { get; init; } = 300;
}
