using KingAim.Core.Inference;
using KingAim.Core.Models.Onnx;
using KingAim.Core.Perception;
using KingAim.Core.Preprocessing;

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
    /// Checks whether this decoder can handle the given model contract.
    /// Should be called before the first inference to catch shape mismatches early.
    /// </summary>
    DecoderCompatibilityReport CheckCompatibility(OnnxModelContract contract);

    /// <summary>
    /// Validates that the output tensor names and shapes are compatible with this decoder.
    /// Throws if the contract is violated.
    /// </summary>
    void ValidateOutputContract(InferenceOutput output);

    /// <summary>
    /// Decodes raw tensors into normalized detections in source-frame pixel space.
    /// <paramref name="preprocessing"/> provides the inverse letterbox transform.
    /// </summary>
    IReadOnlyList<PoseDetection> Decode(InferenceOutput output, PreprocessingMetadata preprocessing);
}

[Obsolete("Use PoseDecoderOptions.")]
public sealed class NmsParameters
{
    public float ConfidenceThreshold { get; init; } = 0.25f;
    public float NmsIouThreshold     { get; init; } = 0.45f;
    public int   MaxDetections       { get; init; } = 300;

    public PoseDecoderOptions ToPoseDecoderOptions() => new()
    {
        DetectionConfidenceThreshold = ConfidenceThreshold,
        NmsIouThreshold              = NmsIouThreshold,
        MaximumDetections            = MaxDetections,
    };
}
