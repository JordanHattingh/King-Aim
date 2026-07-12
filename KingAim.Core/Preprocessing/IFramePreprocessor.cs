using KingAim.Core.Capture;

namespace KingAim.Core.Preprocessing;

/// <summary>Result of preprocessing a single frame.</summary>
public sealed class PreprocessedFrame
{
    /// <summary>Float32 tensor in NCHW layout [1, 3, H, W], values in [0, 1].</summary>
    public float[]              Tensor   { get; init; } = [];
    public PreprocessingMetadata Meta    { get; init; } = new();
    public long                 FrameId { get; init; }
}

/// <summary>
/// Converts raw captured frames to model-ready tensors.
/// Shared across all models where inputs are compatible.
/// </summary>
public interface IFramePreprocessor
{
    /// <summary>
    /// Applies letterbox resize, colour conversion, normalisation, and NCHW layout.
    /// Returns the float tensor and the metadata needed to reverse the transform.
    /// </summary>
    PreprocessedFrame Preprocess(CapturedFrame frame);
}
