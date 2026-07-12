using KingAim.Core.Inference;
using KingAim.Core.Perception;

namespace KingAim.Core.Decoding.Mock;

/// <summary>
/// Returns a configurable set of detections without inspecting the raw tensors.
/// Used for integration tests and for building the full pipeline before ONNX export.
/// </summary>
public sealed class MockPoseDecoder : IModelDecoder
{
    private readonly Func<InferenceOutput, IReadOnlyList<PoseDetection>> _factory;

    public string DecoderId => "mock-v1";

    /// <param name="factory">
    /// Produces detections for a given inference output.
    /// Defaults to returning an empty list (no detections).
    /// </param>
    public MockPoseDecoder(Func<InferenceOutput, IReadOnlyList<PoseDetection>>? factory = null)
    {
        _factory = factory ?? (_ => []);
    }

    public void ValidateOutputContract(InferenceOutput output) { /* Mock: always passes. */ }

    public IReadOnlyList<PoseDetection> Decode(InferenceOutput output, NmsParameters nms)
        => _factory(output);

    // Convenience: create a decoder that always returns a fixed list.
    public static MockPoseDecoder WithDetections(params PoseDetection[] detections)
        => new(_ => detections);
}
