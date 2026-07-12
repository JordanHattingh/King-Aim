using KingAim.Core.Inference;
using KingAim.Core.Models.Onnx;
using KingAim.Core.Perception;
using KingAim.Core.Preprocessing;

namespace KingAim.Core.Decoding.Mock;

/// <summary>
/// Returns a configurable set of detections without inspecting the raw tensors.
/// Kept for backwards compatibility; prefer <see cref="KingAim.Core.Decoding.MockPoseDecoder"/>.
/// </summary>
public sealed class MockPoseDecoder : IModelDecoder
{
    private readonly Func<InferenceOutput, IReadOnlyList<PoseDetection>> _factory;

    public string DecoderId => "mock-v1";

    public MockPoseDecoder(Func<InferenceOutput, IReadOnlyList<PoseDetection>>? factory = null)
    {
        _factory = factory ?? (_ => []);
    }

    public DecoderCompatibilityReport CheckCompatibility(OnnxModelContract contract)
        => DecoderCompatibilityReport.Compatible(DecoderId,
            observations: ["MockPoseDecoder.Mock accepts any contract."]);

    public void ValidateOutputContract(InferenceOutput output) { }

    public IReadOnlyList<PoseDetection> Decode(InferenceOutput output, PreprocessingMetadata preprocessing)
        => _factory(output);

    public static MockPoseDecoder WithDetections(params PoseDetection[] detections)
        => new(_ => detections);
}
