using KingAim.Core.Inference;
using KingAim.Core.Models.Onnx;
using KingAim.Core.Perception;
using KingAim.Core.Preprocessing;

namespace KingAim.Core.Decoding;

/// <summary>
/// Deterministic decoder for integration tests.
/// Returns a caller-supplied list of detections regardless of tensor contents.
/// The detection list can be swapped between frames via <see cref="SetDetections"/>.
/// </summary>
public sealed class MockPoseDecoder : IModelDecoder
{
    private IReadOnlyList<PoseDetection> _detections = [];

    public string DecoderId => "mock";

    public void SetDetections(IReadOnlyList<PoseDetection> detections)
        => _detections = detections;

    public void ClearDetections() => _detections = [];

    public DecoderCompatibilityReport CheckCompatibility(OnnxModelContract contract)
        => DecoderCompatibilityReport.Compatible(
            DecoderId,
            observations: ["MockPoseDecoder accepts any contract."]);

    public void ValidateOutputContract(InferenceOutput output) { }

    public IReadOnlyList<PoseDetection> Decode(InferenceOutput output, PreprocessingMetadata preprocessing)
        => _detections;
}
