using KingAim.Core.Inference;
using KingAim.Core.Perception;

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

    public void ValidateOutputContract(InferenceOutput output) { }

    public IReadOnlyList<PoseDetection> Decode(InferenceOutput output, NmsParameters nms)
        => _detections;
}
