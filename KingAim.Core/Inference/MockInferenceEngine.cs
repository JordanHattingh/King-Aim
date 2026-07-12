using KingAim.Core.Preprocessing;

namespace KingAim.Core.Inference;

/// <summary>
/// Deterministic stand-in for integration tests. Returns a minimal InferenceOutput
/// with no actual tensor data. The MockPoseDecoder ignores the tensor contents.
/// </summary>
public sealed class MockInferenceEngine : IInferenceEngine
{
    public string RuntimeName => "Mock";
    public double MedianLatencyMs => 0.5;
    public bool   IsLoaded { get; private set; }

    public Task LoadModelAsync(string onnxPath, CancellationToken ct = default)
    {
        IsLoaded = true;
        return Task.CompletedTask;
    }

    public Task WarmUpAsync(CancellationToken ct = default) => Task.CompletedTask;

    public Task<InferenceOutput> RunAsync(InferenceInput input, CancellationToken ct = default)
    {
        var output = new InferenceOutput
        {
            FrameId     = input.FrameId,
            InferenceUs = 500L,
            Meta        = input.Meta,
            Tensors     = [],
            TensorNames = [],
        };
        return Task.FromResult(output);
    }

    public void Dispose() { }
}
