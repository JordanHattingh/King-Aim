namespace KingAim.Core.Inference.Mock;

/// <summary>
/// Deterministic mock inference engine for pipeline integration tests.
/// Returns a configurable fixed output without loading any real model.
/// </summary>
public sealed class MockInferenceEngine : IInferenceEngine
{
    private readonly Func<InferenceInput, float[][]> _outputFactory;
    private readonly string[]                         _outputNames;
    private bool _loaded;
    private bool _disposed;

    /// <param name="outputFactory">
    /// Produces the raw output tensors given the input.
    /// Defaults to a single empty tensor (no detections).
    /// </param>
    public MockInferenceEngine(
        Func<InferenceInput, float[][]>? outputFactory = null,
        string[]? outputNames = null)
    {
        _outputFactory = outputFactory ?? (_ => [[]]);
        _outputNames   = outputNames   ?? ["output0"];
    }

    public string RuntimeName     => "Mock";
    public bool   IsLoaded        => _loaded;
    public double MedianLatencyMs => 0.1;

    public Task LoadModelAsync(string onnxPath, CancellationToken ct = default)
    {
        _loaded = true;
        return Task.CompletedTask;
    }

    public Task WarmUpAsync(CancellationToken ct = default) => Task.CompletedTask;

    public Task<InferenceOutput> RunAsync(InferenceInput input, CancellationToken ct = default)
    {
        ThrowIfDisposed();

        var tensors = _outputFactory(input);
        var output  = new InferenceOutput
        {
            Tensors      = tensors,
            TensorNames  = _outputNames,
            FrameId      = input.FrameId,
            InferenceUs  = 100,
            Meta         = input.Meta,
        };
        return Task.FromResult(output);
    }

    public void Dispose() => _disposed = true;

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MockInferenceEngine));
    }
}
