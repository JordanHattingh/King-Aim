using KingAim.Core.Preprocessing;

namespace KingAim.Core.Inference;

/// <summary>Input to a single model run.</summary>
public sealed class InferenceInput
{
    public float[] Tensor   { get; init; } = [];
    public long    FrameId  { get; init; }
    public PreprocessingMetadata Meta { get; init; } = new();
}

/// <summary>Raw output from a single model run, before decoding.</summary>
public sealed class InferenceOutput
{
    public float[][] Tensors      { get; init; } = [];    // one per output head
    public string[]  TensorNames  { get; init; } = [];
    public long      FrameId      { get; init; }
    public long      InferenceUs  { get; init; }          // inference duration in µs
    public PreprocessingMetadata Meta { get; init; } = new();
}

/// <summary>
/// Runtime abstraction. The pipeline depends on this, not on ONNX Runtime directly.
/// </summary>
public interface IInferenceEngine : IDisposable
{
    string RuntimeName { get; }   // "DirectML", "CPU", "Mock"

    /// <summary>Load the model at the given path and validate the input/output shapes.</summary>
    Task LoadModelAsync(string onnxPath, CancellationToken ct = default);

    /// <summary>Run a single warm-up inference to initialise GPU shaders.</summary>
    Task WarmUpAsync(CancellationToken ct = default);

    /// <summary>Run inference on a preprocessed frame.</summary>
    Task<InferenceOutput> RunAsync(InferenceInput input, CancellationToken ct = default);

    /// <summary>Median inference latency over recent frames, in milliseconds.</summary>
    double MedianLatencyMs { get; }

    bool IsLoaded { get; }
}
