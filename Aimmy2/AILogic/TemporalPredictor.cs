using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Aimmy2.AILogic
{
    /// <summary>
    /// Neural Network 2: GRU-64 trajectory predictor.
    ///
    /// Architecture: 1-layer GRU, input_size=8, hidden=64, head 64→32→4 (SiLU).
    /// Output: [Δcx, Δcy, vx, vy] in normalized screen-fraction units per second.
    ///
    /// Runs on CPU (ORT CPUExecutionProvider). Typical latency: 0.02–0.15 ms.
    /// Replaces KalmanPrediction for the locked target once the buffer has 8 frames.
    /// Falls back to Kalman when:
    ///   - no model loaded
    ///   - ring buffer not yet full (< 8 observations)
    /// </summary>
    public sealed class TemporalPredictor : IDisposable
    {
        private InferenceSession? _session;
        private readonly float[] _inputBuf = new float[TrackRingBuffer.Capacity * TrackRingBuffer.FeatureCount];
        private bool _disposed;

        public bool IsLoaded => _session != null;

        public void Load(string modelPath)
        {
            var prev = _session;
            var opts = new SessionOptions
            {
                EnableMemoryPattern = true,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = 2,
            };
            opts.AppendExecutionProvider_CPU();
            _session = new InferenceSession(modelPath, opts);
            prev?.Dispose();
        }

        public void Unload()
        {
            _session?.Dispose();
            _session = null;
        }

        /// <summary>
        /// Runs GRU inference. Returns predicted (Δcx, Δcy, vx, vy) in normalized units,
        /// or null if the model is not loaded or the buffer is not ready.
        /// </summary>
        public (float DeltaCxNorm, float DeltaCyNorm, float VxNorm, float VyNorm)?
            Predict(TrackRingBuffer buffer, GruNormConstants norm)
        {
            if (_session == null || !buffer.IsReady) return null;

            buffer.FillSequence(_inputBuf, norm);

            var inputTensor = new DenseTensor<float>(
                _inputBuf,
                new[] { 1, TrackRingBuffer.Capacity, TrackRingBuffer.FeatureCount });

            using var inputs = new DisposableNamedOnnxValueList
            {
                NamedOnnxValue.CreateFromTensor("sequence", inputTensor)
            };

            using var results = _session.Run(inputs);
            var output = results[0].AsEnumerable<float>().ToArray();

            if (output.Length < 4) return null;
            return (output[0], output[1], output[2], output[3]);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _session?.Dispose();
        }

        private sealed class DisposableNamedOnnxValueList
            : List<NamedOnnxValue>, IDisposable
        {
            public void Dispose() { }
        }
    }
}
