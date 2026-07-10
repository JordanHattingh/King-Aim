using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Aimmy2.AILogic
{
    /// <summary>
    /// Neural Network 2: GRU-64 trajectory predictor.
    ///
    /// Architecture: 1-layer GRU, input_size=8, hidden=64, head 64→32→4 (SiLU).
    /// Training target: [Δcx*2, Δcy*2, (Δcx*2)/dt, (Δcy*2)/dt]. The factor-of-two
    /// comes from the centred coordinate transform used by train_gru.py. Runtime therefore
    /// multiplies predicted deltas by 0.5 before adding them to 0..1 screen fractions.
    ///
    /// Runs on the CPU execution provider. The class serializes session access so a model hot-swap
    /// cannot dispose an InferenceSession while Predict() is using it.
    /// </summary>
    public sealed class TemporalPredictor : IDisposable
    {
        public const float TrainingDeltaScale = 2f;
        public const float RuntimeDeltaInverseScale = 1f / TrainingDeltaScale;

        private readonly object _sync = new();
        private InferenceSession? _session;
        private readonly float[] _inputBuf = new float[TrackRingBuffer.Capacity * TrackRingBuffer.FeatureCount];
        private bool _disposed;

        public bool IsLoaded
        {
            get
            {
                lock (_sync)
                    return !_disposed && _session != null;
            }
        }

        public void Load(string modelPath)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

            var opts = new SessionOptions
            {
                EnableMemoryPattern = true,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = 2,
            };
            opts.AppendExecutionProvider_CPU();

            var candidate = new InferenceSession(modelPath, opts);
            InferenceSession? previous;
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                previous = _session;
                _session = candidate;
            }
            previous?.Dispose();
        }

        public void Unload()
        {
            InferenceSession? previous;
            lock (_sync)
            {
                previous = _session;
                _session = null;
            }
            previous?.Dispose();
        }

        /// <summary>
        /// Runs GRU inference. Returns model-space [Δcx*2, Δcy*2, vx*2, vy*2],
        /// or null when no model is loaded or the buffer has fewer than eight observations.
        /// </summary>
        public (float DeltaCxNorm, float DeltaCyNorm, float VxNorm, float VyNorm)?
            Predict(TrackRingBuffer buffer, GruNormConstants norm)
        {
            ArgumentNullException.ThrowIfNull(buffer);
            ArgumentNullException.ThrowIfNull(norm);
            if (!buffer.IsReady)
                return null;

            norm.Validate();

            lock (_sync)
            {
                if (_disposed || _session == null)
                    return null;

                buffer.FillSequence(_inputBuf, norm);

                var inputTensor = new DenseTensor<float>(
                    _inputBuf,
                    new[] { 1, TrackRingBuffer.Capacity, TrackRingBuffer.FeatureCount });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("sequence", inputTensor),
                };

                using var results = _session.Run(inputs);
                float[] output = results[0].AsEnumerable<float>().Take(4).ToArray();
                if (output.Length < 4)
                    return null;

                return (output[0], output[1], output[2], output[3]);
            }
        }

        public static float ConvertModelDeltaToScreenFraction(float modelDelta) =>
            modelDelta * RuntimeDeltaInverseScale;

        public void Dispose()
        {
            InferenceSession? previous;
            lock (_sync)
            {
                if (_disposed)
                    return;
                _disposed = true;
                previous = _session;
                _session = null;
            }
            previous?.Dispose();
        }
    }
}
