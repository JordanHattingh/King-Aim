using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace InputLogic
{
    /// <summary>
    /// Neural Network 4: controlled accessibility/TestArena pointing movement MLP.
    ///
    /// Feature schema pointing-velocity-v1 (exactly mirrored by train_movement.py):
    /// [dx_pixels, dy_pixels, distance_pixels, current_speed_pixels_per_ms,
    ///  target_size_pixels, dt_seconds, previous_output_vx, previous_output_vy].
    ///
    /// Output is normalized velocity [-1,+1]. The host scales to pixels/second, integrates with
    /// dt, and carries fractional residuals. A context change resets previous velocity and residuals
    /// so state from one pointing target cannot leak into the next task/target.
    /// </summary>
    public sealed class MovementMlp : IDisposable
    {
        public const int FeatureCount = 8;

        private readonly object _sync = new();
        private InferenceSession? _session;
        private readonly float[] _inputBuf = new float[FeatureCount];
        private float _prevVx;
        private float _prevVy;
        private float _residualX;
        private float _residualY;
        private int? _contextId;
        private bool _disposed;

        /// <summary>Maximum normalized-output scale in pixels per second.</summary>
        public float MaxVelocityPixPerSec { get; set; } = 1200f;

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
                IntraOpNumThreads = 1,
            };
            opts.AppendExecutionProvider_CPU();

            var candidate = new InferenceSession(modelPath, opts);
            InferenceSession? previous;
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                previous = _session;
                _session = candidate;
                ResetStateLocked();
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
                ResetStateLocked();
            }
            previous?.Dispose();
        }

        /// <summary>
        /// Sets the identity of the current generic pointing task/target. State is reset whenever
        /// identity changes, including a transition to null on target/task loss.
        /// </summary>
        public void SetContext(int? contextId)
        {
            lock (_sync)
            {
                if (_contextId == contextId)
                    return;

                _contextId = contextId;
                ResetMotionStateLocked();
            }
        }

        public void Reset()
        {
            lock (_sync)
                ResetStateLocked();
        }

        /// <summary>
        /// Computes the integer pixel delta for a controlled pointing update. Returns null when the
        /// model is not loaded. Call SetContext when task/target identity changes.
        /// </summary>
        public (int dx, int dy)? Move(
            float targetDx,
            float targetDy,
            float targetW,
            float targetH,
            float dtSeconds,
            float currentSpeedPixPerMs = 0f)
        {
            lock (_sync)
            {
                if (_disposed || _session == null)
                    return null;

                EncodeFeatures(
                    _inputBuf,
                    targetDx,
                    targetDy,
                    targetW,
                    targetH,
                    dtSeconds,
                    currentSpeedPixPerMs,
                    _prevVx,
                    _prevVy);

                var inputTensor = new DenseTensor<float>(_inputBuf, new[] { 1, FeatureCount });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("features", inputTensor),
                };
                using var results = _session.Run(inputs);
                float[] output = results[0].AsEnumerable<float>().Take(2).ToArray();
                if (output.Length < 2)
                    return null;

                _prevVx = output[0];
                _prevVy = output[1];

                float boundedDt = Math.Clamp(dtSeconds, 0.001f, 0.1f);
                _residualX += output[0] * MaxVelocityPixPerSec * boundedDt;
                _residualY += output[1] * MaxVelocityPixPerSec * boundedDt;

                int emitX = (int)MathF.Truncate(_residualX);
                int emitY = (int)MathF.Truncate(_residualY);
                _residualX -= emitX;
                _residualY -= emitY;

                return (emitX, emitY);
            }
        }

        internal static void EncodeFeatures(
            Span<float> destination,
            float targetDx,
            float targetDy,
            float targetW,
            float targetH,
            float dtSeconds,
            float currentSpeedPixPerMs,
            float previousVx,
            float previousVy)
        {
            if (destination.Length < FeatureCount)
                throw new ArgumentException($"Movement feature buffer requires {FeatureCount} floats.", nameof(destination));

            destination[0] = targetDx;
            destination[1] = targetDy;
            destination[2] = MathF.Sqrt(targetDx * targetDx + targetDy * targetDy);
            destination[3] = currentSpeedPixPerMs;
            destination[4] = MathF.Max(targetW, targetH);
            destination[5] = dtSeconds;
            destination[6] = previousVx;
            destination[7] = previousVy;
        }

        private void ResetStateLocked()
        {
            _contextId = null;
            ResetMotionStateLocked();
        }

        private void ResetMotionStateLocked()
        {
            _prevVx = 0f;
            _prevVy = 0f;
            _residualX = 0f;
            _residualY = 0f;
        }

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
                ResetStateLocked();
            }
            previous?.Dispose();
        }
    }
}
