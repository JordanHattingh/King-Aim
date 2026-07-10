using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace InputLogic
{
    /// <summary>
    /// Neural Network 4: Mouse Movement MLP.
    ///
    /// Architecture: 8 → 64 (SiLU) → 64 (SiLU) → 32 (SiLU) → 2 (Tanh). ~8k parameters.
    /// Runs on CPU. Typical latency: &lt; 0.05 ms.
    ///
    /// Replaces Bezier curves in MouseManager.MoveCrosshair when loaded.
    /// Outputs velocity in [-1, +1] normalized units. The host scales by MaxVelocity and
    /// integrates: delta = velocity * dt, accumulating sub-pixel residuals to avoid drift.
    ///
    /// Input features:
    ///   [0] dx_pixels          — raw x distance to target
    ///   [1] dy_pixels          — raw y distance to target
    ///   [2] distance_pixels    — Euclidean distance to target
    ///   [3] current_speed      — magnitude of last emitted delta (pixels/ms)
    ///   [4] target_size_pixels — max(target_w, target_h)
    ///   [5] dt_seconds         — time since last movement update
    ///   [6] prev_vx            — previous output velocity x
    ///   [7] prev_vy            — previous output velocity y
    /// </summary>
    public sealed class MovementMlp : IDisposable
    {
        private InferenceSession? _session;
        private readonly float[] _inputBuf = new float[8];
        private float _prevVx, _prevVy;
        private float _residualX, _residualY;
        private bool _disposed;

        /// <summary>Maximum output speed in pixels per second. Tuned at runtime via MouseSensitivity.</summary>
        public float MaxVelocityPixPerSec { get; set; } = 1200f;

        public bool IsLoaded => _session != null;

        public void Load(string modelPath)
        {
            var prev = _session;
            var opts = new SessionOptions
            {
                EnableMemoryPattern = true,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = 1,
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

        public void Reset()
        {
            _prevVx = _prevVy = 0f;
            _residualX = _residualY = 0f;
        }

        /// <summary>
        /// Computes the integer pixel delta to emit this frame.
        /// Sub-pixel residuals are accumulated so no fractional motion is lost.
        /// Returns (0,0) and falls back to caller when the model is not loaded.
        /// </summary>
        public (int dx, int dy)? Move(
            float targetDx, float targetDy,
            float targetW, float targetH,
            float dtSeconds,
            float currentSpeedPixPerMs = 0f)
        {
            if (_session == null) return null;

            float dist = MathF.Sqrt(targetDx * targetDx + targetDy * targetDy);
            float size = MathF.Max(targetW, targetH);

            _inputBuf[0] = targetDx;
            _inputBuf[1] = targetDy;
            _inputBuf[2] = dist;
            _inputBuf[3] = currentSpeedPixPerMs;
            _inputBuf[4] = size;
            _inputBuf[5] = dtSeconds;
            _inputBuf[6] = _prevVx;
            _inputBuf[7] = _prevVy;

            var inputTensor = new DenseTensor<float>(_inputBuf, new[] { 1, 8 });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("features", inputTensor)
            };
            using var results = _session.Run(inputs);
            var output = results[0].AsEnumerable<float>().ToArray();
            if (output.Length < 2) return null;

            _prevVx = output[0];
            _prevVy = output[1];

            // Scale tanh output [-1,+1] to pixel/sec, integrate with dt, accumulate residuals.
            _residualX += output[0] * MaxVelocityPixPerSec * dtSeconds;
            _residualY += output[1] * MaxVelocityPixPerSec * dtSeconds;

            int emitX = (int)_residualX;
            int emitY = (int)_residualY;
            _residualX -= emitX;
            _residualY -= emitY;

            return (emitX, emitY);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _session?.Dispose();
        }
    }
}
