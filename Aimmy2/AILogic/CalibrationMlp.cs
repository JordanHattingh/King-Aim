using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Aimmy2.AILogic
{
    /// <summary>
    /// Neural Network 3: Confidence Calibration MLP.
    ///
    /// Architecture: 6 → 16 (SiLU) → 8 (SiLU) → 1 (Sigmoid). ~300 parameters.
    /// Runs on CPU. Typical latency: &lt; 0.02 ms.
    ///
    /// Input features (all pre-normalized at training time, apply same transform here):
    ///   [0] logit(raw_confidence)      = log(p / (1-p))
    ///   [1] log(box_area_fraction)     = log(w_norm * h_norm)
    ///   [2] log(aspect_ratio)          = log(h_norm / w_norm)
    ///   [3] radial_dist_from_center    = sqrt((cx-0.5)^2 + (cy-0.5)^2) / 0.707
    ///   [4] frame_age_ms / 100
    ///   [5] pose_quality               = mean visibility of visible keypoints (0 if no pose)
    ///
    /// Output: calibrated confidence in [0, 1].
    /// Falls back to raw confidence when not loaded.
    /// </summary>
    public sealed class CalibrationMlp : IDisposable
    {
        private InferenceSession? _session;
        private readonly float[] _inputBuf = new float[6];
        private bool _disposed;

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

        /// <summary>
        /// Returns calibrated confidence, or rawConfidence if the model is not loaded.
        /// </summary>
        public float Calibrate(
            float rawConfidence,
            float wNorm, float hNorm,
            float cxNorm, float cyNorm,
            float frameAgeMs,
            PlayerKeypoints? keypoints = null)
        {
            if (_session == null) return rawConfidence;

            float p = Math.Clamp(rawConfidence, 1e-6f, 1f - 1e-6f);
            _inputBuf[0] = MathF.Log(p / (1f - p));
            _inputBuf[1] = MathF.Log(MathF.Max(wNorm * hNorm, 1e-10f));
            _inputBuf[2] = MathF.Log(MathF.Max(hNorm / MathF.Max(wNorm, 1e-5f), 1e-5f));
            float dx = cxNorm - 0.5f, dy = cyNorm - 0.5f;
            _inputBuf[3] = MathF.Sqrt(dx * dx + dy * dy) / 0.7071f;
            _inputBuf[4] = Math.Clamp(frameAgeMs, 0f, 500f) / 100f;
            _inputBuf[5] = ComputePoseQuality(keypoints);

            var inputTensor = new DenseTensor<float>(_inputBuf, new[] { 1, 6 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("features", inputTensor)
            };
            using var results = _session.Run(inputs);

            float cal = results[0].AsEnumerable<float>().FirstOrDefault(rawConfidence);
            return Math.Clamp(cal, 0f, 1f);
        }

        private static float ComputePoseQuality(PlayerKeypoints? kpts)
        {
            if (kpts == null) return 0f;
            float sum = 0f;
            int count = 0;
            for (int i = 0; i < PlayerKeypoints.KeypointCount; i++)
            {
                var kp = kpts[i];
                if (kp.Visibility > 0.1f) { sum += kp.Visibility; count++; }
            }
            return count > 0 ? sum / count : 0f;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _session?.Dispose();
        }
    }
}
