using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Aimmy2.AILogic
{
    /// <summary>
    /// Neural Network 3: contextual confidence calibration MLP.
    ///
    /// Feature schema detection-context-v2 (exactly mirrored by train_calibration.py):
    ///   [0] logit(raw_confidence)
    ///   [1] log(w_norm * h_norm)
    ///   [2] log(h_norm / w_norm)
    ///   [3] radial distance from normalized screen centre / sqrt(0.5)
    ///   [4] clamp(frame_age_ms, 0, 500) / 100
    ///   [5] mean visibility of keypoints whose visibility is > 0.1, else zero
    /// </summary>
    public sealed class CalibrationMlp : IDisposable
    {
        public const int FeatureCount = 6;
        private const float MaximumFrameAgeMs = 500f;
        private const float FrameAgeScaleMs = 100f;
        private const float MaximumCenterRadius = 0.70710678f;

        private readonly object _sync = new();
        private InferenceSession? _session;
        private readonly float[] _inputBuf = new float[FeatureCount];
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

        /// <summary>Returns calibrated confidence, or raw confidence when no model is loaded.</summary>
        public float Calibrate(
            float rawConfidence,
            float wNorm,
            float hNorm,
            float cxNorm,
            float cyNorm,
            float frameAgeMs,
            PlayerKeypoints? keypoints = null)
        {
            lock (_sync)
            {
                if (_disposed || _session == null)
                    return rawConfidence;

                EncodeFeatures(
                    _inputBuf,
                    rawConfidence,
                    wNorm,
                    hNorm,
                    cxNorm,
                    cyNorm,
                    frameAgeMs,
                    keypoints);

                var inputTensor = new DenseTensor<float>(_inputBuf, new[] { 1, FeatureCount });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("features", inputTensor),
                };
                using var results = _session.Run(inputs);

                float calibrated = results[0].AsEnumerable<float>().FirstOrDefault(rawConfidence);
                return Math.Clamp(calibrated, 0f, 1f);
            }
        }

        internal static void EncodeFeatures(
            Span<float> destination,
            float rawConfidence,
            float wNorm,
            float hNorm,
            float cxNorm,
            float cyNorm,
            float frameAgeMs,
            PlayerKeypoints? keypoints = null)
        {
            if (destination.Length < FeatureCount)
                throw new ArgumentException($"Calibration feature buffer requires {FeatureCount} floats.", nameof(destination));

            float probability = Math.Clamp(rawConfidence, 1e-6f, 1f - 1e-6f);
            destination[0] = MathF.Log(probability / (1f - probability));
            destination[1] = MathF.Log(MathF.Max(wNorm * hNorm, 1e-10f));
            destination[2] = MathF.Log(MathF.Max(hNorm / MathF.Max(wNorm, 1e-5f), 1e-5f));

            float dx = cxNorm - 0.5f;
            float dy = cyNorm - 0.5f;
            destination[3] = MathF.Sqrt(dx * dx + dy * dy) / MaximumCenterRadius;
            destination[4] = Math.Clamp(frameAgeMs, 0f, MaximumFrameAgeMs) / FrameAgeScaleMs;
            destination[5] = ComputePoseQuality(keypoints);
        }

        public static float ComputePoseQualityStatic(PlayerKeypoints? keypoints) =>
            ComputePoseQuality(keypoints);

        private static float ComputePoseQuality(PlayerKeypoints? keypoints)
        {
            if (keypoints == null)
                return 0f;

            float sum = 0f;
            int count = 0;
            for (int i = 0; i < PlayerKeypoints.KeypointCount; i++)
            {
                Keypoint keypoint = keypoints[i];
                if (keypoint.Visibility > 0.1f)
                {
                    sum += keypoint.Visibility;
                    count++;
                }
            }
            return count > 0 ? sum / count : 0f;
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
            }
            previous?.Dispose();
        }
    }
}
