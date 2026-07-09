namespace Aimmy2.Gamepad
{
    public sealed class GamepadAssistController
    {
        /// <summary>Proportional gain applied when error is large (far from target).</summary>
        public float Gain { get; set; } = 2.5f;
        /// <summary>Proportional gain applied when error is small (nearly on-target). Prevents overshoot.</summary>
        public float GainNear { get; set; } = 1.0f;
        /// <summary>Integral gain — corrects steady-state lag on constant-velocity targets. Keep small to avoid windup.</summary>
        public float IntegralGain { get; set; } = 0.08f;
        /// <summary>Clamp on the integral accumulator to prevent windup when target is occluded for a long time.</summary>
        public float IntegralMax { get; set; } = 0.25f;
        public float DeadbandRadius { get; set; } = 0.01f;
        public float MaxOutput { get; set; } = 1.0f;
        public float MaxSlewRate { get; set; } = 8.0f;
        public int MaxObservationAge { get; set; } = 25;
        /// <summary>Scales the normalized target velocity added as feed-forward to lead moving targets.</summary>
        public float VelocityFeedForwardGain { get; set; } = 0.4f;
        /// <summary>EMA alpha for error smoothing: 0 = frozen, 1 = raw. Higher = more responsive, more jitter.</summary>
        public float SmoothingAlpha { get; set; } = 0.65f;
        /// <summary>Frames to apply the acquisition gain boost when a new target is first locked.</summary>
        public int AcquisitionBoostFrames { get; set; } = 6;
        /// <summary>Gain multiplier applied during the acquisition burst window. 1.0 = no boost.</summary>
        public float AcquisitionBoostFactor { get; set; } = 1.8f;
        /// <summary>Minimum detection confidence required for full assist. Below this the assist output is scaled down.</summary>
        public float MinConfidenceForFullAssist { get; set; } = 0.70f;

        private float _currentRx;
        private float _currentRy;
        private float _smoothedErrorX;
        private float _smoothedErrorY;
        private float _integralX;
        private float _integralY;
        private int? _lastTrackId;
        private int _acquisitionFramesRemaining;

        public float RightStickX => _currentRx;
        public float RightStickY => _currentRy;
        public int AcquisitionFramesRemaining => _acquisitionFramesRemaining;

        public (float RightStickX, float RightStickY) Update(
            bool hasTarget,
            float errorX,
            float errorY,
            float targetVelocityX,
            float targetVelocityY,
            float confidence,
            int observationAgeSamples,
            double dtSeconds,
            int? trackId = null)
        {
            float desiredRx = 0f;
            float desiredRy = 0f;

            bool newTargetAcquired = hasTarget && trackId.HasValue && trackId != _lastTrackId;
            if (!hasTarget)
            {
                _lastTrackId = null;
                _acquisitionFramesRemaining = 0;
                _smoothedErrorX *= 0.5f;
                _smoothedErrorY *= 0.5f;
                // Decay integral when no target — prevents stale windup on reacquisition.
                _integralX *= 0.8f;
                _integralY *= 0.8f;
            }
            else
            {
                if (newTargetAcquired)
                {
                    _lastTrackId = trackId;
                    _acquisitionFramesRemaining = AcquisitionBoostFrames;
                    // Clear integral on target switch — old integral is invalid for a new target.
                    _integralX = 0f;
                    _integralY = 0f;
                }
                else if (_acquisitionFramesRemaining > 0)
                {
                    _acquisitionFramesRemaining--;
                }

                _smoothedErrorX = _smoothedErrorX * (1f - SmoothingAlpha) + errorX * SmoothingAlpha;
                _smoothedErrorY = _smoothedErrorY * (1f - SmoothingAlpha) + errorY * SmoothingAlpha;

                // Integral accumulation with anti-windup clamp.
                if (dtSeconds > 0 && observationAgeSamples == 0)
                {
                    _integralX = Math.Clamp(_integralX + _smoothedErrorX * (float)dtSeconds, -IntegralMax, IntegralMax);
                    _integralY = Math.Clamp(_integralY + _smoothedErrorY * (float)dtSeconds, -IntegralMax, IntegralMax);
                }
            }

            if (hasTarget && observationAgeSamples <= MaxObservationAge)
            {
                float radius = MathF.Sqrt(_smoothedErrorX * _smoothedErrorX + _smoothedErrorY * _smoothedErrorY);
                if (radius >= DeadbandRadius)
                {
                    float adaptiveGain = GainNear + (Gain - GainNear) * Math.Clamp(radius, 0f, 1f);

                    if (_acquisitionFramesRemaining > 0 && AcquisitionBoostFrames > 0)
                    {
                        float burstFraction = (float)_acquisitionFramesRemaining / AcquisitionBoostFrames;
                        adaptiveGain *= 1f + (AcquisitionBoostFactor - 1f) * burstFraction;
                    }

                    // PID: proportional + integral (steady-state correction) + velocity feed-forward.
                    float px = adaptiveGain * _smoothedErrorX;
                    float py = adaptiveGain * _smoothedErrorY;
                    float ix = IntegralGain * _integralX;
                    float iy = IntegralGain * _integralY;
                    float ffx = VelocityFeedForwardGain * targetVelocityX;
                    float ffy = VelocityFeedForwardGain * targetVelocityY;

                    // Confidence scaling: low-confidence detections get proportionally weaker assist.
                    float confScale = MinConfidenceForFullAssist < 1f
                        ? Math.Clamp((confidence - 0f) / MinConfidenceForFullAssist, 0f, 1f)
                        : 1f;

                    desiredRx = Math.Clamp((px + ix + ffx) * confScale, -MaxOutput, MaxOutput);
                    desiredRy = Math.Clamp((py + iy + ffy) * confScale, -MaxOutput, MaxOutput);
                }
            }

            _currentRx = SlewTowards(_currentRx, desiredRx, dtSeconds);
            _currentRy = SlewTowards(_currentRy, desiredRy, dtSeconds);

            _currentRx = Math.Clamp(_currentRx, -1f, 1f);
            _currentRy = Math.Clamp(_currentRy, -1f, 1f);

            return (_currentRx, _currentRy);
        }

        private float SlewTowards(float current, float target, double dtSeconds)
        {
            if (dtSeconds <= 0)
                return current;

            float maxDelta = (float)(MaxSlewRate * dtSeconds);
            float delta = target - current;

            if (delta > maxDelta)
                delta = maxDelta;
            else if (delta < -maxDelta)
                delta = -maxDelta;

            return current + delta;
        }

        public void Reset()
        {
            _currentRx = 0f;
            _currentRy = 0f;
            _smoothedErrorX = 0f;
            _smoothedErrorY = 0f;
            _integralX = 0f;
            _integralY = 0f;
            _lastTrackId = null;
            _acquisitionFramesRemaining = 0;
        }
    }
}
