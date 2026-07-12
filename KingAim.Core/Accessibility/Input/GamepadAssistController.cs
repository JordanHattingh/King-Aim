namespace KingAim.Core.Accessibility.Input;

/// <summary>
/// Time-based bounded 2D controller for gamepad right-stick pointing assist.
/// All temporal configuration is expressed in seconds or milliseconds so behaviour is
/// independent of inference/update frequency.
/// </summary>
public sealed class GamepadAssistController : IGamepadAssistController
{
    public float Gain { get; set; } = 2.5f;
    public float GainNear { get; set; } = 1.0f;
    public float IntegralGain { get; set; } = 0.08f;
    public float IntegralMax { get; set; } = 0.25f;
    public float DeadbandRadius { get; set; } = 0.01f;
    public float MaxOutput { get; set; } = 1.0f;
    public float MaxSlewRate { get; set; } = 8.0f;

    public double MaxObservationAgeMs { get; set; } = 250.0;
    public float VelocityFeedForwardGain { get; set; } = 0.4f;

    /// <summary>First-order error response time. Zero disables smoothing.</summary>
    public double ErrorResponseTimeMs { get; set; } = 0.0;

    /// <summary>Duration of the initial acquisition gain boost.</summary>
    public double AcquisitionBoostDurationMs { get; set; } = 35.0;
    public float AcquisitionBoostFactor { get; set; } = 3.0f;
    public float MinConfidenceForFullAssist { get; set; } = 0.70f;

    public double NoTargetErrorDecayTimeMs { get; set; } = 50.0;
    public double NoTargetIntegralDecayTimeMs { get; set; } = 150.0;

    private float _currentRx;
    private float _currentRy;
    private float _smoothedErrorX;
    private float _smoothedErrorY;
    private float _integralX;
    private float _integralY;
    private int? _lastTrackId;
    private double _acquisitionBoostRemainingMs;

    public float RightStickX => _currentRx;
    public float RightStickY => _currentRy;
    public double AcquisitionBoostRemainingMs => _acquisitionBoostRemainingMs;

    public (float RightStickX, float RightStickY) Update(
        bool   hasTarget,
        float  errorX,
        float  errorY,
        float  targetVelocityX,
        float  targetVelocityY,
        float  confidence,
        double observationAgeMs,
        double dtSeconds,
        int?   trackId = null)
    {
        dtSeconds = Math.Clamp(dtSeconds, 0.0, 0.25);
        double dtMs = dtSeconds * 1000.0;
        float desiredRx = 0f;
        float desiredRy = 0f;

        bool newTargetAcquired = hasTarget && trackId.HasValue && trackId != _lastTrackId;
        if (!hasTarget)
        {
            _lastTrackId = null;
            _acquisitionBoostRemainingMs = 0;
            _smoothedErrorX = DecayTowardsZero(_smoothedErrorX, NoTargetErrorDecayTimeMs, dtSeconds);
            _smoothedErrorY = DecayTowardsZero(_smoothedErrorY, NoTargetErrorDecayTimeMs, dtSeconds);
            _integralX = DecayTowardsZero(_integralX, NoTargetIntegralDecayTimeMs, dtSeconds);
            _integralY = DecayTowardsZero(_integralY, NoTargetIntegralDecayTimeMs, dtSeconds);
        }
        else
        {
            if (newTargetAcquired)
            {
                _lastTrackId = trackId;
                _acquisitionBoostRemainingMs = Math.Max(0, AcquisitionBoostDurationMs);
                _integralX = 0f;
                _integralY = 0f;
            }

            float alpha = FirstOrderAlpha(ErrorResponseTimeMs, dtSeconds);
            _smoothedErrorX += (errorX - _smoothedErrorX) * alpha;
            _smoothedErrorY += (errorY - _smoothedErrorY) * alpha;

            bool observationFresh = observationAgeMs <= Math.Max(2.0, dtMs * 1.5);
            if (dtSeconds > 0 && observationFresh)
            {
                _integralX = Math.Clamp(
                    _integralX + _smoothedErrorX * (float)dtSeconds,
                    -IntegralMax,
                    IntegralMax);
                _integralY = Math.Clamp(
                    _integralY + _smoothedErrorY * (float)dtSeconds,
                    -IntegralMax,
                    IntegralMax);
            }
        }

        if (hasTarget && observationAgeMs <= MaxObservationAgeMs)
        {
            float radius = MathF.Sqrt(
                _smoothedErrorX * _smoothedErrorX +
                _smoothedErrorY * _smoothedErrorY);

            if (radius >= DeadbandRadius)
            {
                float adaptiveGain = GainNear +
                    (Gain - GainNear) * Math.Clamp(radius, 0f, 1f);

                if (_acquisitionBoostRemainingMs > 0 && AcquisitionBoostDurationMs > 0)
                {
                    float burstFraction = (float)Math.Clamp(
                        _acquisitionBoostRemainingMs / AcquisitionBoostDurationMs,
                        0.0,
                        1.0);
                    adaptiveGain *= 1f + (AcquisitionBoostFactor - 1f) * burstFraction;
                }

                float px = adaptiveGain * _smoothedErrorX;
                float py = adaptiveGain * _smoothedErrorY;
                float ix = IntegralGain * _integralX;
                float iy = IntegralGain * _integralY;
                float ffx = VelocityFeedForwardGain * targetVelocityX;
                float ffy = VelocityFeedForwardGain * targetVelocityY;

                float confScale = MinConfidenceForFullAssist < 1f
                    ? Math.Clamp(confidence / Math.Max(MinConfidenceForFullAssist, 1e-6f), 0f, 1f)
                    : 1f;

                desiredRx = Math.Clamp((px + ix + ffx) * confScale, -MaxOutput, MaxOutput);
                desiredRy = Math.Clamp((py + iy + ffy) * confScale, -MaxOutput, MaxOutput);
            }
        }

        _currentRx = SlewTowards(_currentRx, desiredRx, dtSeconds);
        _currentRy = SlewTowards(_currentRy, desiredRy, dtSeconds);
        _currentRx = Math.Clamp(_currentRx, -1f, 1f);
        _currentRy = Math.Clamp(_currentRy, -1f, 1f);

        _acquisitionBoostRemainingMs = Math.Max(0, _acquisitionBoostRemainingMs - dtMs);
        return (_currentRx, _currentRy);
    }

    private float SlewTowards(float current, float target, double dtSeconds)
    {
        if (dtSeconds <= 0)
            return current;

        float maxDelta = (float)(MaxSlewRate * dtSeconds);
        return current + Math.Clamp(target - current, -maxDelta, maxDelta);
    }

    private static float FirstOrderAlpha(double responseTimeMs, double dtSeconds)
    {
        if (responseTimeMs <= 0 || dtSeconds <= 0)
            return responseTimeMs <= 0 ? 1f : 0f;

        double tauSeconds = responseTimeMs / 1000.0;
        return (float)(1.0 - Math.Exp(-dtSeconds / tauSeconds));
    }

    private static float DecayTowardsZero(float value, double timeMs, double dtSeconds)
    {
        if (timeMs <= 0)
            return 0f;
        if (dtSeconds <= 0)
            return value;

        double tauSeconds = timeMs / 1000.0;
        return value * (float)Math.Exp(-dtSeconds / tauSeconds);
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
        _acquisitionBoostRemainingMs = 0;
    }
}
