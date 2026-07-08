namespace Aimmy2.Gamepad
{
    public sealed class GamepadAssistController
    {
        public float Gain { get; set; } = 1.5f;
        public float DeadbandRadius { get; set; } = 0.02f;
        public float MaxOutput { get; set; } = 1.0f;
        public float MaxSlewRate { get; set; } = 4.0f;
        public int MaxObservationAge { get; set; } = 10;

        private float _currentRx;
        private float _currentRy;

        public float RightStickX => _currentRx;
        public float RightStickY => _currentRy;

        public (float RightStickX, float RightStickY) Update(
            bool hasTarget,
            float errorX,
            float errorY,
            float targetVelocityX,
            float targetVelocityY,
            float confidence,
            int observationAgeSamples,
            double dtSeconds)
        {
            float desiredRx = 0f;
            float desiredRy = 0f;

            if (hasTarget && observationAgeSamples <= MaxObservationAge)
            {
                float radius = MathF.Sqrt(errorX * errorX + errorY * errorY);
                if (radius >= DeadbandRadius)
                {
                    desiredRx = Math.Clamp(Gain * errorX, -MaxOutput, MaxOutput);
                    desiredRy = Math.Clamp(Gain * errorY, -MaxOutput, MaxOutput);
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
        }
    }
}
