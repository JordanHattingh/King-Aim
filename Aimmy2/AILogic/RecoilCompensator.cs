namespace Aimmy2.AILogic
{
    /// <summary>
    /// Mouse-based recoil compensation. Applies an upward mouse nudge every frame
    /// while the player is firing. Strength and pattern are configurable.
    ///
    /// How it works:
    ///   - DetectFiring() returns true when the trigger condition is met
    ///     (left mouse button held, or right trigger > threshold on controller).
    ///   - Each Update() call returns a (dx, dy) delta to add to mouse output.
    ///   - dy is negative (upward) to counteract muzzle climb.
    ///
    /// The built-in pattern is a simple vertical ramp that matches the average
    /// CoD full-auto recoil: strong upward pull for the first ~10 shots, then
    /// a steady drift. Override PatternY[] with a custom per-gun array to get
    /// weapon-specific compensation.
    /// </summary>
    public sealed class RecoilCompensator
    {
        // Master enable — can be toggled from UI.
        public bool Enabled { get; set; } = true;

        // How many pixels upward to move per frame at full strength.
        public float Strength { get; set; } = 2.5f;

        // Horizontal drift correction (most CoD guns drift slightly right).
        public float HorizontalDrift { get; set; } = 0.3f;

        // Trigger threshold for right-trigger (0-1). Above this = firing.
        public float TriggerThreshold { get; set; } = 0.15f;

        // Optional per-gun pattern: normalised Y values (0-1), indexed by shot count.
        // If null, a built-in ramp is used. Set from UI / gun profile.
        public float[]? PatternY { get; set; }
        public float[]? PatternX { get; set; }

        private int _shotIndex;
        private bool _wasFiring;
        private int _framesSinceShot;

        // Built-in ramp: strong for first 8 shots, then steady.
        private static readonly float[] DefaultPatternY =
        [
            1.0f, 1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f, 0.7f,
            0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f,
            0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f,
            0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f
        ];

        private static readonly float[] DefaultPatternX =
        [
            0f, 0.05f, 0.1f, 0.05f, -0.05f, -0.1f, 0f, 0.05f,
            0.1f, 0.05f, 0f, -0.05f, 0f, 0.05f, 0f, -0.05f,
            0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f, 0f, 0f
        ];

        /// <summary>
        /// Call once per frame. isFiring should be true when the player is shooting.
        /// Returns (dx, dy) pixel delta to apply via MouseAimOutput.ApplyRaw().
        /// </summary>
        public (int dx, int dy) Update(bool isFiring)
        {
            if (!Enabled || !isFiring)
            {
                if (_wasFiring)
                {
                    // Brief cooldown before resetting shot index so burst fire
                    // continues the pattern rather than restarting from shot 0.
                    _framesSinceShot++;
                    if (_framesSinceShot > 8)
                    {
                        _shotIndex = 0;
                        _framesSinceShot = 0;
                    }
                }
                _wasFiring = false;
                return (0, 0);
            }

            _wasFiring = true;
            _framesSinceShot = 0;

            float[] py = PatternY ?? DefaultPatternY;
            float[] px = PatternX ?? DefaultPatternX;

            int idx = Math.Min(_shotIndex, py.Length - 1);
            _shotIndex++;

            float dy = -Strength * py[idx];             // negative = upward
            float dx = HorizontalDrift * px[idx];

            return ((int)Math.Round(dx), (int)Math.Round(dy));
        }

        public void Reset()
        {
            _shotIndex = 0;
            _wasFiring = false;
            _framesSinceShot = 0;
        }
    }
}
