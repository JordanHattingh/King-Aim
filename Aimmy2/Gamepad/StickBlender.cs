namespace Aimmy2.Gamepad
{
    /// <summary>
    /// Cooperative gamepad composition. The physical player's right-stick contribution is never
    /// suppressed. Vision/TestArena assistance is bounded down as physical intent grows so strong
    /// user input always has authority over the composed state.
    /// </summary>
    public static class StickBlender
    {
        /// <summary>Physical stick magnitude where assistance starts yielding to the user.</summary>
        public const float UserOverrideStart = 0.15f;

        /// <summary>Physical stick magnitude where assistance contribution reaches zero.</summary>
        public const float FullUserAuthority = 0.85f;

        public static (float RX, float RY) Blend(
            float playerX,
            float playerY,
            float assistX,
            float assistY)
        {
            playerX = Math.Clamp(playerX, -1f, 1f);
            playerY = Math.Clamp(playerY, -1f, 1f);
            assistX = Math.Clamp(assistX, -1f, 1f);
            assistY = Math.Clamp(assistY, -1f, 1f);

            float playerMagnitude = Math.Clamp(
                MathF.Sqrt(playerX * playerX + playerY * playerY),
                0f,
                1f);

            float assistWeight = 1f - SmoothStep(
                UserOverrideStart,
                FullUserAuthority,
                playerMagnitude);

            float rx = Math.Clamp(playerX + assistX * assistWeight, -1f, 1f);
            float ry = Math.Clamp(playerY + assistY * assistWeight, -1f, 1f);
            return (rx, ry);
        }

        private static float SmoothStep(float edge0, float edge1, float value)
        {
            if (edge1 <= edge0)
                return value >= edge1 ? 1f : 0f;

            float t = Math.Clamp((value - edge0) / (edge1 - edge0), 0f, 1f);
            return t * t * (3f - 2f * t);
        }
    }
}
