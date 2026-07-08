namespace Aimmy2.Gamepad
{
    /// <summary>
    /// Combines the player's own physical right-stick input with the assist controller's
    /// computed correction. When no target is locked (assistMagnitude == 0), the player's
    /// input passes through unchanged. As the assist correction grows (closer/urgent target),
    /// the player's own stick contribution is progressively suppressed so it can't fight the
    /// assist while it's actively settling onto a target.
    /// </summary>
    public static class StickBlender
    {
        /// <summary>
        /// DominanceThreshold is the assist output magnitude (0..1) at which the player's own
        /// input is fully suppressed. Below it, player input is scaled down linearly as assist
        /// magnitude grows; above it, only the assist output is used.
        /// </summary>
        public const float DominanceThreshold = 0.5f;

        public static (float RX, float RY) Blend(
            float playerX,
            float playerY,
            float assistX,
            float assistY)
        {
            float assistMagnitude = MathF.Sqrt(assistX * assistX + assistY * assistY);

            float playerWeight = DominanceThreshold <= 0f
                ? 0f
                : 1f - Math.Clamp(assistMagnitude / DominanceThreshold, 0f, 1f);

            float rx = Math.Clamp(playerX * playerWeight + assistX, -1f, 1f);
            float ry = Math.Clamp(playerY * playerWeight + assistY, -1f, 1f);

            return (rx, ry);
        }
    }
}
