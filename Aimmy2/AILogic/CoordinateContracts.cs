namespace Aimmy2.AILogic
{
    /// <summary>
    /// Position expressed as a fraction of the active desktop/display bounds.
    /// Values are not clamped so callers can deliberately represent off-screen extrapolation.
    /// </summary>
    public readonly record struct ScreenFraction(float X, float Y)
    {
        public DesktopPixel ToDesktopPixel(int left, int top, int width, int height)
        {
            CoordinateContractGuards.ValidateDimensions(width, height);
            return new DesktopPixel(
                left + X * width,
                top + Y * height);
        }
    }

    /// <summary>Absolute desktop-pixel position.</summary>
    public readonly record struct DesktopPixel(float X, float Y)
    {
        public ScreenFraction ToScreenFraction(int left, int top, int width, int height)
        {
            CoordinateContractGuards.ValidateDimensions(width, height);
            return new ScreenFraction(
                (X - left) / width,
                (Y - top) / height);
        }
    }

    internal static class CoordinateContractGuards
    {
        internal static void ValidateDimensions(int width, int height)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), width, "Display width must be greater than zero.");
            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height), height, "Display height must be greater than zero.");
        }
    }
}
