using MouseMovementLibraries.SendInputSupport;
using System.Runtime.InteropServices;

namespace Aimmy2.AILogic
{
    /// <summary>
    /// Converts normalised aim-assist error (-1..+1) into real mouse movement.
    /// Supports SendInput (default), mouse_event, and a passthrough-only mode.
    /// Sensitivity scales the error into pixel deltas — tune it to match your
    /// in-game sensitivity so 1.0 error = one full FOV-radius of mouse movement.
    /// </summary>
    public sealed class MouseAimOutput
    {
        private const uint MOUSEEVENTF_MOVE = 0x0001;

        [DllImport("user32.dll")]
        private static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, int dwExtraInfo);

        // How many pixels to move for a fully-saturated error of 1.0.
        // Higher = faster snap; lower = more subtle assist.
        public float Sensitivity { get; set; } = 12f;

        // Deadzone: errors smaller than this fraction are ignored entirely
        // so micro-jitter doesn't cause constant tiny mouse nudges.
        public float DeadbandRadius { get; set; } = 0.02f;

        // Maximum pixels moved per call — hard cap so a bad detection
        // can't teleport the crosshair across the screen.
        public float MaxPixelsPerFrame { get; set; } = 300f;

        // Speed curve exponent: 1.0 = linear (instant, no slowdown near target).
        // Values < 1 add a precision zone near centre at the cost of snap speed.
        public float SpeedCurveExponent { get; set; } = 1.0f;

        // "SendInput" or "mouse_event"
        public string Method { get; set; } = "SendInput";

        private float _accumX;
        private float _accumY;

        /// <summary>
        /// Move the mouse by (errorX, errorY) in normalised aim space.
        /// Returns the actual pixel delta applied.
        /// </summary>
        public (int dx, int dy) Move(float errorX, float errorY)
        {
            float mag = MathF.Sqrt(errorX * errorX + errorY * errorY);
            if (mag < DeadbandRadius)
                return (0, 0);

            // Adaptive speed: sub-linear curve so we snap fast when far but stay
            // precise when the crosshair is almost on target. At mag=0.1 the
            // multiplier is ~0.5×; at mag=1.0 it is 1.0×; everything in between
            // follows a smooth power curve.
            float adaptiveScale = MathF.Pow(mag, SpeedCurveExponent) / MathF.Max(mag, 1e-6f);

            // Scale normalised error to pixels, accumulate sub-pixel remainder.
            _accumX += errorX * Sensitivity * adaptiveScale;
            _accumY += errorY * Sensitivity * adaptiveScale;

            int dx = (int)_accumX;
            int dy = (int)_accumY;
            _accumX -= dx;
            _accumY -= dy;

            // Hard cap.
            dx = Math.Clamp(dx, -(int)MaxPixelsPerFrame, (int)MaxPixelsPerFrame);
            dy = Math.Clamp(dy, -(int)MaxPixelsPerFrame, (int)MaxPixelsPerFrame);

            if (dx == 0 && dy == 0)
                return (0, 0);

            Apply(dx, dy);
            return (dx, dy);
        }

        /// <summary>Apply a raw pixel delta directly (used for recoil compensation).</summary>
        public void ApplyRaw(int dx, int dy)
        {
            if (dx == 0 && dy == 0) return;
            Apply(dx, dy);
        }

        private void Apply(int dx, int dy)
        {
            if (Method == "SendInput")
                SendInputMouse.SendMouseCommand(MOUSEEVENTF_MOVE, dx, dy);
            else
                mouse_event(MOUSEEVENTF_MOVE, (uint)dx, (uint)dy, 0, 0);
        }

        public void Reset()
        {
            _accumX = 0;
            _accumY = 0;
        }
    }
}
