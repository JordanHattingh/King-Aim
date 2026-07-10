using Aimmy2.AILogic;
using Aimmy2.Class;
using Aimmy2.MouseMovementLibraries.GHubSupport;
using Class;
using MouseMovementLibraries.ddxoftSupport;
using MouseMovementLibraries.RazerSupport;
using MouseMovementLibraries.SendInputSupport;
using System.Drawing;
using System.Runtime.InteropServices;

namespace InputLogic
{
    internal class MouseManager
    {
        private static readonly double ScreenWidth = WinAPICaller.ScreenWidth;
        private static readonly double ScreenHeight = WinAPICaller.ScreenHeight;

        private static DateTime LastClickTime = DateTime.MinValue;
        private static bool isSpraying = false;

        private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        private const uint MOUSEEVENTF_LEFTUP = 0x0004;
        private const uint MOUSEEVENTF_MOVE = 0x0001;
        private static double previousX = 0;
        private static double previousY = 0;
        public static double smoothingFactor = 0.5;
        public static bool IsEMASmoothingEnabled = false;

        [DllImport("user32.dll")]
        private static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, int dwExtraInfo);

        private static Random MouseRandom = new();

        private static MovementMlp? _neuralMovement;
        private static float _lastTargetSizePixels;
        private static float _lastEmittedSpeedPixPerMs;

        /// <summary>
        /// Optional controlled-pointing movement model. King Aim's default live accessibility runtime
        /// does not wire the perception pipeline into this property. Atomic access prevents a stale
        /// reference from being observed during model lifecycle changes.
        /// </summary>
        public static MovementMlp? NeuralMovement
        {
            get => Volatile.Read(ref _neuralMovement);
            set
            {
                MovementMlp? previous = Interlocked.Exchange(ref _neuralMovement, value);
                if (!ReferenceEquals(previous, value))
                {
                    Volatile.Write(ref _lastEmittedSpeedPixPerMs, 0f);
                    value?.Reset();
                }
            }
        }

        /// <summary>Width/height of the current generic pointing target in pixels.</summary>
        public static float LastTargetSizePixels
        {
            get => Volatile.Read(ref _lastTargetSizePixels);
            set => Volatile.Write(ref _lastTargetSizePixels, value);
        }

        public static bool ClearNeuralMovementIf(MovementMlp expected)
        {
            MovementMlp? previous = Interlocked.CompareExchange(ref _neuralMovement, null, expected);
            if (!ReferenceEquals(previous, expected))
                return false;

            Volatile.Write(ref _lastEmittedSpeedPixPerMs, 0f);
            return true;
        }

        private static DateTime _lastMoveCrosshairTime = DateTime.UtcNow;

        private static double EmaSmoothing(double previousValue, double currentValue, double smoothingFactor) => (currentValue * smoothingFactor) + (previousValue * (1 - smoothingFactor));

        // Cleanup
        private static (Action down, Action up) GetMouseActions()
        {
            string mouseMovementMethod = AimSettings.MouseMovementMethod;
            Action mouseDownAction;
            Action mouseUpAction;

            switch (mouseMovementMethod)
            {
                case "SendInput":
                    mouseDownAction = () => SendInputMouse.SendMouseCommand(MOUSEEVENTF_LEFTDOWN);
                    mouseUpAction = () => SendInputMouse.SendMouseCommand(MOUSEEVENTF_LEFTUP);
                    break;
                case "LG HUB":
                    mouseDownAction = () => LGMouse.Move(1, 0, 0, 0);
                    mouseUpAction = () => LGMouse.Move(0, 0, 0, 0);
                    break;
                case "Razer Synapse (Require Razer Peripheral)":
                    mouseDownAction = () => RZMouse.mouse_click(1);
                    mouseUpAction = () => RZMouse.mouse_click(0);
                    break;
                case "ddxoft Virtual Input Driver":
                    mouseDownAction = () => DdxoftMain.ddxoftInstance.btn!(1);
                    mouseUpAction = () => DdxoftMain.ddxoftInstance.btn(2);
                    break;
                default:
                    mouseDownAction = () => mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                    mouseUpAction = () => mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                    break;
            }

            return (mouseDownAction, mouseUpAction);
        }

        public static async Task DoTriggerClick(RectangleF? detectionBox = null)
        {
            // there was a toggle for this, but i realized if it was off, it would never stop spraying. - T
            if (!(InputBindingManager.IsHoldingBinding("Aim Keybind") || InputBindingManager.IsHoldingBinding("Second Aim Keybind")))
            {
                ResetSprayState();
                return;
            }


            if (AimSettings.SprayMode)
            {
                if (AimSettings.CursorCheck)
                {
                    Point mousePos = WinAPICaller.GetCursorPosition();

                    if (detectionBox.HasValue && !detectionBox.Value.Contains(mousePos.X, mousePos.Y))
                    {
                        if (isSpraying) ReleaseMouseButton();
                        return;
                    }
                }

                if (!isSpraying) HoldMouseButton();
                return;
            }

            // Single click logic if spray mode off
            int timeSinceLastClick = (int)(DateTime.UtcNow - LastClickTime).TotalMilliseconds;
            int triggerDelayMilliseconds = AimSettings.AutoTriggerDelayMilliseconds;
            const int clickDelayMilliseconds = 20;

            if (timeSinceLastClick < triggerDelayMilliseconds && LastClickTime != DateTime.MinValue)
            {
                return;
            }

            var (mouseDown, mouseUp) = GetMouseActions();

            mouseDown.Invoke();
            await Task.Delay(clickDelayMilliseconds);
            mouseUp.Invoke();

            LastClickTime = DateTime.UtcNow;
        }

        #region Spray Mode Methods
        public static void HoldMouseButton()
        {
            if (isSpraying) return;

            var (mouseDown, _) = GetMouseActions();
            mouseDown.Invoke();
            isSpraying = true;
        }

        public static void ReleaseMouseButton()
        {
            if (!isSpraying) return;

            var (_, mouseUp) = GetMouseActions();
            mouseUp.Invoke();
            isSpraying = false;
        }

        public static void ResetSprayState()
        {
            if (isSpraying)
            {
                ReleaseMouseButton();
            }
        }
        #endregion

        public static void MoveCrosshair(int detectedX, int detectedY)
        {
            int halfScreenWidth = (int)ScreenWidth / 2;
            int halfScreenHeight = (int)ScreenHeight / 2;

            int targetX = detectedX - halfScreenWidth;
            int targetY = detectedY - halfScreenHeight;

            double aspectRatioCorrection = ScreenWidth / ScreenHeight;

            int MouseJitter = AimSettings.MouseJitter;
            int jitterX = MouseRandom.Next(-MouseJitter, MouseJitter);
            int jitterY = MouseRandom.Next(-MouseJitter, MouseJitter);

            Point start = new(0, 0);
            Point end = new(targetX, targetY);
            Point newPosition = new Point(0, 0);

            var now = DateTime.UtcNow;
            float dtSec = Math.Clamp((float)(now - _lastMoveCrosshairTime).TotalSeconds, 0.001f, 0.1f);
            _lastMoveCrosshairTime = now;

            // Optional controlled-pointing MLP. The previous emitted speed uses the same
            // pixels/ms unit recorded by training/record_movement.py.
            MovementMlp? neuralMovement = NeuralMovement;
            float targetSizePixels = LastTargetSizePixels;
            float previousSpeedPixPerMs = Volatile.Read(ref _lastEmittedSpeedPixPerMs);
            var mlpDelta = neuralMovement?.Move(
                targetX,
                targetY,
                targetSizePixels,
                targetSizePixels,
                dtSec,
                previousSpeedPixPerMs);

            if (mlpDelta.HasValue)
            {
                newPosition = new Point(mlpDelta.Value.dx, mlpDelta.Value.dy);
            }
            else
            {
                double t = 1.0 - AimSettings.MouseSensitivity;
                switch (AimSettings.MovementPath)
                {
                    case "Human Bezier":
                        newPosition = MovementPaths.HumanBezier(start, end, t);
                        break;
                    case "Cubic Bezier":
                        {
                            double dx = end.X - start.X;
                            double dy = end.Y - start.Y;
                            double dist = Math.Sqrt(dx * dx + dy * dy);
                            double px = dist > 1 ? -dy / dist : 0;
                            double py = dist > 1 ?  dx / dist : 0;
                            double off = dist * 0.12;
                            Point control1 = new Point(
                                (int)(start.X + dx / 3 + px * off),
                                (int)(start.Y + dy / 3 + py * off));
                            Point control2 = new Point(
                                (int)(start.X + 2 * dx / 3 - px * off * 0.5),
                                (int)(start.Y + 2 * dy / 3 - py * off * 0.5));
                            newPosition = MovementPaths.CubicBezier(start, end, control1, control2, t);
                        }
                        break;
                    case "Linear":
                        newPosition = MovementPaths.Lerp(start, end, t);
                        break;
                    case "Exponential":
                        newPosition = MovementPaths.Exponential(start, end, 1 - (AimSettings.MouseSensitivity - 0.2), 3.0);
                        break;
                    case "Adaptive":
                        newPosition = MovementPaths.Adaptive(start, end, t);
                        break;
                    case "Perlin Noise":
                        newPosition = MovementPaths.PerlinNoise(start, end, t, 20, 0.5);
                        break;
                    default:
                        newPosition = MovementPaths.Lerp(start, end, t);
                        break;
                }
            }

            if (IsEMASmoothingEnabled)
            {
                newPosition.X = (int)EmaSmoothing(previousX, newPosition.X, smoothingFactor);
                newPosition.Y = (int)EmaSmoothing(previousY, newPosition.Y, smoothingFactor);
            }

            newPosition.X = Math.Clamp(newPosition.X, -400, 400);
            newPosition.Y = Math.Clamp(newPosition.Y, -400, 400);

            newPosition.Y = (int)(newPosition.Y / aspectRatioCorrection);

            newPosition.X += jitterX;
            newPosition.Y += jitterY;

            switch (AimSettings.MouseMovementMethod)
            {
                case "SendInput":
                    SendInputMouse.SendMouseCommand(MOUSEEVENTF_MOVE, newPosition.X, newPosition.Y);
                    break;

                case "LG HUB":
                    LGMouse.Move(0, newPosition.X, newPosition.Y, 0);
                    break;

                case "Razer Synapse (Require Razer Peripheral)":
                    RZMouse.mouse_move(newPosition.X, newPosition.Y, true);
                    break;

                case "ddxoft Virtual Input Driver":
                    DdxoftMain.ddxoftInstance.movR!(newPosition.X, newPosition.Y);
                    break;

                default:
                    mouse_event(MOUSEEVENTF_MOVE, (uint)newPosition.X, (uint)newPosition.Y, 0, 0);
                    break;
            }

            previousX = newPosition.X;
            previousY = newPosition.Y;
            float emittedSpeedPixPerMs = MathF.Sqrt(
                newPosition.X * newPosition.X + newPosition.Y * newPosition.Y)
                / Math.Max(dtSec * 1000f, 1e-3f);
            Volatile.Write(ref _lastEmittedSpeedPixPerMs, emittedSpeedPixPerMs);

            if (!AimSettings.AutoTrigger)
            {
                ResetSprayState();
            }
        }
    }
}
