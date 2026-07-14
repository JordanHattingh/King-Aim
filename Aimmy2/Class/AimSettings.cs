using System.Globalization;

namespace Aimmy2.Class
{
    internal static class AimSettings
    {
        public static int ImageSize
        {
            get => int.Parse(GetDropdown("Image Size"), CultureInfo.InvariantCulture);
            set => SetDropdown("Image Size", value.ToString(CultureInfo.InvariantCulture));
        }

        public static string ScreenCaptureMethod
        {
            get => GetDropdown("Screen Capture Method");
            set => SetDropdown("Screen Capture Method", value);
        }

        public static string DetectionAreaType => GetDropdown("Detection Area Type");
        public static string TargetClass => GetDropdown("Target Class");
        public static string PredictionMethod => GetDropdown("Prediction Method");
        public static string AimingBoundariesAlignment => GetDropdown("Aiming Boundaries Alignment");
        public static string TracerPosition => GetDropdown("Tracer Position");
        public static string MovementPath => GetDropdown("Movement Path");
        public static string MouseMovementMethod => GetDropdown("Mouse Movement Method");
        public static string GamepadTargetMode => GetDropdown("Gamepad Target Mode");

        public static double FovSize => GetSlider("FOV Size");
        public static float MinimumConfidence => (float)(GetSlider("AI Minimum Confidence") / 100.0);
        public static double StickyAimThreshold => GetSlider("Sticky Aim Threshold");
        public static double OverlayOpacity => GetSlider("Opacity");
        public static double YOffset => GetSlider("Y Offset (Up/Down)");
        public static double XOffset => GetSlider("X Offset (Left/Right)");
        public static double YOffsetPercent => GetSlider("Y Offset (%)");
        public static double XOffsetPercent => GetSlider("X Offset (%)");
        public static int AutoTriggerDelayMilliseconds => (int)(GetSlider("Auto Trigger Delay") * 1000);
        public static int MouseJitter => (int)GetSlider("Mouse Jitter");
        public static double MouseSensitivity => GetSlider("Mouse Sensitivity (+/-)");
        public static double KalmanLeadTime => GetSlider("Kalman Lead Time");
        public static double WiseTheFoxLeadTime => GetSlider("WiseTheFox Lead Time");
        public static double ShalloeLeadMultiplier => GetSlider("Shalloe Lead Multiplier");
        public static int AiFpsLimit => Math.Max(0, (int)Math.Round(GetSlider("AI FPS Limit")));
        public static double GamepadAssistStrength => GetSlider("Gamepad Assist Strength");
        public static double GamepadAssistSmoothness => GetSlider("Gamepad Assist Smoothness");
        public static double GamepadRecoilH    => GetSlider("Gamepad Recoil H");
        public static double GamepadRecoilV    => GetSlider("Gamepad Recoil V");
        public static double GamepadTargetPull => GetSlider("Gamepad Target Pull");
        public static double ViewmodelExclusion => GetSlider("Viewmodel Exclusion Zone") / 100.0;
        public static double CursorExclusionRadius => GetSlider("Cursor Exclusion Radius");
        public static double DynamicFovSize => GetSlider("Dynamic FOV Size");

        public static bool AutoTrigger => GetToggle("Auto Trigger");
        public static bool ConstantAiTracking => GetToggle("Constant AI Tracking");
        public static bool SprayMode => GetToggle("Spray Mode");
        public static bool CursorCheck => GetToggle("Cursor Check");
        public static bool AimAssist => GetToggle("Aim Assist");
        public static bool ShowDetectedPlayer => GetToggle("Show Detected Player");
        public static bool ShowFov => GetToggle("FOV");
        public static bool ShowAiConfidence => GetToggle("Show AI Confidence");
        public static bool ShowTracers => GetToggle("Show Tracers");
        public static bool UseXAxisPercentageAdjustment => GetToggle("X Axis Percentage Adjustment");
        public static bool UseYAxisPercentageAdjustment => GetToggle("Y Axis Percentage Adjustment");
        public static bool Predictions => GetToggle("Predictions");
        public static bool StickyAim => GetToggle("Sticky Aim");
        public static bool CollectDataWhilePlaying => GetToggle("Collect Data While Playing");
        public static bool AutoLabelData => GetToggle("Auto Label Data");
        public static bool ThirdPersonSupport => GetToggle("Third Person Support");
        public static bool GamepadAssist => GetToggle("Gamepad Assist");
        public static bool DynamicFovEnabled   => GetToggle("Dynamic FOV");
        public static bool DetectionLogging    => GetToggle("Detection Logging");

        public static double GetSlider(string key) =>
            Convert.ToDouble(Dictionary.sliderSettings[key], CultureInfo.InvariantCulture);

        public static bool GetToggle(string key) =>
            Convert.ToBoolean(Dictionary.toggleState[key], CultureInfo.InvariantCulture);

        public static string GetDropdown(string key) =>
            Convert.ToString(Dictionary.dropdownState[key], CultureInfo.InvariantCulture) ?? string.Empty;

        private static void SetDropdown(string key, string value)
        {
            Dictionary.dropdownState[key] = value;
        }
    }
}
