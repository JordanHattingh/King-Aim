namespace Aimmy2.AILogic
{
    internal static class AimProcessingDecisions
    {
        internal static bool ShouldRunPrediction(
            bool showDetectedPlayer,
            bool constantAiTracking,
            bool aimKeyHeld,
            bool secondAimKeyHeld,
            bool gamepadAssistEnabled) =>
            showDetectedPlayer ||
            constantAiTracking ||
            aimKeyHeld ||
            secondAimKeyHeld ||
            gamepadAssistEnabled;

        internal static bool ShouldProcessFrame(
            bool aimAssist,
            bool showDetectedPlayer,
            bool gamepadAssistEnabled) =>
            aimAssist ||
            showDetectedPlayer ||
            gamepadAssistEnabled;
    }
}
