namespace Aimmy2.AILogic
{
    /// <summary>
    /// Stable train/runtime feature-contract identifiers. The Python trainers write the same values
    /// into their schema artifacts and update_manifest.py copies those contracts into model bundles.
    /// </summary>
    public static class NeuralFeatureSchemas
    {
        public const string TemporalV2 = "track-motion-8x8-v2";
        public const string CalibrationV2 = "detection-context-v2";
        public const string MovementV1 = "pointing-velocity-v1";
    }
}
