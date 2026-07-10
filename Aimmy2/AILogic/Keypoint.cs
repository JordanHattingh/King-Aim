namespace Aimmy2.AILogic
{
    /// <summary>
    /// Single pose keypoint decoded from a YOLO-pose ONNX output.
    /// Coordinates are absolute desktop pixels.
    /// </summary>
    public readonly struct Keypoint
    {
        public float X { get; init; }
        public float Y { get; init; }

        /// <summary>Visibility/confidence score in [0,1] after the manifest-declared decode rule.</summary>
        public float Visibility { get; init; }

        public bool IsVisible => Visibility >= 0.5f;
        public bool IsUsable => Visibility >= 0.25f;

        public static readonly Keypoint Empty = new() { X = 0, Y = 0, Visibility = 0 };

        public override string ToString() => $"({X:F1}, {Y:F1}) v={Visibility:F2}";
    }

    /// <summary>
    /// Four-keypoint midline skeleton. Index order must match the model manifest/training schema.
    /// </summary>
    public sealed class PlayerKeypoints
    {
        public const int KeypointCount = 4;

        public Keypoint Head { get; init; }
        public Keypoint Neck { get; init; }
        public Keypoint Chest { get; init; }
        public Keypoint Hip { get; init; }

        /// <summary>
        /// Semantic observation point priority for accessibility overlays/cues: neck, chest, head, hip.
        /// </summary>
        public Keypoint? BestObservationPoint =>
            Neck.IsUsable ? Neck :
            Chest.IsUsable ? Chest :
            Head.IsUsable ? Head :
            Hip.IsUsable ? Hip :
            null;

        // Backward-compatible name for legacy code. New code should use BestObservationPoint.
        public Keypoint? BestAimPoint => BestObservationPoint;

        public Keypoint this[int i] => i switch
        {
            0 => Head,
            1 => Neck,
            2 => Chest,
            3 => Hip,
            _ => Keypoint.Empty
        };
    }
}
