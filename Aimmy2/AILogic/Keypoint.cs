namespace Aimmy2.AILogic
{
    /// <summary>
    /// Single pose keypoint decoded from a YOLO-pose ONNX output.
    /// Coordinates are in the same space as the parent Prediction.Rectangle (real screen pixels).
    /// </summary>
    public readonly struct Keypoint
    {
        public float X { get; init; }
        public float Y { get; init; }

        /// <summary>
        /// Raw visibility score from the model (0-1 after sigmoid).
        /// 0 = unknown/absent, ~0.5 = occluded but inferable, ~1.0 = clearly visible.
        /// </summary>
        public float Visibility { get; init; }

        public bool IsVisible => Visibility >= 0.5f;

        public static readonly Keypoint Empty = new() { X = 0, Y = 0, Visibility = 0 };

        public override string ToString() =>
            $"({X:F1}, {Y:F1}) v={Visibility:F2}";
    }

    /// <summary>
    /// Four-keypoint skeleton produced by the YOLO11-pose model.
    /// Index order matches kpt_shape in training YAML: head=0, neck=1, chest=2, hip=3.
    /// </summary>
    public sealed class PlayerKeypoints
    {
        public static readonly int KeypointCount = 4;

        public Keypoint Head    { get; init; }
        public Keypoint Neck    { get; init; }
        public Keypoint Chest   { get; init; }
        public Keypoint Hip     { get; init; }

        /// <summary>
        /// Best aim point: neck if visible, chest if visible, fallback to null (use box heuristic).
        /// </summary>
        public Keypoint? BestAimPoint =>
            Neck.IsVisible  ? Neck  :
            Chest.IsVisible ? Chest :
            null;

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
