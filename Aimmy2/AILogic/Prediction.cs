using System.Drawing;

namespace Aimmy2.AILogic
{
    public class Prediction
    {
        public RectangleF Rectangle { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; } = 0;
        public string ClassName { get; set; } = "Enemy";
        public float CenterXTranslated { get; set; }
        public float CenterYTranslated { get; set; }
        public float ScreenCenterX { get; set; }
        public float ScreenCenterY { get; set; }

        /// <summary>
        /// Populated by PredictionFilter when the loaded model is a pose model.
        /// Null for plain detection models — callers must fall back to AimPointFraction heuristic.
        /// </summary>
        public PlayerKeypoints? Keypoints { get; set; }

        /// <summary>
        /// Returns the best aim point: pose keypoint if available and visible, otherwise
        /// the legacy bounding-box fraction position.
        /// </summary>
        public (float X, float Y) GetAimPoint(float aimPointFraction = 0.25f)
        {
            var kp = Keypoints?.BestAimPoint;
            if (kp.HasValue)
                return (kp.Value.X, kp.Value.Y);

            return (ScreenCenterX, Rectangle.Top + Rectangle.Height * aimPointFraction);
        }
    }
}
