using System.Drawing;

namespace Aimmy2.AILogic
{
    public class Prediction
    {
        /// <summary>
        /// Detection rectangle in capture-local pixels. Legacy Aimmy mouse/overlay code relies on this space.
        /// </summary>
        public RectangleF Rectangle { get; set; }

        /// <summary>
        /// Detection rectangle in absolute desktop pixels. Tracking and neural context code must use this space.
        /// </summary>
        public RectangleF ScreenRectangle { get; set; }

        public float Confidence { get; set; }
        public int ClassId { get; set; } = 0;
        public string ClassName { get; set; } = "Unknown";
        public float CenterXTranslated { get; set; }
        public float CenterYTranslated { get; set; }
        public float ScreenCenterX { get; set; }
        public float ScreenCenterY { get; set; }

        /// <summary>
        /// Pose keypoints in absolute desktop-pixel coordinates. Null for detection-only models.
        /// </summary>
        public PlayerKeypoints? Keypoints { get; set; }

        /// <summary>
        /// Returns the best semantic observation point in absolute desktop pixels.
        /// </summary>
        public (float X, float Y) GetObservationPoint(float fallbackFraction = 0.25f)
        {
            var kp = Keypoints?.BestObservationPoint;
            if (kp.HasValue)
                return (kp.Value.X, kp.Value.Y);

            RectangleF box = ScreenRectangle.Width > 0 ? ScreenRectangle : Rectangle;
            return (box.Left + box.Width * 0.5f, box.Top + box.Height * fallbackFraction);
        }
    }
}
