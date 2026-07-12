using KingAim.Core.Perception;

namespace KingAim.Core.Validation;

/// <summary>
/// Default geometry validation. All checks are documented and independently configurable.
/// </summary>
public sealed class GeometryValidator : IGeometryValidator
{
    public float MinConfidence           { get; init; } = 0.25f;
    public float MinBoxAreaFraction      { get; init; } = 0.0001f;  // 0.01% of frame
    public float MaxBoxAreaFraction      { get; init; } = 0.80f;
    public float MaxOffscreenFraction    { get; init; } = 0.20f;
    public float KeypointBodyMarginPx    { get; init; } = 20f;
    public float MaxPlausibleBodyAspect  { get; init; } = 4.0f;

    public ValidationResult Validate(PoseDetection det, int srcW, int srcH)
    {
        var issues = new List<string>();
        var box    = det.BoundingBox;
        float frameArea = srcW * srcH;

        // Confidence
        if (det.ObjectConfidence < MinConfidence)
        {
            issues.Add($"Confidence {det.ObjectConfidence:F3} < {MinConfidence}");
            return Rejected(issues);
        }

        // Box validity
        if (!box.IsValid)
        {
            issues.Add("Zero or negative area bounding box.");
            return Rejected(issues);
        }

        float boxArea = box.Area;
        if (boxArea / frameArea < MinBoxAreaFraction)
        {
            issues.Add($"Box area fraction {boxArea / frameArea:F4} too small.");
            return Rejected(issues);
        }

        // Off-screen fraction
        float overlapX = Math.Max(0, Math.Min(box.Right, srcW) - Math.Max(box.Left, 0));
        float overlapY = Math.Max(0, Math.Min(box.Bottom, srcH) - Math.Max(box.Top, 0));
        float overlapA = overlapX * overlapY;
        float offscreen = 1f - overlapA / boxArea;
        if (offscreen > MaxOffscreenFraction)
        {
            issues.Add($"Box is {offscreen:P0} off-screen.");
            return Rejected(issues);
        }

        // Aspect ratio plausibility
        float aspect = box.Height > 0 ? box.Width / box.Height : float.MaxValue;
        if (aspect > MaxPlausibleBodyAspect)
            issues.Add($"Unusual box aspect ratio {aspect:F2}.");

        // Keypoint topology
        var head  = det.Head;
        var neck  = det.Neck;
        var chest = det.UpperChest;
        var hip   = det.Hip;

        if (head.HasValue && neck.HasValue)
        {
            if (head.Value.Y > neck.Value.Y + 10)
                issues.Add("Head is below neck — likely false detection.");
        }

        if (neck.HasValue && chest.HasValue && neck.Value.Y > chest.Value.Y + 10)
            issues.Add("Neck is below upper chest.");

        if (chest.HasValue && hip.HasValue && chest.Value.Y > hip.Value.Y + 10)
            issues.Add("Upper chest is below hip.");

        // Assign quality
        bool hasPoseIssues = issues.Count > 0;

        // All keypoints absent/uncertain → CueOnly
        int visibleKps = det.Keypoints.Count(kp =>
            kp.Visibility == KeypointVisibility.Visible && kp.Confidence > 0.3f);

        return issues.Count == 0
            ? new ValidationResult { Quality = DetectionQuality.High,    Issues = issues }
            : visibleKps >= 2
                ? new ValidationResult { Quality = DetectionQuality.Usable,  Issues = issues }
                : new ValidationResult { Quality = DetectionQuality.CueOnly, Issues = issues };
    }

    private static ValidationResult Rejected(List<string> issues) =>
        new() { Quality = DetectionQuality.Rejected, Issues = issues };
}
