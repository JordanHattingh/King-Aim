namespace KingAim.Core.Perception;

/// <summary>Axis-aligned bounding box in source-frame pixel space.</summary>
public readonly record struct DetectionBoundingBox(float Left, float Top, float Right, float Bottom)
{
    public float Width  => Right  - Left;
    public float Height => Bottom - Top;
    public float CentreX => (Left + Right)  / 2f;
    public float CentreY => (Top  + Bottom) / 2f;
    public float Area    => Width * Height;

    public bool IsValid => Width > 0 && Height > 0;

    public static DetectionBoundingBox FromXywh(float cx, float cy, float w, float h) =>
        new(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f);
}
