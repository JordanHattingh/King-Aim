namespace KingAim.Core.Preprocessing;

/// <summary>
/// Records every transform applied to a frame before inference.
/// Required to map model-space coordinates back to source-frame space.
/// </summary>
public sealed record PreprocessingMetadata
{
    public int    SourceWidth   { get; init; }
    public int    SourceHeight  { get; init; }
    public int    ModelWidth    { get; init; }
    public int    ModelHeight   { get; init; }

    /// <summary>Uniform scale applied: min(ModelWidth/SourceWidth, ModelHeight/SourceHeight).</summary>
    public float  ScaleR        { get; init; }

    /// <summary>Pixels of padding added to the left side.</summary>
    public int    PadLeft       { get; init; }

    /// <summary>Pixels of padding added to the top.</summary>
    public int    PadTop        { get; init; }

    public string ColourConversion { get; init; } = "BGR→RGB";
    public string TensorLayout    { get; init; } = "NCHW";

    /// <summary>
    /// Maps a point from model-letterbox space back to source-frame space.
    /// </summary>
    public (float X, float Y) ModelToSource(float modelX, float modelY) =>
        ((modelX - PadLeft) / ScaleR,
         (modelY - PadTop)  / ScaleR);

    /// <summary>
    /// Maps a point from source-frame space to model-letterbox space.
    /// </summary>
    public (float X, float Y) SourceToModel(float srcX, float srcY) =>
        (srcX * ScaleR + PadLeft,
         srcY * ScaleR + PadTop);
}
