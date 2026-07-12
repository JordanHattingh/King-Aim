namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Describes a single input or output tensor as read from the ONNX session metadata.
/// All values are taken directly from the model — nothing is inferred or assumed.
/// </summary>
public sealed record TensorContract(
    string Name,
    string ElementType,
    IReadOnlyList<TensorDimension> Dimensions)
{
    /// <summary>True when any dimension is symbolic/dynamic.</summary>
    public bool HasDynamicDimensions => Dimensions.Any(d => d.IsDynamic);

    /// <summary>
    /// Returns fixed integer extents for all dims, or -1 for dynamic ones.
    /// Useful when the caller needs a flat int array for a known-static shape.
    /// </summary>
    public int[] FixedShape => Dimensions.Select(d => d.FixedValue ?? -1).ToArray();
}
