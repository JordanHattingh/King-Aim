namespace KingAim.Core.Models.Onnx;

/// <summary>
/// One axis in a tensor shape. Either a fixed positive integer or a symbolic
/// (dynamic) name. Never stored as -1 — that sentinel is normalized at the
/// inspection boundary.
/// </summary>
public sealed record TensorDimension(int? FixedValue, string? SymbolicName)
{
    public bool IsDynamic => FixedValue is null;

    public override string ToString() =>
        IsDynamic ? (SymbolicName ?? "?") : FixedValue!.Value.ToString();
}
