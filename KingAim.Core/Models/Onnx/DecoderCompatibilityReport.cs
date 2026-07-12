namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Result of checking a decoder against an <see cref="OnnxModelContract"/>.
/// Errors block inference. Warnings allow inference with caveats.
/// Observations are informational only.
/// </summary>
public sealed record DecoderCompatibilityReport(
    string DecoderId,
    bool   IsCompatible,
    IReadOnlyList<string> Errors,
    IReadOnlyList<string> Warnings,
    IReadOnlyList<string> Observations)
{
    public static DecoderCompatibilityReport Compatible(
        string decoderId,
        IReadOnlyList<string>? warnings      = null,
        IReadOnlyList<string>? observations  = null) =>
        new(decoderId, true,
            Errors:       [],
            Warnings:     warnings     ?? [],
            Observations: observations ?? []);

    public static DecoderCompatibilityReport Incompatible(
        string decoderId,
        IReadOnlyList<string> errors,
        IReadOnlyList<string>? warnings      = null,
        IReadOnlyList<string>? observations  = null) =>
        new(decoderId, false,
            Errors:       errors,
            Warnings:     warnings     ?? [],
            Observations: observations ?? []);
}
