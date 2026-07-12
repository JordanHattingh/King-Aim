using System.Text.Json.Serialization;

namespace KingAim.Core.Models;

/// <summary>
/// Describes a model package's contract. Stored as manifest.json alongside the ONNX file.
/// All downstream components depend only on this contract, not on the filename.
/// </summary>
public sealed class ModelManifest
{
    [JsonRequired] public string ModelId { get; init; } = "";
    [JsonRequired] public string Architecture { get; init; } = "";
    [JsonRequired] public ModelTask Task { get; init; }
    [JsonRequired] public ModelStatus Status { get; init; }

    public bool HumanVerified { get; init; }

    [JsonRequired] public int InputWidth  { get; init; }
    [JsonRequired] public int InputHeight { get; init; }
    [JsonRequired] public string InputDtype { get; init; } = "float32";

    /// <summary>
    /// Ordered keypoint names. Null for detection-only models.
    /// For pose models: ["head", "neck", "upper_chest", "hip"].
    /// </summary>
    public IReadOnlyList<string>? Keypoints { get; init; }

    public int OnnxOpset { get; init; } = 18;

    /// <summary>SHA-256 of the ONNX file. Must match before loading.</summary>
    [JsonRequired] public string ChecksumSha256 { get; init; } = "";

    /// <summary>
    /// Identifies which decoder to use. e.g. "yolo26-pose-v1", "yolo11-pose-v1", "mock-v1".
    /// </summary>
    [JsonRequired] public string Decoder { get; init; } = "";

    /// <summary>Dataset version used for training, for provenance tracking.</summary>
    public string DatasetVersion { get; init; } = "";

    /// <summary>Export metadata: tool, date, source commit.</summary>
    public ExportMetadata? Export { get; init; }

    /// <summary>Optional: path to the matrix_summary.json from the validation run.</summary>
    public string? MatrixSummaryPath { get; init; }
}

public sealed class ExportMetadata
{
    public string Tool { get; init; } = "";
    public string ExportDate { get; init; } = "";
    public string SourceCommit { get; init; } = "";
}
