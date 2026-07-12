namespace KingAim.Core.Models;

public static class ModelManifestValidator
{
    /// <summary>
    /// Returns all validation errors found in the manifest.
    /// Empty list means the manifest is structurally valid.
    /// Does NOT verify the ONNX checksum — that requires the file path.
    /// </summary>
    public static IReadOnlyList<string> Validate(ModelManifest manifest)
    {
        var errors = new List<string>();

        if (string.IsNullOrWhiteSpace(manifest.ModelId))
            errors.Add("ModelId is required.");
        if (string.IsNullOrWhiteSpace(manifest.Architecture))
            errors.Add("Architecture is required.");
        if (string.IsNullOrWhiteSpace(manifest.Decoder))
            errors.Add("Decoder is required.");
        if (manifest.InputWidth  <= 0)
            errors.Add($"InputWidth must be positive (got {manifest.InputWidth}).");
        if (manifest.InputHeight <= 0)
            errors.Add($"InputHeight must be positive (got {manifest.InputHeight}).");
        if (string.IsNullOrWhiteSpace(manifest.ChecksumSha256) || manifest.ChecksumSha256.Length != 64)
            errors.Add("ChecksumSha256 must be a 64-character hex string.");
        if (manifest.Status == ModelStatus.Rejected)
            errors.Add("Model status is Rejected and must not be loaded.");

        if (manifest.Task == ModelTask.Pose)
        {
            if (manifest.Keypoints == null || manifest.Keypoints.Count == 0)
                errors.Add("Pose models must declare at least one keypoint.");
        }

        return errors;
    }

    /// <summary>Throws if the manifest has any validation errors.</summary>
    public static void RequireValid(ModelManifest manifest)
    {
        var errors = Validate(manifest);
        if (errors.Count > 0)
            throw new InvalidOperationException(
                $"ModelManifest '{manifest.ModelId}' is invalid:\n" +
                string.Join("\n", errors.Select(e => "  " + e)));
    }
}
