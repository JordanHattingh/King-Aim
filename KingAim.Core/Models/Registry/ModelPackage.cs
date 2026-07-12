namespace KingAim.Core.Models.Registry;

/// <summary>
/// A resolved model package: manifest + verified ONNX path.
/// </summary>
public sealed class ModelPackage
{
    public ModelManifest Manifest { get; }
    public string OnnxPath       { get; }
    public bool   ChecksumVerified { get; }

    public ModelPackage(ModelManifest manifest, string onnxPath, bool checksumVerified)
    {
        Manifest         = manifest;
        OnnxPath         = onnxPath;
        ChecksumVerified = checksumVerified;
    }

    public string ModelId => Manifest.ModelId;
    public bool   IsApproved => Manifest.Status == ModelStatus.Approved && ChecksumVerified;
}
