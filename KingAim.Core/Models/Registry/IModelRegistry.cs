namespace KingAim.Core.Models.Registry;

/// <summary>
/// Discovers, validates, and provides model packages.
/// Components must use this rather than hard-coding model filenames.
/// </summary>
public interface IModelRegistry
{
    /// <summary>All packages discovered in the models directory.</summary>
    IReadOnlyList<ModelPackage> Available { get; }

    /// <summary>Returns the package with this ID, or null.</summary>
    ModelPackage? Find(string modelId);

    /// <summary>Returns the first approved package, ordered by preference.</summary>
    ModelPackage? Default { get; }

    /// <summary>
    /// Loads and verifies a package by ID. Throws if the checksum fails
    /// or the manifest is invalid.
    /// </summary>
    Task<ModelPackage> LoadAsync(string modelId, CancellationToken ct = default);

    /// <summary>Rescans the models directory.</summary>
    Task RefreshAsync(CancellationToken ct = default);
}
