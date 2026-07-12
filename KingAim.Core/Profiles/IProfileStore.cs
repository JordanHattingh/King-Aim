namespace KingAim.Core.Profiles;

public interface IProfileStore
{
    IReadOnlyList<AccessibilityProfile> All { get; }
    AccessibilityProfile? Active { get; }

    Task<AccessibilityProfile> LoadAsync(string id, CancellationToken ct = default);
    Task SaveAsync(AccessibilityProfile profile, CancellationToken ct = default);
    Task DeleteAsync(string id, CancellationToken ct = default);
    Task SetActiveAsync(string id, CancellationToken ct = default);

    /// <summary>
    /// Exports a profile to a file. Private recordings are never included.
    /// </summary>
    Task ExportAsync(string id, string filePath, CancellationToken ct = default);
    Task<AccessibilityProfile> ImportAsync(string filePath, CancellationToken ct = default);
}
