namespace KingAim.Core.Models;

public enum ModelStatus
{
    /// <summary>Under active development. Not for production use.</summary>
    Experimental,

    /// <summary>Passed all benchmark gates. Approved for production.</summary>
    Approved,

    /// <summary>Superseded by a newer model. Still loads but not the default.</summary>
    Deprecated,

    /// <summary>Failed a gate. Must not be loaded.</summary>
    Rejected,
}
