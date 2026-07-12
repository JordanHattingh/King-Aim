namespace KingAim.Core.Safety;

/// <summary>
/// Defines what King Aim is and is not permitted to do.
/// These restrictions apply universally — online, offline, multiplayer, single-player —
/// because King Aim is an accessibility tool, not a game-mode-specific cheat.
/// </summary>
public enum AssistanceCapability
{
    // ── Permitted ────────────────────────────────────────────────────────────
    /// <summary>Passive visual cues: overlays, bounding boxes, direction indicators.</summary>
    VisualCues,

    /// <summary>Passive audio cues: spatial panning, pitch, distance feedback.</summary>
    AudioCues,

    /// <summary>Passive haptic cues: controller vibration mapped to target position.</summary>
    HapticCues,

    /// <summary>
    /// Controlled pointing assistance: user must hold an enable key;
    /// subject to strict movement limits; no automatic firing.
    /// </summary>
    ControlledPointingAssistance,

    /// <summary>Replay and diagnostic recording (with explicit user consent).</summary>
    DiagnosticRecording,

    // ── Prohibited ───────────────────────────────────────────────────────────
    /// <summary>Automatic firing input without continuous user hold.</summary>
    AutomaticFiring,

    /// <summary>Recoil control or compensation input.</summary>
    RecoilControl,

    /// <summary>Game process injection or memory reading.</summary>
    ProcessInjectionOrMemoryAccess,

    /// <summary>Anti-cheat system interaction, bypass, or evasion.</summary>
    AntiCheatInteraction,

    /// <summary>Stealth operation (hiding from the OS or other processes).</summary>
    StealthOperation,

    /// <summary>Unattended operation without continuous user presence.</summary>
    UnattendedOperation,

    /// <summary>Network traffic manipulation or packet injection.</summary>
    NetworkManipulation,
}
