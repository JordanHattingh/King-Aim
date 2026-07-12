namespace KingAim.Core.Safety;

/// <summary>
/// Describes what input source King Aim is processing.
/// The mode controls which capabilities are available but does NOT restrict
/// by game type — disabled gamers play online and multiplayer too.
/// </summary>
public enum OperatingMode
{
    /// <summary>
    /// Processing a pre-recorded video file.
    /// Full accessibility output allowed. Pointing assistance against test canvas only.
    /// </summary>
    RecordedVideo,

    /// <summary>
    /// Processing a folder of static images.
    /// Used for model evaluation and accessibility cue development.
    /// </summary>
    StaticImages,

    /// <summary>
    /// Live screen capture from any running game — online, offline, multiplayer, single-player.
    /// All permitted capabilities available. Prohibited capabilities remain blocked.
    /// </summary>
    LiveCapture,

    /// <summary>
    /// Synthetic test-pattern source. Used for pipeline integration tests.
    /// No pointing assistance. Cue output goes to test sinks only.
    /// </summary>
    AccessibilityTestHarness,
}
