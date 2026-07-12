using KingAim.Core.Accessibility.Cues;
using KingAim.Core.Accessibility.Pointing;
using KingAim.Core.Capture;
using KingAim.Core.Decoding;
using KingAim.Core.Diagnostics;
using KingAim.Core.Focus;
using KingAim.Core.Inference;
using KingAim.Core.Logging;
using KingAim.Core.Models.Registry;
using KingAim.Core.Preprocessing;
using KingAim.Core.Safety;
using KingAim.Core.Scene;
using KingAim.Core.Tracking;
using KingAim.Core.Validation;

namespace KingAim.Core.AppHost;

/// <summary>
/// Declares all services that form a complete pipeline.
/// Passed to the ApplicationHost at startup.
/// </summary>
public sealed class PipelineDescriptor
{
    public required SafetyPolicy         Safety       { get; init; }
    public required IModelRegistry       Models       { get; init; }
    public required IFrameSource         FrameSource  { get; init; }
    public required IFramePreprocessor   Preprocessor { get; init; }
    public required IInferenceEngine     Engine       { get; init; }
    public required IModelDecoder        Decoder      { get; init; }
    public required NmsParameters        Nms          { get; init; }
    public required IGeometryValidator   Validator    { get; init; }
    public required ITrackerService      Tracker      { get; init; }
    public required ISceneAnalyzer       Scene        { get; init; }
    public required IFocusSelector       Focus        { get; init; }
    public required IVisualCueProvider   Visual       { get; init; }
    public required IAudioCueProvider    Audio        { get; init; }
    public required IHapticCueProvider   Haptic       { get; init; }
    public required IPointingAssistController Pointing { get; init; }
    public required IDiagnosticsService  Diagnostics  { get; init; }
    public required IStructuredLogger    Logger       { get; init; }
}
