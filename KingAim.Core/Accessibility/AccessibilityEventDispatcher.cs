using KingAim.Core.Accessibility.Events;
using KingAim.Core.Scene;
using KingAim.Core.Tracking;

namespace KingAim.Core.Accessibility;

/// <summary>
/// Converts SceneState transitions into AccessibilityEvents and fans them out
/// to all registered sinks. Stateless per-event; stateful only for delta detection.
/// </summary>
public sealed class AccessibilityEventDispatcher
{
    private readonly List<IAccessibilityEventSink> _sinks = [];
    private SceneState? _prev;

    public void AddSink(IAccessibilityEventSink sink)    => _sinks.Add(sink);
    public void RemoveSink(IAccessibilityEventSink sink) => _sinks.Remove(sink);

    public bool IsEmergencyDisabled { get; private set; }

    public void EmergencyDisable()
    {
        IsEmergencyDisabled = true;
        Emit(new AccessibilityEvent { Kind = AccessibilityEventKind.EmergencyDisabled });
    }

    public void Dispatch(SceneState current, long frameId, int srcW, int srcH)
    {
        if (IsEmergencyDisabled) return;

        bool hadFocus = _prev?.PrimaryTrack != null;
        bool hasFocus = current.PrimaryTrack != null;
        int? prevId   = _prev?.PrimaryTrack?.TrackId;
        int? curId    = current.PrimaryTrack?.TrackId;

        if (!hadFocus && hasFocus)
        {
            Emit(new AccessibilityEvent
            {
                Kind    = AccessibilityEventKind.FocusTargetAppeared,
                FrameId = frameId, TrackId = curId,
            });
        }
        else if (hadFocus && !hasFocus)
        {
            Emit(new AccessibilityEvent
            {
                Kind    = AccessibilityEventKind.FocusTargetLost,
                FrameId = frameId, TrackId = prevId,
            });
        }
        else if (hadFocus && hasFocus && prevId != curId)
        {
            // Focus switched targets: treat as lost + appeared
            Emit(new AccessibilityEvent
            {
                Kind    = AccessibilityEventKind.FocusTargetLost,
                FrameId = frameId, TrackId = prevId,
            });
            Emit(new AccessibilityEvent
            {
                Kind    = AccessibilityEventKind.FocusTargetAppeared,
                FrameId = frameId, TrackId = curId,
            });
        }

        // Movement delta (same target only)
        if (hadFocus && hasFocus && prevId == curId)
        {
            var prev = _prev!.PrimaryTrack!;
            var curr = current.PrimaryTrack!;
            float dx = curr.Box.CentreX - prev.Box.CentreX;
            float dy = curr.Box.CentreY - prev.Box.CentreY;
            float minPx = srcW * 0.01f;

            if (Math.Abs(dx) >= minPx)
                Emit(new AccessibilityEvent
                {
                    Kind    = dx > 0 ? AccessibilityEventKind.FocusTargetMovedRight
                                     : AccessibilityEventKind.FocusTargetMovedLeft,
                    FrameId = frameId, TrackId = curId,
                    Intensity = Math.Min(1f, Math.Abs(dx) / (srcW * 0.1f)),
                });

            if (Math.Abs(dy) >= minPx)
                Emit(new AccessibilityEvent
                {
                    Kind    = dy > 0 ? AccessibilityEventKind.FocusTargetMovedDown
                                     : AccessibilityEventKind.FocusTargetMovedUp,
                    FrameId = frameId, TrackId = curId,
                    Intensity = Math.Min(1f, Math.Abs(dy) / (srcH * 0.1f)),
                });
        }

        // Near-centre alert (within 10% of screen diagonal)
        if (hasFocus)
        {
            var t   = current.PrimaryTrack!;
            float d = MathF.Sqrt(MathF.Pow(t.Box.CentreX - srcW / 2f, 2) +
                                 MathF.Pow(t.Box.CentreY - srcH / 2f, 2));
            float maxD = MathF.Sqrt(srcW * srcW + srcH * srcH) / 2f;
            if (d / maxD < 0.10f)
                Emit(new AccessibilityEvent
                {
                    Kind    = AccessibilityEventKind.FocusTargetNearCentre,
                    FrameId = frameId, TrackId = curId,
                    Intensity = 1f - d / (maxD * 0.10f),
                });
        }

        // Multiple-targets edge rising
        bool hadMultiple = (_prev?.VisibleEnemyCount ?? 0) > 1;
        bool hasMultiple = current.VisibleEnemyCount > 1;
        if (!hadMultiple && hasMultiple)
            Emit(new AccessibilityEvent
            {
                Kind = AccessibilityEventKind.MultipleTargetsVisible, FrameId = frameId,
            });

        // Tracking-interrupted / restored
        if (hasFocus && current.PrimaryTrack!.IsLost &&
            (!hadFocus || (_prev?.PrimaryTrack?.MissingFrames ?? 0) == 0))
        {
            Emit(new AccessibilityEvent
            {
                Kind = AccessibilityEventKind.TrackingInterrupted, FrameId = frameId, TrackId = curId,
            });
        }
        if (hasFocus && !current.PrimaryTrack!.IsLost &&
            hadFocus && (_prev?.PrimaryTrack?.IsLost ?? false) && prevId == curId)
        {
            Emit(new AccessibilityEvent
            {
                Kind = AccessibilityEventKind.TrackingRestored, FrameId = frameId, TrackId = curId,
            });
        }

        _prev = current;
    }

    public void Reset() => _prev = null;

    private void Emit(AccessibilityEvent evt)
    {
        foreach (var s in _sinks) s.OnEvent(evt);
    }
}
