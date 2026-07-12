using System.Diagnostics;
using Aimmy2.Gamepad;
using KingAim.Core.Accessibility;
using KingAim.Core.AppHost;
using KingAim.Core.Capture;
using KingAim.Core.Diagnostics;
using KingAim.Core.Pipeline;
using KingAim.Core.Safety;
using KingAim.Core.Scheduling;
using KingAim.Core.Tracking;

namespace Aimmy2.AppHost;

/// <summary>
/// Concrete <see cref="IApplicationHost"/> that drives two concurrent loops:
/// <list type="bullet">
///   <item>A <b>pipeline loop</b> calling <see cref="PipelineRunner.TickAsync"/> for
///         capture → inference → tracking → cue dispatch.</item>
///   <item>A <b>gamepad loop</b> (~250 Hz) that blends the AI assist delta with physical
///         XInput and forwards the result to a virtual controller.</item>
/// </list>
/// Construct once per session; call <see cref="StartAsync"/> to begin.
/// </summary>
public sealed class KingAimApplicationHost : IApplicationHost
{
    private readonly XInputReader    _xInput;
    private readonly IGamepadOutput? _gamepadOutput;
    private uint                     _gamepadIndex;

    private PipelineRunner?          _runner;
    private IFrameSource?            _frameSource;
    private SafetyPolicy?            _safety;
    private CancellationTokenSource? _cts;
    private Task?                    _pipelineTask;
    private Task?                    _gamepadTask;

    // Latest focus from the pipeline loop — published for the gamepad loop.
    // Use Interlocked.Exchange for atomic reference swap on a plain reference field.
    private TrackState?              _lastFocus;
    private int                      _captureWidth;
    private int                      _captureHeight;

    // ── Diagnostics properties (UI-facing) ────────────────────────────────────
    private volatile bool _physicalConnected;
    private volatile bool _assistActive;
    private float         _lastRx;
    private float         _lastRy;

    /// <summary>Whether a physical XInput controller is currently connected.</summary>
    public bool PhysicalControllerConnected => _physicalConnected;

    /// <summary>True when the last gamepad loop iteration produced non-zero assist.</summary>
    public bool AssistIsActive => _assistActive;

    /// <summary>Last blended right-stick X sent to the virtual controller.</summary>
    public float LastRx => _lastRx;

    /// <summary>Last blended right-stick Y sent to the virtual controller.</summary>
    public float LastRy => _lastRy;

    /// <summary>Latest focus track from the most recent pipeline tick.</summary>
    public TrackState? LatestFocus => Interlocked.CompareExchange(ref _lastFocus, null, null);

    // ── IApplicationHost ───────────────────────────────────────────────────────
    public SafetyPolicy Safety =>
        _safety ?? throw new InvalidOperationException("Host has not been started.");

    public SystemHealthState Health { get; private set; } = SystemHealthState.Healthy;
    public bool IsRunning { get; private set; }

    // ── Constructor ────────────────────────────────────────────────────────────

    /// <param name="xInput">Physical controller reader; a default instance is created when null.</param>
    /// <param name="gamepadOutput">Virtual controller output. May be null (assist values are still
    ///     computed and exposed via diagnostics).</param>
    /// <param name="gamepadIndex">Initial XInput player index (0–3).</param>
    public KingAimApplicationHost(
        XInputReader?   xInput        = null,
        IGamepadOutput? gamepadOutput = null,
        uint            gamepadIndex  = 0)
    {
        _xInput       = xInput ?? new XInputReader();
        _gamepadOutput = gamepadOutput;
        _gamepadIndex  = gamepadIndex;
    }

    // ── IApplicationHost implementation ────────────────────────────────────────

    public async Task StartAsync(PipelineDescriptor pipeline, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (IsRunning)
            throw new InvalidOperationException("Host is already running.");

        _safety       = pipeline.Safety;
        _frameSource  = pipeline.FrameSource;
        _captureWidth  = pipeline.FrameSource.Width;
        _captureHeight = pipeline.FrameSource.Height;

        _runner = new PipelineRunner(
            pipeline.Safety,
            pipeline.FrameSource,
            new FrameScheduler(),
            pipeline.Preprocessor,
            pipeline.Engine,
            pipeline.Decoder,
            pipeline.Validator,
            pipeline.Tracker,
            pipeline.Scene,
            pipeline.Focus,
            new AccessibilityEventDispatcher(),
            pipeline.Visual,
            pipeline.Audio,
            pipeline.Haptic,
            pipeline.Pointing,
            pipeline.Drift,
            pipeline.GamepadAssist,
            pipeline.Diagnostics);

        await pipeline.FrameSource.StartAsync(ct).ConfigureAwait(false);

        _gamepadOutput?.Connect();

        _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        CancellationToken loopCt = _cts.Token;

        _pipelineTask = Task.Factory.StartNew(
            () => PipelineLoopAsync(loopCt),
            loopCt,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default).Unwrap();

        _gamepadTask = Task.Factory.StartNew(
            () => GamepadLoopAsync(loopCt),
            loopCt,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default).Unwrap();

        IsRunning = true;
        Health    = SystemHealthState.Healthy;
    }

    public async Task StopAsync(CancellationToken ct = default)
    {
        if (!IsRunning) return;

        _cts?.Cancel();

        var tasks = new[] { _pipelineTask, _gamepadTask }
            .Where(t => t != null)
            .Cast<Task>()
            .ToArray();

        if (tasks.Length > 0)
        {
            try
            {
                await Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(5), ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException) { }
            catch (TimeoutException) { }
        }

        _gamepadOutput?.Disconnect();

        try { await (_frameSource?.StopAsync(ct) ?? Task.CompletedTask).ConfigureAwait(false); }
        catch { }

        IsRunning = false;
    }

    public void EmergencyStop()
    {
        _runner?.EmergencyStop();
        _assistActive = false;
    }

    public Task SwapModelAsync(string modelId, CancellationToken ct = default)
        => throw new NotSupportedException(
            "Model swapping is not supported in this host implementation.");

    // ── Loops ─────────────────────────────────────────────────────────────────

    private async Task PipelineLoopAsync(CancellationToken ct)
    {
        try { Thread.CurrentThread.Priority = ThreadPriority.AboveNormal; } catch { }

        while (!ct.IsCancellationRequested)
        {
            try
            {
                PipelineRunResult result = await _runner!.TickAsync(ct).ConfigureAwait(false);
                Interlocked.Exchange(ref _lastFocus, result.Focus);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                Health = SystemHealthState.Degraded;
                System.Diagnostics.Debug.WriteLine($"[KingAimApplicationHost] Pipeline loop error: {ex.Message}");
                await Task.Delay(50, ct).ConfigureAwait(false);
            }
        }
    }

    private async Task GamepadLoopAsync(CancellationToken ct)
    {
        var sw     = Stopwatch.StartNew();
        var scanSw = Stopwatch.StartNew();
        const double ScanIntervalSeconds = 5.0;

        while (!ct.IsCancellationRequested)
        {
            // Clamp dt to absorb system wake/sleep spikes (>100ms treated as first frame).
            double dt = Math.Clamp(sw.Elapsed.TotalSeconds, 0.0, 0.1);
            sw.Restart();

            // Periodic physical controller re-discovery.
            if (scanSw.Elapsed.TotalSeconds >= ScanIntervalSeconds)
            {
                scanSw.Restart();
                uint idx = _xInput.FindFirstConnectedIndex();
                if (_xInput.Read(idx).Connected)
                    _gamepadIndex = idx;
            }

            PhysicalGamepadState physical = _xInput.Read(_gamepadIndex);
            _physicalConnected = physical.Connected;

            PipelineRunner? runner = _runner;
            if (runner != null)
            {
                TrackState? focus = Interlocked.CompareExchange(ref _lastFocus, null, null);

                var (assistRx, assistRy) = runner.GetGamepadAssistDelta(
                    focus, _captureWidth, _captureHeight, dt);

                _assistActive = runner.AssistIsActive;

                if (runner.GamepadAssistEnabled)
                {
                    var (rx, ry) = StickBlender.Blend(
                        physical.Connected ? physical.RightStickX : 0f,
                        physical.Connected ? physical.RightStickY : 0f,
                        assistRx,
                        assistRy);

                    _lastRx = rx;
                    _lastRy = ry;

                    if (_gamepadOutput?.IsConnected == true)
                        _gamepadOutput.SetFullState(physical, rx, ry);
                }
                else
                {
                    // Pass-through: physical input only, no assist.
                    _lastRx = physical.RightStickX;
                    _lastRy = physical.RightStickY;

                    if (_gamepadOutput?.IsConnected == true && physical.Connected)
                        _gamepadOutput.SetPassthroughState(physical);
                }
            }

            try
            {
                await Task.Delay(4, ct).ConfigureAwait(false);  // ~250 Hz
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }
    }

    // ── IAsyncDisposable ───────────────────────────────────────────────────────

    private bool _disposed;

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        await StopAsync().ConfigureAwait(false);
        _cts?.Dispose();
        _cts = null;
    }
}
