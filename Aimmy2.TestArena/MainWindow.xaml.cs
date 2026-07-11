using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Aimmy2.AILogic;
using Aimmy2.Class;
using Aimmy2.Gamepad;
using Path = System.IO.Path;
using Rectangle = System.Windows.Shapes.Rectangle;

namespace Aimmy2.TestArena
{
    public partial class MainWindow : Window
    {
        private readonly DispatcherTimer _renderTimer;
        private readonly Dictionary<string, (Rectangle Rect, TextBlock Label)> _visuals = new();
        private Scenario _scenario;
        private DateTime _lastTick;

        private AIManager? _aiManager;
        private IGamepadOutput? _gamepadOutput;
        private SyntheticObservationSource? _syntheticSource;
        private double _lastRenderDtSeconds = 1.0 / 60;

        private readonly UdpClient _pointingTelemetry = new();
        private readonly IPEndPoint _pointingTelemetryEndpoint = new(IPAddress.Loopback, 28761);
        private readonly string _pointingSessionId = $"testarena_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";
        private readonly string _reportDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "KingAim", "TestArenaReports");
        private ScenarioMetricsRecorder? _metricsRecorder;
        private readonly Queue<ScenarioKind> _reportQueue = new();
        private DateTime? _reportScenarioEndsAt;
        private bool _runningAllReports;
        private static readonly TimeSpan AutomatedScenarioDuration = TimeSpan.FromSeconds(10);

        public MainWindow()
        {
            // AIManager's capture path (DXGI Desktop Duplication) requires DisplayManager to know
            // the current display before it initializes; the main Aimmy2 app does this in its own
            // startup sequence, but Test Arena runs standalone and must do it too.
            DisplayManager.Initialize();

            InitializeComponent();

            foreach (ScenarioKind kind in Enum.GetValues<ScenarioKind>())
            {
                ScenarioComboBox.Items.Add(kind);
            }
            ScenarioComboBox.SelectedIndex = 0;

            _scenario = new Scenario(ScenarioKind.StaticEnemy, 900, 700);
            _lastTick = DateTime.UtcNow;

            _renderTimer = new DispatcherTimer(DispatcherPriority.Render)
            {
                Interval = TimeSpan.FromMilliseconds(16),
            };
            _renderTimer.Tick += RenderTimer_Tick;
            _renderTimer.Start();

            Loaded += (s, e) => RebuildScenario((ScenarioKind)ScenarioComboBox.SelectedItem!);
            Closing += MainWindow_Closing;
        }

        private void ScenarioComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (ScenarioComboBox.SelectedItem is ScenarioKind kind)
            {
                RebuildScenario(kind);
            }
        }

        private void Window_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            if (IsLoaded && ScenarioComboBox.SelectedItem is ScenarioKind kind)
            {
                RebuildScenario(kind);
            }
        }

        private void Window_SourceInitialized(object? sender, EventArgs e)
        {
            // Clamp to the usable work area (screen minus taskbar) so the window can never be
            // taller/wider than the display, which otherwise pushes the top toolbar off-screen
            // on smaller displays (e.g. 1280x720) since CenterScreen centers the requested size,
            // not a size clamped to what actually fits.
            var workArea = SystemParameters.WorkArea;
            MaxWidth = workArea.Width;
            MaxHeight = workArea.Height;

            if (Width > workArea.Width) Width = workArea.Width;
            if (Height > workArea.Height) Height = workArea.Height;

            Left = Math.Max(0, workArea.Left + (workArea.Width - Width) / 2);
            Top = Math.Max(0, workArea.Top + (workArea.Height - Height) / 2);
        }

        private void RebuildScenario(ScenarioKind kind)
        {
            FlushScenarioReport();
            foreach (var visual in _visuals.Values)
            {
                ArenaCanvas.Children.Remove(visual.Rect);
                ArenaCanvas.Children.Remove(visual.Label);
            }
            _visuals.Clear();

            _syntheticSource?.Dispose();
            _syntheticSource = kind.Domain() == BenchmarkDomain.SyntheticTracking
                ? new SyntheticObservationSource(SyntheticNoiseConfig.Clean)
                : null;

            double width = ArenaCanvas.ActualWidth > 0 ? ArenaCanvas.ActualWidth : 900;
            double height = ArenaCanvas.ActualHeight > 0 ? ArenaCanvas.ActualHeight : 700;
            _scenario = new Scenario(kind, width, height);

            bool detectorExecuted = kind.Domain() == BenchmarkDomain.GameplayReplay && _aiManager != null;
            _metricsRecorder = new ScenarioMetricsRecorder(kind.ToString(), kind, detectorExecuted: detectorExecuted);

            foreach (var target in _scenario.Targets)
            {
                var rect = new Rectangle
                {
                    Width = target.Size.Width,
                    Height = target.Size.Height,
                    Fill = BrushFor(target.Kind),
                    Stroke = Brushes.White,
                    StrokeThickness = 1,
                };

                var label = new TextBlock
                {
                    Text = LabelFor(target.Kind),
                    Foreground = Brushes.White,
                    FontWeight = FontWeights.Bold,
                    FontSize = 12,
                };

                ArenaCanvas.Children.Add(rect);
                ArenaCanvas.Children.Add(label);
                _visuals[target.Id] = (rect, label);
            }
        }

        private static Brush BrushFor(TargetKind kind) => kind switch
        {
            TargetKind.Player => Brushes.DodgerBlue,
            TargetKind.Enemy => Brushes.Red,
            TargetKind.Friendly => Brushes.LimeGreen,
            _ => Brushes.Gray,
        };

        private static string LabelFor(TargetKind kind) => kind switch
        {
            TargetKind.Player => "PLAYER",
            TargetKind.Enemy => "ENEMY",
            TargetKind.Friendly => "FRIENDLY",
            _ => "?",
        };

        private void RenderTimer_Tick(object? sender, EventArgs e)
        {
            var now = DateTime.UtcNow;
            double dt = Math.Clamp((now - _lastTick).TotalSeconds, 0, 0.1);
            _lastTick = now;
            _lastRenderDtSeconds = dt > 0 ? dt : _lastRenderDtSeconds;

            _scenario.Advance(dt);

            if (_syntheticSource != null && ArenaCanvas.ActualWidth > 0)
            {
                _syntheticSource.Inject(
                    _scenario.Targets,
                    now,
                    p => { var s = ArenaCanvas.PointToScreen(p); return new System.Drawing.PointF((float)s.X, (float)s.Y); },
                    DisplayManager.ScreenLeft,
                    DisplayManager.ScreenTop,
                    DisplayManager.ScreenWidth,
                    DisplayManager.ScreenHeight);
            }

            foreach (var target in _scenario.Targets)
            {
                if (!_visuals.TryGetValue(target.Id, out var visual))
                    continue;

                bool visible = target.Visible;
                visual.Rect.Visibility = visible ? Visibility.Visible : Visibility.Hidden;
                visual.Label.Visibility = visible ? Visibility.Visible : Visibility.Hidden;

                double left = target.Position.X - target.Size.Width / 2;
                double top = target.Position.Y - target.Size.Height / 2;
                Canvas.SetLeft(visual.Rect, left);
                Canvas.SetTop(visual.Rect, top);
                Canvas.SetLeft(visual.Label, left);
                Canvas.SetTop(visual.Label, top - 16);
            }

            PublishPointingTelemetry();
            RecordScenarioMetrics(now);
            AdvanceAutomatedReports(now);
            UpdateDiagnostics();
        }

        private void RunAllReports_Click(object sender, RoutedEventArgs e)
        {
            _reportQueue.Clear();
            foreach (ScenarioKind kind in Enum.GetValues<ScenarioKind>())
                _reportQueue.Enqueue(kind);
            _runningAllReports = true;
            RunNextAutomatedScenario();
        }

        private void AdvanceAutomatedReports(DateTime now)
        {
            if (_runningAllReports && _reportScenarioEndsAt is DateTime endsAt && now >= endsAt)
            {
                FlushScenarioReport();
                RunNextAutomatedScenario();
            }
        }

        private void RunNextAutomatedScenario()
        {
            if (!_reportQueue.TryDequeue(out ScenarioKind kind))
            {
                _runningAllReports = false;
                _reportScenarioEndsAt = null;
                ReportStatusLabel.Text = $"Complete: {_reportDirectory}";
                return;
            }

            if (!Equals(ScenarioComboBox.SelectedItem, kind))
                ScenarioComboBox.SelectedItem = kind;
            else
                RebuildScenario(kind);
            _reportScenarioEndsAt = DateTime.UtcNow + AutomatedScenarioDuration;
            ReportStatusLabel.Text = $"Recording {kind} ({_reportQueue.Count + 1} remaining)";
        }

        private void RecordScenarioMetrics(DateTime timestamp)
        {
            if (_metricsRecorder == null || ArenaCanvas.ActualWidth <= 0)
                return;

            // SyntheticTracking uses injected observations; GameplayReplay requires the AI pipeline.
            IReadOnlyList<AccessibilityObservation>? observations =
                _scenario.Kind.Domain() == BenchmarkDomain.SyntheticTracking && _syntheticSource != null
                    ? _syntheticSource.CurrentObservations
                    : _aiManager?.CurrentObservations;

            if (observations == null)
                return;

            ArenaGroundTruth[] targets = _scenario.Targets.Select(target =>
            {
                Point screen = ArenaCanvas.PointToScreen(target.Position);
                return new ArenaGroundTruth(target.Id, new System.Drawing.PointF((float)screen.X, (float)screen.Y), target.Visible);
            }).ToArray();

            ArenaDetection[] detections = observations.Select(observation => new ArenaDetection(
                observation.TrackId,
                new System.Drawing.PointF(observation.Center.X, observation.Center.Y),
                observation.BoundingBoxIsExtrapolated,
                observation.ObservationAge.TotalMilliseconds,
                observation.GruPredictedCenterFraction is { } predicted
                    ? new System.Drawing.PointF(
                        (float)(SystemParameters.VirtualScreenLeft + predicted.X * SystemParameters.VirtualScreenWidth),
                        (float)(SystemParameters.VirtualScreenTop + predicted.Y * SystemParameters.VirtualScreenHeight))
                    : null)).ToArray();

            double inferenceMs = _aiManager?.InferenceMs ?? 0.0;
            double frameAgeMs = _aiManager?.FrameAge ?? 0.0;
            double captureFps = _aiManager?.CaptureFps ?? (1.0 / Math.Max(0.001, _lastRenderDtSeconds));
            _metricsRecorder.Record(timestamp, targets, detections, inferenceMs, frameAgeMs, captureFps);
        }

        private void FlushScenarioReport()
        {
            if (_metricsRecorder is { Frames: > 0 } recorder)
                ScenarioReportWriter.Write(_reportDirectory, recorder.Summarize());
            _metricsRecorder = null;
        }


        private void PublishPointingTelemetry()
        {
            // Generic TestArena pointing telemetry used only by training/record_movement.py.
            // Select the nearest visible synthetic object to the canvas centre.
            ArenaTarget? target = _scenario.Targets
                .Where(t => t.Visible)
                .OrderBy(t =>
                {
                    double dx = t.Position.X - ArenaCanvas.ActualWidth / 2.0;
                    double dy = t.Position.Y - ArenaCanvas.ActualHeight / 2.0;
                    return dx * dx + dy * dy;
                })
                .FirstOrDefault();

            if (target == null || ArenaCanvas.ActualWidth <= 0 || ArenaCanvas.ActualHeight <= 0)
                return;

            try
            {
                Point referenceScreen = ArenaCanvas.PointToScreen(new Point(
                    ArenaCanvas.ActualWidth / 2.0,
                    ArenaCanvas.ActualHeight / 2.0));
                Point targetScreen = ArenaCanvas.PointToScreen(target.Position);
                Point targetRightScreen = ArenaCanvas.PointToScreen(new Point(
                    target.Position.X + target.Size.Width / 2.0,
                    target.Position.Y));

                var message = new
                {
                    source = "testarena_pointing",
                    session_id = _pointingSessionId,
                    task_id = _scenario.Kind.ToString(),
                    dx = targetScreen.X - referenceScreen.X,
                    dy = targetScreen.Y - referenceScreen.Y,
                    target_size = Math.Abs(targetRightScreen.X - targetScreen.X) * 2.0,
                };
                byte[] payload = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(message));
                _pointingTelemetry.Send(payload, payload.Length, _pointingTelemetryEndpoint);
            }
            catch (SocketException)
            {
                // Recorder is optional. UDP delivery is deliberately best-effort.
            }
        }

        private void ModelPipelineCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            if (ModelPipelineCheckBox.IsChecked == true)
            {
                TryStartPipeline();
            }
            else
            {
                StopPipeline();
            }
        }

        private void TryStartPipeline()
        {
            try
            {
                string modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bin", "models");
                if (!System.IO.Directory.Exists(modelsDir))
                {
                    ModelPathLabel.Text = "bin/models not found";
                    ModelPipelineCheckBox.IsChecked = false;
                    return;
                }

                string[] onnxFiles = System.IO.Directory.GetFiles(modelsDir, "*.onnx");
                if (onnxFiles.Length == 0)
                {
                    ModelPathLabel.Text = "No .onnx model found in bin/models";
                    ModelPipelineCheckBox.IsChecked = false;
                    return;
                }

                string modelPath = onnxFiles[0];
                ModelPathLabel.Text = Path.GetFileName(modelPath);

                _gamepadOutput = new VirtualGamepadOutput();
                _aiManager = new AIManager(modelPath);
                _aiManager.AttachGamepadOutput(_gamepadOutput);
                _aiManager.GamepadAssistEnabled = true;
                _aiManager.GamepadTargetMode = TargetMode.TestTarget;
            }
            catch (Exception ex)
            {
                ModelPathLabel.Text = $"Pipeline failed: {ex.Message}";
                ModelPipelineCheckBox.IsChecked = false;
                StopPipeline();
            }
        }

        private void StopPipeline()
        {
            _aiManager?.Dispose();
            _aiManager = null;

            _gamepadOutput?.Dispose();
            _gamepadOutput = null;

            ModelPathLabel.Text = "(pipeline stopped)";
        }

        private void UpdateDiagnostics()
        {
            bool hasSyntheticInjection = _syntheticSource != null
                && _scenario.Kind.Domain() == BenchmarkDomain.SyntheticTracking;

            if (_aiManager == null && !hasSyntheticInjection)
            {
                DiagnosticsText.Text =
                    $"Scenario: {_scenario.Kind}\nTargets: {_scenario.Targets.Count}\n\n" +
                    "Full pipeline not running.\nEnable the checkbox above to drive\nreal screen capture + inference\nagainst this window.";
                return;
            }

            if (hasSyntheticInjection && _aiManager == null)
            {
                DiagnosticsText.Text =
                    $"Scenario: {_scenario.Kind}\n" +
                    $"Targets: {_scenario.Targets.Count}\n\n" +
                    $"Mode: Synthetic injection (YOLO bypassed)\n" +
                    $"Render FPS: {1.0 / Math.Max(0.001, _lastRenderDtSeconds):F1}\n\n" +
                    $"Active Tracks: {_syntheticSource!.ActiveTracks}\n" +
                    $"Observations: {_syntheticSource.CurrentObservations.Count}\n\n" +
                    $"Reports: {_reportDirectory}";
                return;
            }

            DiagnosticsText.Text =
                $"Scenario: {_scenario.Kind}\n" +
                $"Targets: {_scenario.Targets.Count}\n\n" +
                $"Capture FPS: {_aiManager!.CaptureFps:F1}\n" +
                $"Inference: {_aiManager.InferenceMs:F1} ms\n" +
                $"Frame Age: {_aiManager.FrameAge:F1} ms\n\n" +
                $"Players: {_aiManager.PlayerDetections}\n" +
                $"Enemies: {_aiManager.EnemyDetections}\n" +
                $"Friendlies: {_aiManager.FriendlyDetections}\n" +
                $"Active Tracks: {_aiManager.ActiveTracks}\n\n" +
                $"Selected Track: #{_aiManager.SelectedTrackId?.ToString() ?? "-"}\n" +
                $"Selected Class: {_aiManager.SelectedClass ?? "-"}\n" +
                $"ErrorX: {_aiManager.ErrorX:F2}\n" +
                $"ErrorY: {_aiManager.ErrorY:F2}\n" +
                $"TargetVel: ({_aiManager.TargetVelocityX:F1}, {_aiManager.TargetVelocityY:F1})\n" +
                $"RX: {_aiManager.RX:F2}\n" +
                $"RY: {_aiManager.RY:F2}\n" +
                $"Gamepad Connected: {_aiManager.GamepadConnected}";
            DiagnosticsText.Text += $"\nReports: {_reportDirectory}";
        }

        private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
        {
            _renderTimer.Stop();
            FlushScenarioReport();
            _pointingTelemetry.Dispose();
            _syntheticSource?.Dispose();
            StopPipeline();
        }
    }
}
