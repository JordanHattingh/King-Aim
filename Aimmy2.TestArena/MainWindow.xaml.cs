using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Aimmy2.AILogic;
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

        public MainWindow()
        {
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

        private void RebuildScenario(ScenarioKind kind)
        {
            foreach (var visual in _visuals.Values)
            {
                ArenaCanvas.Children.Remove(visual.Rect);
                ArenaCanvas.Children.Remove(visual.Label);
            }
            _visuals.Clear();

            double width = ArenaCanvas.ActualWidth > 0 ? ArenaCanvas.ActualWidth : 900;
            double height = ArenaCanvas.ActualHeight > 0 ? ArenaCanvas.ActualHeight : 700;
            _scenario = new Scenario(kind, width, height);

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

            _scenario.Advance(dt);

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

            UpdateDiagnostics();
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
            if (_aiManager == null)
            {
                DiagnosticsText.Text =
                    $"Scenario: {_scenario.Kind}\nTargets: {_scenario.Targets.Count}\n\n" +
                    "Full pipeline not running.\nEnable the checkbox above to drive\nreal screen capture + inference\nagainst this window.";
                return;
            }

            DiagnosticsText.Text =
                $"Scenario: {_scenario.Kind}\n" +
                $"Targets: {_scenario.Targets.Count}\n\n" +
                $"Capture FPS: {_aiManager.CaptureFps:F1}\n" +
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
        }

        private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
        {
            _renderTimer.Stop();
            StopPipeline();
        }
    }
}
