using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text.Json;
using AILogic;
using Aimmy2.AILogic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace Aimmy2.Tests
{
    /// <summary>
    /// DirectML hardware execution gate for E040 and E050 ONNX models.
    ///
    /// OPT-IN: set environment variable KINGAIM_DML_GATE=1 before running.
    /// Without that variable the tests pass vacuously and are excluded from
    /// the default portable unit-test run.  In CI and on machines without a
    /// DirectML-capable GPU, do not set the variable.
    ///
    /// Filter out of a normal test run with:
    ///   dotnet test --filter "Category!=Hardware"
    ///
    /// Path configuration (environment variables):
    ///   KINGAIM_MODELS_ROOT    — base directory for checkpoint subdirs
    ///                            (default: C:\KingAimTraining\baseline)
    ///   KINGAIM_DML_IMAGE_DIR  — directory containing DT*.png deployment images
    ///                            (default: C:\KingAimTraining\benchmark\deployment-v1)
    ///   KINGAIM_DML_REPORT_DIR — directory to write JSON reports
    ///                            (default: C:\KingAimTraining)
    ///   KINGAIM_DML_RUN_INDEX  — 1/2/3: appended to report filename so repeated runs
    ///                            do not overwrite each other.  Omit for a single run.
    ///
    /// Provider evidence:
    ///   Session is created exclusively via AppendExecutionProvider_DML(); a
    ///   missing or unsupported DML provider causes session creation to throw.
    ///   OrtEnv.GetAvailableProviders() is asserted to contain
    ///   "DmlExecutionProvider" as independent build-level confirmation.
    ///   Node-level EP assignment requires ORT profiling (SessionOptions
    ///   .EnableProfiling); enable via KINGAIM_DML_PROFILING=1 when a
    ///   per-operator breakdown is needed.
    ///
    /// Tensor allocation:
    ///   The input DenseTensor is allocated once before the measured loop.
    ///   Only session.Run() time appears in the latency numbers.
    /// </summary>
    [Trait("Category", "Hardware")]
    public sealed class DirectMlGateTests(ITestOutputHelper output)
    {
        private const int ImageSize    = 512;
        private const int WarmupIter   = 20;
        private const int MeasuredIter = 200;

        // ── path helpers ──────────────────────────────────────────────────────

        private static string ModelsRoot =>
            Environment.GetEnvironmentVariable("KINGAIM_MODELS_ROOT")
            ?? @"C:\KingAimTraining\baseline";

        private static string ImageDir =>
            Environment.GetEnvironmentVariable("KINGAIM_DML_IMAGE_DIR")
            ?? @"C:\KingAimTraining\benchmark\deployment-v1";

        private static string ReportDir =>
            Environment.GetEnvironmentVariable("KINGAIM_DML_REPORT_DIR")
            ?? @"C:\KingAimTraining";

        private static string E050OnnxPath =>
            Environment.GetEnvironmentVariable("KINGAIM_DML_E050_ONNX")
            ?? Path.Combine(ModelsRoot, "yolov8-e050", "kingaim-yolov8-baseline-e050-fp32.onnx");

        private static string E040OnnxPath =>
            Environment.GetEnvironmentVariable("KINGAIM_DML_E040_ONNX")
            ?? Path.Combine(ModelsRoot, "yolov8-e040", "kingaim-yolov8-baseline-e040-fp32.onnx");

        // ── opt-in guard ──────────────────────────────────────────────────────

        private bool IsEnabled()
        {
            if (Environment.GetEnvironmentVariable("KINGAIM_DML_GATE") != "1")
            {
                output.WriteLine(
                    "SKIP — set KINGAIM_DML_GATE=1 to enable the DirectML hardware gate.");
                return false;
            }
            return true;
        }

        // ── entry-point tests ─────────────────────────────────────────────────

        [Fact] public void DirectMl_E050_Gate() { if (IsEnabled()) RunModelGate(E050OnnxPath, "E050"); }
        [Fact] public void DirectMl_E040_Gate() { if (IsEnabled()) RunModelGate(E040OnnxPath, "E040"); }

        /// <summary>
        /// Counterbalanced latency comparison using three runs per model.
        ///
        /// Requires six report files written with KINGAIM_DML_RUN_INDEX=1/2/3:
        ///   directml_gate_e040_run1.json  directml_gate_e050_run1.json
        ///   directml_gate_e040_run2.json  directml_gate_e050_run2.json
        ///   directml_gate_e040_run3.json  directml_gate_e050_run3.json
        ///
        /// Run order (separate fresh dotnet-test process each time):
        ///   RUN_INDEX=1 E040  →  RUN_INDEX=1 E050
        ///   RUN_INDEX=2 E050  →  RUN_INDEX=2 E040
        ///   RUN_INDEX=3 E040  →  RUN_INDEX=3 E050
        ///
        /// Gate: median(E050 p95) <= median(E040 p95) × 1.10
        /// Also reports min/median/max for each model so variance is visible.
        /// </summary>
        [Fact]
        public void DirectMl_LatencyComparison_Balanced()
        {
            if (!IsEnabled()) return;

            double[] e040P95 = ReadRunP95("e040");
            double[] e050P95 = ReadRunP95("e050");
            double[] e040P50 = ReadRunP50("e040");
            double[] e050P50 = ReadRunP50("e050");

            double medE040P95 = Median(e040P95);
            double medE050P95 = Median(e050P95);
            double medE040P50 = Median(e040P50);
            double medE050P50 = Median(e050P50);

            output.WriteLine($"E040 p95  min={e040P95.Min():F2}  median={medE040P95:F2}  max={e040P95.Max():F2} ms");
            output.WriteLine($"E050 p95  min={e050P95.Min():F2}  median={medE050P95:F2}  max={e050P95.Max():F2} ms");
            output.WriteLine($"E040 p50  min={e040P50.Min():F2}  median={medE040P50:F2}  max={e040P50.Max():F2} ms");
            output.WriteLine($"E050 p50  min={e050P50.Min():F2}  median={medE050P50:F2}  max={e050P50.Max():F2} ms");

            double threshP95 = medE040P95 * 1.10;
            double threshP50 = medE040P50 * 1.10;
            output.WriteLine($"Threshold (×1.10)  p50≤{threshP50:F2}  p95≤{threshP95:F2}");

            bool p50Ok = medE050P50 <= threshP50;
            bool p95Ok = medE050P95 <= threshP95;
            output.WriteLine($"p50: {(p50Ok ? "PASS" : "FAIL")}   p95: {(p95Ok ? "PASS" : "FAIL")}");

            Assert.True(p50Ok,
                $"Median E050 p50={medE050P50:F2} ms exceeds median E040 p50 × 1.10 = {threshP50:F2} ms");
            Assert.True(p95Ok,
                $"Median E050 p95={medE050P95:F2} ms exceeds median E040 p95 × 1.10 = {threshP95:F2} ms");
        }

        private double[] ReadRunP95(string label) => ReadRunMetric(label, "latency_p95_ms");
        private double[] ReadRunP50(string label) => ReadRunMetric(label, "latency_p50_ms");

        private double[] ReadRunMetric(string label, string field)
        {
            var values = new List<double>();
            for (int i = 1; i <= 3; i++)
            {
                string path = Path.Combine(ReportDir, $"directml_gate_{label}_run{i}.json");
                Assert.True(File.Exists(path),
                    $"Run {i} report not found: {path} — run DirectMl_{label.ToUpperInvariant()}_Gate with KINGAIM_DML_RUN_INDEX={i} first.");
                using var doc = JsonDocument.Parse(File.ReadAllText(path));
                values.Add(doc.RootElement.GetProperty(field).GetDouble());
            }
            return values.ToArray();
        }

        private static double Median(double[] values)
        {
            var sorted = values.OrderBy(v => v).ToArray();
            int n = sorted.Length;
            return (n % 2 == 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        }

        // ── shared harness ────────────────────────────────────────────────────

        private void RunModelGate(string onnxPath, string label)
        {
            Assert.True(File.Exists(onnxPath), $"ONNX not found: {onnxPath}");

            // ── Confirm DML is available in this ORT build ───────────────────
            // This is a static build-level check; session creation below is
            // the runtime check that DML was actually registered for this session.
            string[] availableProviders = OrtEnv.Instance().GetAvailableProviders();
            output.WriteLine($"[{label}] Available providers: {string.Join(", ", availableProviders)}");
            Assert.Contains("DmlExecutionProvider", availableProviders);

            bool profilingEnabled =
                Environment.GetEnvironmentVariable("KINGAIM_DML_PROFILING") == "1";

            // ── Cold session creation ────────────────────────────────────────
            output.WriteLine($"[{label}] Creating session (cold)...");
            var coldSw = Stopwatch.StartNew();
            OnnxModelLoadResult loaded;
            try
            {
                loaded = OnnxModelSessionFactory.Load(onnxPath, useDirectML: true);
            }
            catch (Exception ex)
            {
                Assert.Fail($"[{label}] DirectML session creation threw: {ex.Message}");
                return;
            }
            coldSw.Stop();
            double coldCreationMs = coldSw.Elapsed.TotalMilliseconds;
            output.WriteLine($"[{label}] Cold session creation: {coldCreationMs:F1} ms");

            // ── Validate metadata and first inference on cold session ────────
            double firstInferenceMs;
            using (InferenceSession sessionProbe = loaded.Session)
            {
                Assert.True(sessionProbe.InputMetadata.ContainsKey("images"),
                    "Input 'images' not found.");
                Assert.True(sessionProbe.OutputMetadata.ContainsKey("output0"),
                    "Output 'output0' not found.");
                Assert.Equal(new[] { 1, 3, ImageSize, ImageSize },
                    sessionProbe.InputMetadata["images"].Dimensions);
                Assert.Equal(new[] { 1, 5, 5376 },
                    sessionProbe.OutputMetadata["output0"].Dimensions);

                float[] probeInput = MakeSyntheticInput(ImageSize);
                var probeT = new DenseTensor<float>(probeInput, new[] { 1, 3, ImageSize, ImageSize });
                var probeInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("images", probeT)
                };
                var firstSw = Stopwatch.StartNew();
                using var probeResults = sessionProbe.Run(probeInputs);
                firstSw.Stop();
                firstInferenceMs = firstSw.Elapsed.TotalMilliseconds;
                var probeOut = probeResults[0].AsTensor<float>();
                output.WriteLine(
                    $"[{label}] First inference: {firstInferenceMs:F1} ms  " +
                    $"shape=[{string.Join(",", probeOut.Dimensions.ToArray())}]");
                Assert.Equal(new[] { 1, 5, 5376 }, probeOut.Dimensions.ToArray());
            }
            // Cold session disposed — GPU shader cache is now warm.

            // ── Warm session creation ────────────────────────────────────────
            output.WriteLine($"[{label}] Creating session (warm)...");
            var warmSw = Stopwatch.StartNew();
            OnnxModelLoadResult warmLoaded;
            try
            {
                warmLoaded = OnnxModelSessionFactory.Load(onnxPath, useDirectML: true);
            }
            catch (Exception ex)
            {
                Assert.Fail($"[{label}] Warm session creation threw: {ex.Message}");
                return;
            }
            warmSw.Stop();
            double warmCreationMs = warmSw.Elapsed.TotalMilliseconds;
            output.WriteLine($"[{label}] Warm session creation: {warmCreationMs:F1} ms");

            using InferenceSession session = warmLoaded.Session;

            // ── Pre-allocate input tensor — reused for every measured iteration ──
            float[] inputData = MakeSyntheticInput(ImageSize);
            var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 3, ImageSize, ImageSize });
            var namedInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            // ── Warm-up ──────────────────────────────────────────────────────
            output.WriteLine($"[{label}] Warming up ({WarmupIter} iterations)...");
            for (int i = 0; i < WarmupIter; i++)
            {
                using var r = session.Run(namedInputs);
            }

            // ── Measured loop — only session.Run() is timed ─────────────────
            output.WriteLine($"[{label}] Measuring ({MeasuredIter} iterations)...");
            long wsBefore = WorkingSetBytes();
            var latencies = new List<double>(MeasuredIter);
            int nonFiniteCount = 0;
            int wrongShapeCount = 0;
            var iterSw = new Stopwatch();

            for (int i = 0; i < MeasuredIter; i++)
            {
                iterSw.Restart();
                using var results = session.Run(namedInputs);
                iterSw.Stop();
                latencies.Add(iterSw.Elapsed.TotalMilliseconds);

                var t = results[0].AsTensor<float>();
                if (!t.Dimensions.ToArray().SequenceEqual(new[] { 1, 5, 5376 }))
                    wrongShapeCount++;
                foreach (float v in t)
                    if (!float.IsFinite(v))
                        nonFiniteCount++;
            }

            long wsAfter = WorkingSetBytes();
            double wsGrowthMb = (wsAfter - wsBefore) / (1024.0 * 1024.0);

            latencies.Sort();
            double p50  = Percentile(latencies, 0.50);
            double p95  = Percentile(latencies, 0.95);
            double p99  = Percentile(latencies, 0.99);
            double maxL = latencies[^1];
            double mean = latencies.Average();

            output.WriteLine(
                $"[{label}] Latency (ms):  p50={p50:F2}  p95={p95:F2}  " +
                $"p99={p99:F2}  max={maxL:F2}  mean={mean:F2}");
            output.WriteLine($"[{label}] Working-set growth: {wsGrowthMb:F1} MB");
            output.WriteLine($"[{label}] Wrong-shape: {wrongShapeCount}/{MeasuredIter}   " +
                             $"Non-finite: {nonFiniteCount}");

            // ── Deployment images ────────────────────────────────────────────
            var imageFiles = Directory.Exists(ImageDir)
                ? Directory.GetFiles(ImageDir, "DT*.png").OrderBy(f => f).ToArray()
                : Array.Empty<string>();
            output.WriteLine($"\n[{label}] Running {imageFiles.Length} deployment images...");
            int imageErrors = 0;
            foreach (string imgPath in imageFiles)
            {
                try
                {
                    using var bmp = new Bitmap(imgPath);
                    float[] imgData = BitmapToInput(bmp, ImageSize);
                    var imgT = new DenseTensor<float>(imgData, new[] { 1, 3, ImageSize, ImageSize });
                    var imgInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("images", imgT)
                    };
                    using var imgResults = session.Run(imgInputs);
                    var outT = imgResults[0].AsTensor<float>();
                    bool shapeOk = outT.Dimensions.ToArray().SequenceEqual(new[] { 1, 5, 5376 });
                    int nf = outT.Count(v => !float.IsFinite(v));
                    output.WriteLine(
                        $"  {Path.GetFileName(imgPath),-10}  shape=[{string.Join(",", outT.Dimensions.ToArray())}]" +
                        $"  non-finite={nf}  {(shapeOk && nf == 0 ? "PASS" : "FAIL")}");
                    if (!shapeOk || nf > 0) imageErrors++;
                }
                catch (Exception ex)
                {
                    output.WriteLine($"  {Path.GetFileName(imgPath),-10}  EXCEPTION: {ex.Message}");
                    imageErrors++;
                }
            }

            // ── Write report ─────────────────────────────────────────────────
            Directory.CreateDirectory(ReportDir);
            string runSuffix = Environment.GetEnvironmentVariable("KINGAIM_DML_RUN_INDEX") is string idx
                               && int.TryParse(idx, out int runNum) && runNum >= 1
                               ? $"_run{runNum}" : string.Empty;
            string reportPath = Path.Combine(ReportDir, $"directml_gate_{label.ToLowerInvariant()}{runSuffix}.json");
            var report = new
            {
                model                    = onnxPath,
                label,
                ort_version              = typeof(InferenceSession).Assembly.GetName().Version?.ToString(),
                available_providers      = availableProviders,
                provider                 = "DmlExecutionProvider",
                cpu_fallback_accepted    = false,
                cold_session_creation_ms = Math.Round(coldCreationMs, 2),
                warm_session_creation_ms = Math.Round(warmCreationMs, 2),
                first_inference_ms       = Math.Round(firstInferenceMs, 2),
                warmup_iterations        = WarmupIter,
                measured_iterations      = MeasuredIter,
                latency_p50_ms           = Math.Round(p50,  3),
                latency_p95_ms           = Math.Round(p95,  3),
                latency_p99_ms           = Math.Round(p99,  3),
                latency_max_ms           = Math.Round(maxL, 3),
                latency_mean_ms          = Math.Round(mean, 3),
                working_set_growth_mb    = Math.Round(wsGrowthMb, 2),
                wrong_shape_count        = wrongShapeCount,
                non_finite_count         = nonFiniteCount,
                deployment_images_run    = imageFiles.Length,
                deployment_image_errors  = imageErrors,
                verdict                  =
                    (nonFiniteCount == 0 && wrongShapeCount == 0 && imageErrors == 0)
                    ? "PASS" : "FAIL",
            };
            File.WriteAllText(reportPath, JsonSerializer.Serialize(
                report, new JsonSerializerOptions { WriteIndented = true }));
            output.WriteLine($"\n[{label}] Report: {reportPath}");
            if (profilingEnabled)
                output.WriteLine(
                    $"[{label}] Note: re-run with KINGAIM_DML_PROFILING=1 and " +
                    $"SessionOptions.EnableProfiling() to capture per-node EP assignments.");

            // ── Structural assertions (latency captured in report, not asserted here) ──
            Assert.Equal(0, nonFiniteCount);
            Assert.Equal(0, wrongShapeCount);
            Assert.Equal(0, imageErrors);
        }

        // ── helpers ───────────────────────────────────────────────────────────

        private static float[] MakeSyntheticInput(int imageSize)
        {
            int n = 3 * imageSize * imageSize;
            var arr = new float[n];
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < imageSize; y++)
                    for (int x = 0; x < imageSize; x++)
                        arr[c * imageSize * imageSize + y * imageSize + x] = 0.5f;
            return arr;
        }

        private static float[] BitmapToInput(Bitmap bmp, int imageSize)
        {
            using var resized = new Bitmap(bmp, imageSize, imageSize);
            var arr = new float[3 * imageSize * imageSize];
            MathUtil.BitmapToFloatArrayInPlace(resized, arr, imageSize);
            return arr;
        }

        private static double Percentile(IReadOnlyList<double> sorted, double p)
        {
            if (sorted.Count == 0) return 0;
            double idx = p * (sorted.Count - 1);
            int lo = (int)idx;
            int hi = Math.Min(lo + 1, sorted.Count - 1);
            return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
        }

        private static long WorkingSetBytes() =>
            Process.GetCurrentProcess().WorkingSet64;
    }
}
