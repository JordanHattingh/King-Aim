using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Validates an ONNX model by creating an ORT session, running warm-up inference,
/// and collecting output tensor statistics.
/// </summary>
public sealed class OnnxRuntimeValidator : IOnnxRuntimeValidator
{
    private static readonly RuntimeValidationOptions DefaultOptions = RuntimeValidationOptions.DefaultCpu;

    public RuntimeCompatibilityReport Validate(
        string modelPath,
        RuntimeValidationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(modelPath))
            return RuntimeCompatibilityReport.SessionFailed(
                (options ?? DefaultOptions).RequestedProvider,
                $"Model file not found: {modelPath}");

        return ValidateBytes(File.ReadAllBytes(modelPath), options,
            Path.GetFileName(modelPath), cancellationToken);
    }

    public RuntimeCompatibilityReport ValidateBytes(
        byte[] modelBytes,
        RuntimeValidationOptions? options = null,
        string sourceName = "",
        CancellationToken cancellationToken = default)
    {
        var opts = options ?? DefaultOptions;
        var errors   = new List<string>();
        var warnings = new List<string>();

        // ── Stage 1: Session creation ─────────────────────────────────────────
        InferenceSession session;
        SessionOptions? sessionOpts = null;
        string actualProvider = opts.RequestedProvider;
        try
        {
            sessionOpts = BuildSessionOptions(opts.RequestedProvider, errors, warnings, out actualProvider);
            if (errors.Count > 0)
                return RuntimeCompatibilityReport.SessionFailed(opts.RequestedProvider, errors[0]);

            session = new InferenceSession(modelBytes, sessionOpts);
        }
        catch (Exception ex)
        {
            sessionOpts?.Dispose();
            return RuntimeCompatibilityReport.SessionFailed(
                opts.RequestedProvider,
                $"Session creation failed: {ex.Message}");
        }

        using (session)
        using (sessionOpts)
        {
            // Read declared inputs via ORT metadata
            var inputs = ReadInputContracts(session);

            // ── Stage 2: Warm-up ──────────────────────────────────────────────
            List<NamedOnnxValue> inputFeed;
            try
            {
                inputFeed = BuildInputFeed(session, opts.DynamicDimensionBindings,
                    opts.MaximumGeneratedInputElements);
            }
            catch (Exception ex)
            {
                return RuntimeCompatibilityReport.WarmupFailed(
                    opts.RequestedProvider, inputs,
                    $"Could not build input tensors: {ex.Message}", warnings);
            }

            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                for (int i = 0; i < opts.WarmupRuns; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    using var _ = session.Run(inputFeed);
                }
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex)
            {
                return RuntimeCompatibilityReport.WarmupFailed(
                    opts.RequestedProvider, inputs,
                    $"Warm-up inference failed: {ex.Message}", warnings);
            }

            // ── Stage 3: Timed runs + output statistics ───────────────────────
            var latencyMs = new List<double>(opts.TimedRuns);
            IReadOnlyList<RuntimeTensorReport>? outputReports = null;

            try
            {
                for (int i = 0; i < opts.TimedRuns; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var sw = Stopwatch.StartNew();
                    using var results = session.Run(inputFeed);
                    sw.Stop();
                    latencyMs.Add(sw.Elapsed.TotalMilliseconds);

                    // Capture statistics from the last run
                    if (i == opts.TimedRuns - 1)
                        outputReports = CollectOutputReports(results);
                }
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex)
            {
                errors.Add($"Timed execution failed: {ex.Message}");
            }

            RuntimeLatencyReport? latency = null;
            if (latencyMs.Count > 0)
            {
                latencyMs.Sort();
                double median = latencyMs[latencyMs.Count / 2];
                double p99    = latencyMs[(int)(latencyMs.Count * 0.99)];
                latency = new RuntimeLatencyReport(median, p99, latencyMs.Count);
            }

            bool executionOk = errors.Count == 0 && outputReports != null;

            return new RuntimeCompatibilityReport(
                SessionCreated:    true,
                WarmupSucceeded:   true,
                ExecutionSucceeded: executionOk,
                RequestedProvider: opts.RequestedProvider,
                ActualProvider:    actualProvider,
                Inputs:            inputs,
                Outputs:           outputReports ?? [],
                Latency:           latency,
                Errors:            errors,
                Warnings:          warnings);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static SessionOptions BuildSessionOptions(
        string provider,
        List<string> errors,
        List<string> warnings,
        out string actualProvider)
    {
        actualProvider = provider;
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
        opts.EnableMemoryPattern    = false;

        if (string.Equals(provider, "DirectML", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(provider, "DML",       StringComparison.OrdinalIgnoreCase))
        {
            actualProvider = "DirectML";
            try
            {
                // GetAvailableProviders lists DmlExecutionProvider when available
                var available = OrtEnv.Instance().GetAvailableProviders();
                bool dmlPresent = Array.Exists(available,
                    p => p.Contains("Dml", StringComparison.OrdinalIgnoreCase));
                if (!dmlPresent)
                {
                    errors.Add("DmlExecutionProvider is not available on this system (no compatible GPU or driver).");
                    return opts;
                }
                opts.AppendExecutionProvider_DML(0);
                // ORT may silently fall back to CPU for unsupported ops — document this.
                warnings.Add("DirectML configured. ORT may fall back individual ops to CPU without reporting it explicitly.");
            }
            catch (Exception ex)
            {
                errors.Add($"Failed to configure DirectML EP: {ex.Message}");
            }
        }
        else if (!string.Equals(provider, "CPU", StringComparison.OrdinalIgnoreCase))
        {
            warnings.Add($"Unknown provider '{provider}'; falling back to CPU.");
            actualProvider = "CPU";
        }

        return opts;
    }

    private static IReadOnlyList<TensorContract> ReadInputContracts(InferenceSession session)
    {
        var result = new List<TensorContract>(session.InputNames.Count);
        foreach (var name in session.InputNames)
        {
            if (!session.InputMetadata.TryGetValue(name, out var nm)) continue;
            result.Add(new TensorContract(
                name,
                ElementTypeName(nm.ElementType),
                BuildDimensions(nm.Dimensions, nm.SymbolicDimensions)));
        }
        return result;
    }

    private static List<NamedOnnxValue> BuildInputFeed(
        InferenceSession session,
        IReadOnlyDictionary<string, int>? bindings,
        int maximumElements)
    {
        if (maximumElements <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumElements));

        var feed = new List<NamedOnnxValue>(session.InputNames.Count);
        try
        {
            foreach (var name in session.InputNames)
            {
                if (!session.InputMetadata.TryGetValue(name, out var nm)) continue;
                int[] shape = ResolveShape(nm.Dimensions, nm.SymbolicDimensions, bindings);
                long count = 1;
                foreach (int d in shape)
                {
                    if (d <= 0) throw new InvalidOperationException($"Input '{name}' dimension must be positive; got {d}.");
                    count = checked(count * d);
                    if (count > maximumElements)
                        throw new InvalidOperationException($"Input '{name}' requires {count:N0} elements; limit is {maximumElements:N0}.");
                }

                NamedOnnxValue value = nm.ElementType == typeof(float)
                    ? NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(new float[(int)count], shape))
                    : NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(new long[(int)count], shape));
                feed.Add(value);
            }
            return feed;
        }
        catch { throw; }
    }

    private static int[] ResolveShape(
        int[] dims, string[] syms,
        IReadOnlyDictionary<string, int>? bindings)
    {
        var result = new int[dims.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] >= 0)
            {
                result[i] = dims[i];
            }
            else
            {
                string sym = syms.Length > i ? syms[i] : "";
                if (string.IsNullOrEmpty(sym) || bindings == null || !bindings.TryGetValue(sym, out int bound))
                    throw new InvalidOperationException($"Dynamic input dimension '{(string.IsNullOrEmpty(sym) ? $"index {i}" : sym)}' requires an explicit binding.");
                result[i] = bound;
            }
        }
        return result;
    }

    private static IReadOnlyList<RuntimeTensorReport> CollectOutputReports(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        var reports = new List<RuntimeTensorReport>();
        foreach (var result in results)
        {
            if (result.Value is not DenseTensor<float> tensor)
            {
                reports.Add(new RuntimeTensorReport(result.Name, "non-float", [], 0,
                    0f, 0f, 0f, 0, 0));
                continue;
            }

            var span   = tensor.Buffer.Span;
            float  min = float.MaxValue, max = float.MinValue;
            double sum  = 0.0;
            int    nans = 0, infs = 0;

            for (int i = 0; i < span.Length; i++)
            {
                float v = span[i];
                if (float.IsNaN(v))      { nans++; continue; }
                if (float.IsInfinity(v)) { infs++; continue; }
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }

            long finite = span.Length - nans - infs;
            float mean  = finite > 0 ? (float)(sum / finite) : 0f;
            if (finite == 0) { min = 0f; max = 0f; }

            var dims = tensor.Dimensions.ToArray()
                .Select(d => new TensorDimension(d, null))
                .ToList();

            reports.Add(new RuntimeTensorReport(
                result.Name, "float32", dims, span.Length,
                min, max, mean, nans, infs));
        }
        return reports;
    }


    private static IReadOnlyList<TensorDimension> BuildDimensions(int[] dims, string[] syms)
    {
        var result = new TensorDimension[dims.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            string sym = syms.Length > i ? syms[i] : "";
            result[i] = dims[i] < 0
                ? new TensorDimension(null, string.IsNullOrEmpty(sym) ? "?" : sym)
                : new TensorDimension(dims[i], null);
        }
        return result;
    }

    private static string ElementTypeName(Type t) =>
        t == typeof(float)  ? "float32" :
        t == typeof(double) ? "float64" :
        t == typeof(int)    ? "int32"   :
        t == typeof(long)   ? "int64"   :
        t == typeof(byte)   ? "uint8"   :
        t.Name.ToLowerInvariant();
}
