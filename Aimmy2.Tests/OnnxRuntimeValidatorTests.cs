using System.IO;
using KingAim.Core.Models.Onnx;
using Microsoft.ML.OnnxRuntime;
using Xunit;

namespace Aimmy2.Tests;

/// <summary>
/// Validates OnnxRuntimeValidator using in-memory Identity-op and Constant-op models.
/// All fixtures are generated in-process — no trained model files required.
/// </summary>
public sealed class OnnxRuntimeValidatorTests
{
    private static readonly OnnxRuntimeValidator Validator = new();

    // =========================================================================
    // Shared executable ONNX fixtures
    // =========================================================================

    private static class RuntimeFixtures
    {
        // Varint helpers
        private static byte[] Varint(ulong v)
        {
            var b = new List<byte>(10);
            do { byte cur = (byte)(v & 0x7F); v >>= 7; if (v > 0) cur |= 0x80; b.Add(cur); } while (v > 0);
            return [.. b];
        }

        // Field encoders
        private static byte[] Tag(int field, int wire) => Varint(((ulong)(uint)field << 3) | (uint)wire);
        private static byte[] I64f(int field, long v)  => [.. Tag(field, 0), .. Varint((ulong)v)];
        private static byte[] Str(int field, string s) { var b = System.Text.Encoding.UTF8.GetBytes(s); return [.. Tag(field, 2), .. Varint((ulong)b.Length), .. b]; }
        private static byte[] Msg(int field, byte[] m) => [.. Tag(field, 2), .. Varint((ulong)m.Length), .. m];
        private static byte[] F32(int field, float v)  { var b = BitConverter.GetBytes(v); return [.. Tag(field, 5), b[0], b[1], b[2], b[3]]; }

        private static byte[] Concat(params byte[][] parts)
        {
            int total = 0; foreach (var p in parts) total += p.Length;
            var buf = new byte[total]; int off = 0;
            foreach (var p in parts) { p.CopyTo(buf, off); off += p.Length; }
            return buf;
        }

        private static byte[] ValueInfo(string name, int[] shape)
        {
            byte[] dimParts = [];
            foreach (int d in shape)
            {
                byte[] dim = d < 0 ? Str(2, "dyn") : I64f(1, d);
                dimParts = [.. dimParts, .. Msg(1, dim)];
            }
            byte[] shapePb  = Msg(2, dimParts);
            byte[] tensor   = Concat(I64f(1, 1 /*float32*/), shapePb);
            byte[] typeProto = Msg(1, tensor);
            return Concat(Str(1, name), Msg(2, typeProto));
        }

        // Minimal float Identity(input→output) model
        public static byte[] Identity(
            int[] inputShape = null!, string inputName = "input",
            string outputName = "output", int opset = 17)
        {
            inputShape ??= [1, 3, 640, 640];
            byte[] node  = Concat(Str(1, inputName), Str(2, outputName), Str(4, "Identity"));
            byte[] graph = Concat(Msg(1, node), Msg(11, ValueInfo(inputName, inputShape)), Msg(12, ValueInfo(outputName, inputShape)));
            byte[] ops   = Concat(Str(1, ""), I64f(2, opset));
            return Concat(I64f(1, 8), Msg(8, ops), Msg(7, graph));
        }

        // Constant(NaN scalar) model — no inputs, one float output
        public static byte[] ConstantNan(string outputName = "nan_out", int opset = 11)
        {
            // TensorProto: dims=[1], data_type=1(float), float_data=[NaN]
            float nan     = float.NaN;
            byte[] fbytes = BitConverter.GetBytes(nan);
            byte[] tensorProto = Concat(
                I64f(1, 1),                              // dims = [1]
                I64f(2, 1),                              // data_type = float
                Msg(4, fbytes));                         // float_data (field 4, packed bytes)

            // AttributeProto: name="value", type=4(TENSOR), t=tensorProto
            // AttributeType TENSOR = 4; field 20 = type (varint tag = (20<<3)|0 = 0xA0 0x01)
            byte[] typeTag    = [0xA0, 0x01, 0x04];     // field 20 wire0, value 4
            byte[] attrProto  = Concat(Str(1, "value"), typeTag, Msg(5, tensorProto));

            byte[] node  = Concat(Str(2, outputName), Str(4, "Constant"), Msg(5, attrProto));
            byte[] graph = Concat(Msg(1, node), Msg(12, ValueInfo(outputName, [1])));
            byte[] ops   = Concat(Str(1, ""), I64f(2, opset));
            return Concat(I64f(1, 8), Msg(8, ops), Msg(7, graph));
        }

        // Garbage bytes — definitely not a valid ONNX model
        public static byte[] Garbage() => [0x01, 0x02, 0x03, 0x04, 0xFF, 0xFE];
    }

    // =========================================================================
    // Tests
    // =========================================================================

    [Fact]
    public void RuntimeValidator_CreatesCpuSession()
    {
        byte[] model  = RuntimeFixtures.Identity();
        var    report = Validator.ValidateBytes(model, RuntimeValidationOptions.DefaultCpu, "test.onnx");

        Assert.True(report.SessionCreated);
        Assert.Equal("CPU", report.RequestedProvider);
    }

    [Fact]
    public void RuntimeValidator_ExecutesWarmup()
    {
        byte[] model  = RuntimeFixtures.Identity([1, 2, 4, 4]);
        var    opts   = new RuntimeValidationOptions { WarmupRuns = 2, TimedRuns = 1 };
        var    report = Validator.ValidateBytes(model, opts);

        Assert.True(report.WarmupSucceeded);
        Assert.True(report.ExecutionSucceeded);
    }

    [Fact]
    public void RuntimeValidator_ReportsActualOutputShape()
    {
        int[] shape   = [1, 3, 8, 8];
        byte[] model  = RuntimeFixtures.Identity(shape, "x", "y");
        var    report = Validator.ValidateBytes(model);

        Assert.True(report.IsCompatible);
        Assert.Single(report.Outputs);
        var tensor = report.Outputs[0];
        Assert.Equal("y", tensor.Name);
        Assert.Equal(4, tensor.Dimensions.Count);
        Assert.Equal(1 * 3 * 8 * 8, (int)tensor.ElementCount);
    }

    [Fact]
    public void RuntimeValidator_ReportsOutputStatistics()
    {
        // Zero-filled input → Identity → all zeros output
        byte[] model  = RuntimeFixtures.Identity([1, 4]);
        var    report = Validator.ValidateBytes(model, new RuntimeValidationOptions { TimedRuns = 1 });

        Assert.True(report.IsCompatible);
        var tensor = report.Outputs[0];
        Assert.Equal(0f, tensor.MinValue);
        Assert.Equal(0f, tensor.MaxValue);
        Assert.Equal(0f, tensor.MeanValue);
        Assert.Equal(0, tensor.NanCount);
        Assert.Equal(0, tensor.InfinityCount);
    }

    [Fact]
    public void RuntimeValidator_DetectsNaNOutput()
    {
        byte[] model  = RuntimeFixtures.ConstantNan();
        var    opts   = new RuntimeValidationOptions { WarmupRuns = 0, TimedRuns = 1 };
        var    report = Validator.ValidateBytes(model, opts, "nan.onnx");

        Assert.True(report.IsCompatible, $"Session failed: {string.Join("; ", report.Errors)}");
        var output = report.Outputs[0];
        Assert.True(output.NanCount > 0, $"Expected NaN output, got nanCount={output.NanCount}");
    }

    [Fact]
    public void RuntimeValidator_RejectsInvalidGraph()
    {
        byte[] garbage = RuntimeFixtures.Garbage();
        var    report  = Validator.ValidateBytes(garbage, null, "bad.onnx");

        Assert.False(report.SessionCreated);
        Assert.False(report.IsCompatible);
        Assert.NotEmpty(report.Errors);
    }

    [Fact]
    public void RuntimeValidator_RejectsUnsupportedInputShape()
    {
        // All-dynamic dimensions with no bindings and zero count → checked overflow should fail
        // Use an extreme shape that forces a 0-count tensor (dim 0 = 0)
        byte[] model  = RuntimeFixtures.Identity([1, 0, 4, 4], "x", "y");
        var    opts   = new RuntimeValidationOptions { WarmupRuns = 1 };
        var    report = Validator.ValidateBytes(model, opts, "zero-dim.onnx");

        // ORT may reject 0-dim or produce empty output — either SessionFailed or WarmupFailed
        Assert.False(report.IsCompatible);
    }

    [Fact]
    public void RuntimeValidator_ReportsMissingProvider()
    {
        // An entirely unknown provider name falls back to CPU with a warning (not an error).
        byte[] model  = RuntimeFixtures.Identity([1, 2]);
        var    opts   = new RuntimeValidationOptions { RequestedProvider = "NonExistentEP" };
        var    report = Validator.ValidateBytes(model, opts);

        // Should succeed (CPU fallback) and report the unknown provider via warnings
        Assert.True(report.SessionCreated, $"Expected session to succeed on CPU fallback. Errors: {string.Join("; ", report.Errors)}");
        Assert.Equal("CPU", report.ActualProvider);
        Assert.Contains(report.Warnings, w => w.Contains("Unknown provider") || w.Contains("NonExistentEP"));
    }

    [Fact]
    public void RuntimeValidator_DoesNotMisreportFallbackProvider()
    {
        // When DirectML is requested but absent, the validator must NOT claim success with ActualProvider=CPU.
        // If DML is present, skip this test (the concern doesn't apply).
        var available = OrtEnv.Instance().GetAvailableProviders();
        bool dmlPresent = Array.Exists(available,
            p => p.Contains("Dml", StringComparison.OrdinalIgnoreCase));

        if (dmlPresent)
            return; // DML available — silent fallback to CPU is expected behaviour; test not applicable

        byte[] model  = RuntimeFixtures.Identity([1, 4]);
        var    report = Validator.ValidateBytes(model, RuntimeValidationOptions.DefaultDml);

        // DML not available → validator must fail, not silently run on CPU and report success
        Assert.False(report.IsCompatible);
        Assert.False(report.SessionCreated);
        Assert.NotEmpty(report.Errors);
    }

    [Fact]
    public void RuntimeValidator_RepeatedExecutionIsStable()
    {
        byte[] model = RuntimeFixtures.Identity([1, 4, 4]);
        var opts     = new RuntimeValidationOptions { WarmupRuns = 2, TimedRuns = 5 };
        var report   = Validator.ValidateBytes(model, opts);

        Assert.True(report.IsCompatible);
        Assert.NotNull(report.Latency);
        Assert.Equal(5, report.Latency.RunCount);
        Assert.True(report.Latency.MedianMs > 0);
        Assert.True(report.Latency.P99Ms >= report.Latency.MedianMs);
    }

    [Fact]
    public void RuntimeValidator_DisposesSessionAfterValidation()
    {
        // Run twice to confirm no stale session or native handle leak causes the second run to fail.
        byte[] model = RuntimeFixtures.Identity([1, 2]);
        var r1 = Validator.ValidateBytes(model);
        var r2 = Validator.ValidateBytes(model);
        Assert.True(r1.IsCompatible);
        Assert.True(r2.IsCompatible);
    }

    [Fact]
    public void RuntimeValidator_ReportsInputContracts()
    {
        int[] shape  = [1, 3, 32, 32];
        byte[] model = RuntimeFixtures.Identity(shape, "images", "preds");
        var report   = Validator.ValidateBytes(model);

        Assert.True(report.IsCompatible);
        Assert.Single(report.Inputs);
        var inp = report.Inputs[0];
        Assert.Equal("images", inp.Name);
        Assert.Equal("float32", inp.ElementType);
        Assert.Equal(4, inp.Dimensions.Count);
    }
}
