using System.IO;
using System.Text;
using KingAim.Core.Models.Onnx;
using Xunit;

namespace Aimmy2.Tests;

/// <summary>
/// Tests for OnnxModelInspector and OnnxContractJsonWriter.
/// All fixtures are generated in memory — no real trained model files are read.
/// </summary>
public sealed class OnnxInspectorTests
{
    private static readonly OnnxModelInspector Inspector = new();

    // =========================================================================
    // Minimal ONNX model builder — pure in-memory protobuf encoding.
    // Produces valid Identity-op models for inspection tests without any
    // external files or third-party protobuf libraries.
    // =========================================================================

    private static class MinimalOnnxBuilder
    {
        /// <summary>
        /// Build a minimal valid ONNX model (Identity op) from given I/O specs.
        /// </summary>
        public static byte[] Build(
            string inputName,
            int[]  inputShape,
            string outputName,
            int[]? outputShape    = null,
            bool   dynamicBatch   = false,
            int    opsetVersion   = 17,
            string producerName   = "test")
        {
            int[] outShape = outputShape ?? inputShape;

            byte[] nodeProto = Concat(
                Str(1, inputName),
                Str(2, outputName),
                Str(4, "Identity"));

            byte[] inputVI  = ValueInfo(inputName,  inputShape, dynamicBatch);
            byte[] outputVI = ValueInfo(outputName, outShape,   dynamicBatch);

            byte[] graph = Concat(
                Msg(1, nodeProto),
                Msg(11, inputVI),
                Msg(12, outputVI));

            byte[] opset = Concat(Str(1, ""), I64(2, opsetVersion));

            byte[] model = Concat(
                I64(1, 8),                     // ir_version
                Msg(8, opset),                 // opset_import  (ModelProto field 8)
                Msg(7, graph));                // graph         (ModelProto field 7)

            if (!string.IsNullOrEmpty(producerName))
                model = Concat(model, Str(2, producerName)); // producer_name (field 2)

            return model;
        }

        /// <summary>Build a model with multiple outputs, e.g. to simulate multi-head decoders.</summary>
        public static byte[] BuildMultiOutput(
            string inputName,
            int[]  inputShape,
            (string name, int[] shape)[] outputs,
            int opsetVersion = 17)
        {
            // Wire first output only through Identity; extra outputs are metadata-only
            var parts = new List<byte[]>();

            byte[] nodeProto = Concat(
                Str(1, inputName),
                Str(2, outputs[0].name),
                Str(4, "Identity"));
            parts.Add(Msg(1, nodeProto));

            parts.Add(Msg(11, ValueInfo(inputName, inputShape, false)));
            foreach (var (name, shape) in outputs)
                parts.Add(Msg(12, ValueInfo(name, shape, false)));

            byte[] graph  = Concat(parts.ToArray());
            byte[] opset  = Concat(Str(1, ""), I64(2, opsetVersion));
            return Concat(I64(1, 8), Msg(8, opset), Msg(7, graph));
        }

        // ---- shape encoders ----

        private static byte[] ValueInfo(string name, int[] shape, bool dynBatch)
        {
            return Concat(Str(1, name), Msg(2, TypeProto(shape, dynBatch)));
        }

        private static byte[] TypeProto(int[] shape, bool dynBatch)
            => Msg(1, TensorTypeProto(shape, dynBatch));

        private static byte[] TensorTypeProto(int[] shape, bool dynBatch)
            => Concat(I64(1, 1 /*float32*/), Msg(2, ShapeProto(shape, dynBatch)));

        private static byte[] ShapeProto(int[] shape, bool dynBatch)
        {
            var dims = shape.Select((v, i) =>
                Msg(1, dynBatch && i == 0
                    ? Str(2, "batch_size")     // symbolic/dynamic dim
                    : I64(1, v)));             // fixed dim
            return Concat(dims.ToArray());
        }

        // ---- protobuf primitives ----

        private static byte[] Varint(ulong v)
        {
            var buf = new List<byte>(10);
            while (v >= 0x80) { buf.Add((byte)(v | 0x80)); v >>= 7; }
            buf.Add((byte)v);
            return [.. buf];
        }

        private static byte[] Tag(int field, int wire) => Varint((ulong)(field << 3 | wire));

        private static byte[] Msg(int field, byte[] content)
            => Concat(Tag(field, 2), Varint((ulong)content.Length), content);

        private static byte[] Str(int field, string s) => Msg(field, Encoding.UTF8.GetBytes(s));

        private static byte[] I64(int field, long v) => Concat(Tag(field, 0), Varint((ulong)v));

        private static byte[] Concat(params byte[][] parts)
        {
            int total = parts.Sum(p => p.Length);
            var buf   = new byte[total];
            int pos   = 0;
            foreach (var p in parts) { p.CopyTo(buf, pos); pos += p.Length; }
            return buf;
        }
    }

    // =========================================================================
    // Tests
    // =========================================================================

    [Fact]
    public void OnnxInspector_ReadsInputTensorMetadata()
    {
        byte[] model = MinimalOnnxBuilder.Build("images", [1, 3, 512, 512], "output0");
        var contract = Inspector.InspectBytes(model, "test.onnx");

        Assert.Single(contract.Inputs);
        var input = contract.Inputs[0];
        Assert.Equal("images",   input.Name);
        Assert.Equal("float32",  input.ElementType);
        Assert.Equal(4, input.Dimensions.Count);
        Assert.Equal(512, input.Dimensions[2].FixedValue);
        Assert.Equal(512, input.Dimensions[3].FixedValue);
    }

    [Fact]
    public void OnnxInspector_ReadsAllOutputTensors()
    {
        byte[] model = MinimalOnnxBuilder.BuildMultiOutput(
            "images", [1, 3, 512, 512],
            [("output0", [1, 17, 5376]), ("output1", [1, 4, 5376])]);

        var contract = Inspector.InspectBytes(model, "test.onnx");

        Assert.Equal(2, contract.Outputs.Count);
        Assert.Contains(contract.Outputs, t => t.Name == "output0");
        Assert.Contains(contract.Outputs, t => t.Name == "output1");
    }

    [Fact]
    public void OnnxInspector_PreservesDynamicDimensions()
    {
        byte[] model = MinimalOnnxBuilder.Build(
            "images", [1, 3, 512, 512], "output0",
            dynamicBatch: true);

        var contract = Inspector.InspectBytes(model, "test.onnx");

        var batchDim = contract.Inputs[0].Dimensions[0];
        Assert.True(batchDim.IsDynamic, "Batch dimension should be dynamic");
        Assert.Equal("batch_size", batchDim.SymbolicName);
    }

    [Fact]
    public void OnnxInspector_ComputesStableSha256()
    {
        byte[] model = MinimalOnnxBuilder.Build("images", [1, 3, 512, 512], "output0");

        var c1 = Inspector.InspectBytes(model, "a.onnx");
        var c2 = Inspector.InspectBytes(model, "b.onnx");   // same bytes, different name

        Assert.Equal(c1.Sha256, c2.Sha256);
        Assert.Equal(64, c1.Sha256.Length);  // hex SHA-256 = 64 chars
        Assert.Matches("^[0-9a-f]+$", c1.Sha256);
    }

    [Fact]
    public void OnnxInspector_ReportsMissingModel()
    {
        var ex = Assert.Throws<FileNotFoundException>(
            () => Inspector.Inspect(@"C:\does\not\exist\model.onnx"));
        Assert.Contains("not found", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void OnnxInspector_RejectsInvalidOnnxFile()
    {
        byte[] garbage = [0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE];
        Assert.ThrowsAny<Exception>(() => Inspector.InspectBytes(garbage, "bad.onnx"));
    }

    [Fact]
    public void OnnxInspector_WritesDeterministicJson()
    {
        byte[] model = MinimalOnnxBuilder.Build("images", [1, 3, 512, 512], "output0");
        var contract = Inspector.InspectBytes(model, "det.onnx");

        string json1 = OnnxContractJsonWriter.Write(contract);
        string json2 = OnnxContractJsonWriter.Write(contract);

        Assert.Equal(json1, json2);
        Assert.Contains("\"modelPath\"",    json1);
        Assert.Contains("\"sha256\"",       json1);
        Assert.Contains("\"opsetVersion\"", json1);
        Assert.Contains("\"float32\"",      json1);
    }

    [Fact]
    public void CompatibilityReport_RejectsWrongInputSize()
    {
        var report = new TestDecoder(expectedInputShape: [1, 3, 512, 512])
            .CheckCompatibility(ContractFor([1, 3, 640, 640], [1, 17, 8400]));

        Assert.False(report.IsCompatible);
        Assert.NotEmpty(report.Errors);
    }

    [Fact]
    public void CompatibilityReport_RejectsWrongElementType()
    {
        // Build a float64 model (elem_type = 11 in ONNX, but our builder uses 1=float32)
        // We simulate this by constructing a contract manually.
        var contract = new OnnxModelContract(
            "test.onnx", "abc", 100, 17,
            Inputs:  [new TensorContract("images",  "float64", Dims([1, 3, 512, 512]))],
            Outputs: [new TensorContract("output0", "float64", Dims([1, 17, 5376]))],
            Metadata: new Dictionary<string, string>());

        var report = new TestDecoder(expectedInputShape: [1, 3, 512, 512])
            .CheckCompatibility(contract);

        Assert.False(report.IsCompatible);
        Assert.Contains(report.Errors, e => e.Contains("float32", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void CompatibilityReport_RejectsUnknownOutputShape()
    {
        // Output has 3 dimensions instead of expected 3D pose tensor
        var report = new TestDecoder(expectedInputShape: [1, 3, 512, 512])
            .CheckCompatibility(ContractFor([1, 3, 512, 512], [1, 4]));

        Assert.False(report.IsCompatible);
        Assert.NotEmpty(report.Errors);
    }

    [Fact]
    public void CompatibilityReport_AcceptsKnownFixture()
    {
        // Exact matching shape and type → compatible
        var report = new TestDecoder(expectedInputShape: [1, 3, 512, 512])
            .CheckCompatibility(ContractFor([1, 3, 512, 512], [1, 17, 5376]));

        Assert.True(report.IsCompatible);
        Assert.Empty(report.Errors);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static OnnxModelContract ContractFor(int[] inputShape, int[] outputShape)
        => new(
            "test.onnx", "sha256", 100, 17,
            Inputs:  [new TensorContract("images",  "float32", Dims(inputShape))],
            Outputs: [new TensorContract("output0", "float32", Dims(outputShape))],
            Metadata: new Dictionary<string, string>());

    private static IReadOnlyList<TensorDimension> Dims(int[] shape) =>
        shape.Select(v => new TensorDimension(v, null)).ToList();

    // Minimal decoder stub used only in compatibility tests
    private sealed class TestDecoder : KingAim.Core.Decoding.IModelDecoder
    {
        private readonly int[] _expectedInputShape;

        public TestDecoder(int[] expectedInputShape)
            => _expectedInputShape = expectedInputShape;

        public string DecoderId => "test";

        public DecoderCompatibilityReport CheckCompatibility(OnnxModelContract contract)
        {
            var errors = new List<string>();
            var obs    = new List<string>();

            var inp = contract.PrimaryInput;
            if (inp == null)
            {
                errors.Add("No input tensor found.");
                return DecoderCompatibilityReport.Incompatible(DecoderId, errors);
            }

            if (inp.ElementType != "float32")
                errors.Add($"Input element type must be float32, got {inp.ElementType}.");

            var shape = inp.FixedShape;
            if (!shape.SequenceEqual(_expectedInputShape))
                errors.Add($"Input shape mismatch. Expected [{string.Join(",", _expectedInputShape)}], got [{string.Join(",", shape)}].");

            var outp = contract.PrimaryOutput;
            if (outp == null)
            {
                errors.Add("No output tensor found.");
            }
            else if (outp.Dimensions.Count != 3)
            {
                errors.Add($"Expected 3D output tensor, got {outp.Dimensions.Count}D.");
            }
            else
            {
                obs.Add($"Output shape: [{string.Join(",", outp.FixedShape)}]");
            }

            return errors.Count > 0
                ? DecoderCompatibilityReport.Incompatible(DecoderId, errors, observations: obs)
                : DecoderCompatibilityReport.Compatible(DecoderId, observations: obs);
        }

        public void ValidateOutputContract(KingAim.Core.Inference.InferenceOutput output) { }

        public IReadOnlyList<KingAim.Core.Perception.PoseDetection> Decode(
            KingAim.Core.Inference.InferenceOutput output,
            KingAim.Core.Preprocessing.PreprocessingMetadata preprocessing) => [];
    }
}
