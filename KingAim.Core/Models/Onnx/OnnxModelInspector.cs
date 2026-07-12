using System.Security.Cryptography;

namespace KingAim.Core.Models.Onnx;

/// <inheritdoc/>
public sealed class OnnxModelInspector : IOnnxModelInspector
{
    public OnnxModelContract Inspect(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("ONNX model file not found.", modelPath);

        byte[] bytes = File.ReadAllBytes(modelPath);
        return InspectBytes(bytes, Path.GetFileName(modelPath));
    }

    public OnnxModelContract InspectBytes(byte[] modelBytes, string sourceName = "")
    {
        ArgumentNullException.ThrowIfNull(modelBytes);
        if (modelBytes.Length == 0)
            throw new ArgumentException("Model byte array is empty.", nameof(modelBytes));

        // Validate structure via OnnxProtobufParser first. Garbage bytes will either
        // produce empty/nonsensical metadata (caught below) or throw a parse exception.
        string sha256  = ComputeSha256Hex(modelBytes);
        var    parsed  = OnnxProtobufParser.Parse(modelBytes);

        // Reject models that produce no graph (pure garbage / non-ONNX bytes)
        if (parsed.Inputs.Count == 0 && parsed.Outputs.Count == 0 && parsed.OpsetVersion == 0)
            throw new FormatException($"File '{sourceName}' does not appear to be a valid ONNX model.");

        return new OnnxModelContract(
            ModelPath:     sourceName,
            Sha256:        sha256,
            FileSizeBytes: modelBytes.Length,
            OpsetVersion:  parsed.OpsetVersion,
            Inputs:        parsed.Inputs,
            Outputs:       parsed.Outputs,
            Metadata:      parsed.Metadata);
    }

    // -------------------------------------------------------------------------

    private static string ComputeSha256Hex(byte[] bytes)
    {
        Span<byte> hash = stackalloc byte[32];
        SHA256.HashData(bytes, hash);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}

// ---------------------------------------------------------------------------
// Reads ONNX ModelProto fields directly from protobuf bytes.
// Covers: opset_imports, graph.input, graph.output, model_metadata.
// ---------------------------------------------------------------------------
internal static class OnnxProtobufParser
{
    // ONNX DataType enum → friendly name
    private static readonly string[] _elemTypes =
        ["undefined","float32","uint8","int8","uint16","int16",
         "int32","int64","string","bool","float16","float64",
         "uint32","uint64","complex64","complex128","bfloat16"];

    internal sealed class ParsedModel
    {
        public int OpsetVersion { get; set; }
        public List<TensorContract> Inputs  { get; } = [];
        public List<TensorContract> Outputs { get; } = [];
        public Dictionary<string, string> Metadata { get; } = new(StringComparer.Ordinal);
    }

    public static ParsedModel Parse(ReadOnlySpan<byte> data)
    {
        var result = new ParsedModel();
        int pos = 0;
        while (pos < data.Length)
        {
            ulong tag  = ReadVarint(data, ref pos);
            int field  = (int)(tag >> 3);
            int wire   = (int)(tag & 7);

            if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                var sub = data.Slice(pos, len);

                switch (field)
                {
                    // ModelProto field numbers (onnx.proto):
                    // 1=ir_version, 2=producer_name, 3=producer_version, 4=domain,
                    // 5=model_version, 6=doc_string, 7=graph, 8=opset_import, 14=metadata_props
                    case 8:  // opset_import (OperatorSetIdProto)
                        int v = ReadOpsetVersion(sub);
                        if (v > 0) result.OpsetVersion = v;
                        break;
                    case 7:  // graph (GraphProto)
                        ReadGraph(sub, result);
                        break;
                    case 14: // metadata_props (StringStringEntryProto)
                        ReadMetadataEntry(sub, result.Metadata);
                        break;
                    case 2:  // producer_name
                        result.Metadata["producer_name"] = ReadUtf8(sub);
                        break;
                    case 3:  // producer_version
                        result.Metadata["producer_version"] = ReadUtf8(sub);
                        break;
                    case 4:  // domain
                        { var s = ReadUtf8(sub); if (!string.IsNullOrEmpty(s)) result.Metadata["domain"] = s; }
                        break;
                }
                pos += len;
            }
            else if (wire == 0)
            {
                ReadVarint(data, ref pos);
            }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }
        return result;
    }

    private static void ReadGraph(ReadOnlySpan<byte> data, ParsedModel result)
    {
        int pos = 0;
        while (pos < data.Length)
        {
            ulong tag = ReadVarint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);

            if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                var sub = data.Slice(pos, len);

                if (field == 11)       // graph.input
                    result.Inputs.Add(ReadValueInfo(sub));
                else if (field == 12)  // graph.output
                    result.Outputs.Add(ReadValueInfo(sub));
                // field 2 = name, field 10 = doc_string — skip
                pos += len;
            }
            else if (wire == 0) { ReadVarint(data, ref pos); }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }
    }

    private static TensorContract ReadValueInfo(ReadOnlySpan<byte> data)
    {
        // ValueInfoProto: field 1=name, field 2=type (TypeProto)
        string name    = "";
        int    elemType = 0;
        var    dims    = new List<TensorDimension>();

        int pos = 0;
        while (pos < data.Length)
        {
            ulong tag = ReadVarint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);

            if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                var sub = data.Slice(pos, len);

                if (field == 1) name = ReadUtf8(sub);
                else if (field == 2) ReadTypeProto(sub, ref elemType, dims);
                pos += len;
            }
            else if (wire == 0) { ReadVarint(data, ref pos); }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }

        string et = elemType >= 1 && elemType < _elemTypes.Length
            ? _elemTypes[elemType]
            : "unknown";

        return new TensorContract(name, et, dims);
    }

    private static void ReadTypeProto(ReadOnlySpan<byte> data, ref int elemType, List<TensorDimension> dims)
    {
        // TypeProto: field 1 = tensor_type (Tensor)
        int pos = 0;
        while (pos < data.Length)
        {
            ulong tag = ReadVarint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);

            if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                var sub = data.Slice(pos, len);

                if (field == 1) ReadTensorType(sub, ref elemType, dims);  // tensor_type
                pos += len;
            }
            else if (wire == 0) { ReadVarint(data, ref pos); }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }
    }

    private static void ReadTensorType(ReadOnlySpan<byte> data, ref int elemType, List<TensorDimension> dims)
    {
        // TypeProto.Tensor: field 1 = elem_type, field 2 = shape (TensorShapeProto)
        int pos = 0;
        while (pos < data.Length)
        {
            ulong tag = ReadVarint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);

            if (wire == 0)
            {
                long val = (long)ReadVarint(data, ref pos);
                if (field == 1) elemType = (int)val;
            }
            else if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                var sub = data.Slice(pos, len);

                if (field == 2) ReadShape(sub, dims);  // shape
                pos += len;
            }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }
    }

    private static void ReadShape(ReadOnlySpan<byte> data, List<TensorDimension> dims)
    {
        // TensorShapeProto: repeated Dimension dim = field 1
        int pos = 0;
        while (pos < data.Length)
        {
            ulong tag = ReadVarint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);

            if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                var sub = data.Slice(pos, len);

                if (field == 1) dims.Add(ReadDimension(sub));
                pos += len;
            }
            else if (wire == 0) { ReadVarint(data, ref pos); }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }
    }

    private static TensorDimension ReadDimension(ReadOnlySpan<byte> data)
    {
        // Dimension: field 1 = dim_value (int64), field 2 = dim_param (string)
        long   dimValue = -1;
        string dimParam = "";
        int    pos      = 0;

        while (pos < data.Length)
        {
            ulong tag = ReadVarint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);

            if (wire == 0)
            {
                long val = (long)ReadVarint(data, ref pos);
                if (field == 1) dimValue = val;
            }
            else if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                if (field == 2) dimParam = ReadUtf8(data.Slice(pos, len));
                pos += len;
            }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }

        if (!string.IsNullOrEmpty(dimParam))
            return new TensorDimension(null, dimParam);
        if (dimValue >= 0)
            return new TensorDimension((int)dimValue, null);
        // dim_value was never set (no field present) → dynamic
        return new TensorDimension(null, "?");
    }

    private static int ReadOpsetVersion(ReadOnlySpan<byte> data)
    {
        // OperatorSetIdProto: field 1 = domain, field 2 = version
        string domain  = "";
        int    version = 0;
        int    pos     = 0;

        while (pos < data.Length)
        {
            ulong tag = ReadVarint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);

            if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                if (field == 1) domain = ReadUtf8(data.Slice(pos, len));
                pos += len;
            }
            else if (wire == 0)
            {
                long val = (long)ReadVarint(data, ref pos);
                if (field == 2) version = (int)val;
            }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }

        return string.IsNullOrEmpty(domain) ? version : 0;
    }

    private static void ReadMetadataEntry(ReadOnlySpan<byte> data, Dictionary<string, string> meta)
    {
        // StringStringEntryProto: field 1 = key, field 2 = value
        string key = "", val = "";
        int pos = 0;

        while (pos < data.Length)
        {
            ulong tag = ReadVarint(data, ref pos);
            int field = (int)(tag >> 3);
            int wire  = (int)(tag & 7);

            if (wire == 2)
            {
                int len = (int)ReadVarint(data, ref pos);
                if (pos + len > data.Length) break;
                var sub = data.Slice(pos, len);
                if (field == 1) key = ReadUtf8(sub);
                else if (field == 2) val = ReadUtf8(sub);
                pos += len;
            }
            else if (wire == 0) { ReadVarint(data, ref pos); }
            else if (wire == 1) { pos += 8; }
            else if (wire == 5) { pos += 4; }
            else break;
        }

        if (!string.IsNullOrEmpty(key)) meta[key] = val;
    }

    private static string ReadUtf8(ReadOnlySpan<byte> data)
        => System.Text.Encoding.UTF8.GetString(data);

    private static ulong ReadVarint(ReadOnlySpan<byte> data, ref int pos)
    {
        ulong result = 0;
        int   shift  = 0;
        while (pos < data.Length)
        {
            byte b = data[pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }
}

// OnnxProtobufScanner retained for backward compatibility with any callers.
internal static class OnnxProtobufScanner
{
    public static int ReadDefaultOpsetVersion(ReadOnlySpan<byte> data)
        => OnnxProtobufParser.Parse(data).OpsetVersion;
}
