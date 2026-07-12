using System.Text.Json;
using System.Text.Json.Nodes;

namespace KingAim.Core.Models.Onnx;

/// <summary>
/// Serialises an <see cref="OnnxModelContract"/> to a deterministic, human-readable
/// JSON string. Output is byte-for-byte stable for identical inputs:
/// sorted keys, no timestamp, invariant number formatting.
/// </summary>
public static class OnnxContractJsonWriter
{
    private static readonly JsonSerializerOptions _opts = new()
    {
        WriteIndented = true,
    };

    public static string Write(OnnxModelContract contract)
    {
        ArgumentNullException.ThrowIfNull(contract);

        var obj = new JsonObject
        {
            ["modelPath"]     = contract.ModelPath,
            ["sha256"]        = contract.Sha256,
            ["fileSizeBytes"] = contract.FileSizeBytes,
            ["opsetVersion"]  = contract.OpsetVersion,
            ["inputs"]        = TensorsNode(contract.Inputs),
            ["outputs"]       = TensorsNode(contract.Outputs),
        };

        if (contract.Metadata.Count > 0)
        {
            var meta = new JsonObject();
            foreach (var kv in contract.Metadata.OrderBy(x => x.Key, StringComparer.Ordinal))
                meta[kv.Key] = kv.Value;
            obj["metadata"] = meta;
        }

        return obj.ToJsonString(_opts);
    }

    public static void WriteToFile(OnnxModelContract contract, string path)
        => File.WriteAllText(path, Write(contract));

    public static string WriteCompatibility(DecoderCompatibilityReport report)
    {
        ArgumentNullException.ThrowIfNull(report);

        var obj = new JsonObject
        {
            ["decoderId"]    = report.DecoderId,
            ["isCompatible"] = report.IsCompatible,
            ["errors"]       = StringArray(report.Errors),
            ["warnings"]     = StringArray(report.Warnings),
            ["observations"] = StringArray(report.Observations),
        };
        return obj.ToJsonString(_opts);
    }

    // -------------------------------------------------------------------------

    private static JsonArray TensorsNode(IReadOnlyList<TensorContract> tensors)
    {
        var arr = new JsonArray();
        foreach (var t in tensors)
        {
            arr.Add(new JsonObject
            {
                ["name"]        = t.Name,
                ["elementType"] = t.ElementType,
                ["dimensions"]  = DimensionsNode(t.Dimensions),
            });
        }
        return arr;
    }

    private static JsonArray DimensionsNode(IReadOnlyList<TensorDimension> dims)
    {
        var arr = new JsonArray();
        foreach (var d in dims)
        {
            if (d.IsDynamic)
                arr.Add(d.SymbolicName ?? "?");   // dynamic → string
            else
                arr.Add(d.FixedValue!.Value);      // fixed   → integer
        }
        return arr;
    }

    private static JsonArray StringArray(IReadOnlyList<string> items)
    {
        var arr = new JsonArray();
        foreach (var s in items) arr.Add(s);
        return arr;
    }
}
