using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Aimmy2.AILogic
{
    public enum SemanticRole
    {
        Enemy,
        Player,
        Friendly,
        Ignore
    }

    public class ModelClassEntry
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
        public SemanticRole SemanticRole { get; set; }
    }

    public class ModelManifest
    {
        public string SchemaVersion { get; set; } = "1";
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Version { get; set; } = "";
        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public List<ModelClassEntry> Classes { get; set; } = new();
        public float DefaultConfidence { get; set; } = 0.5f;
        /// <summary>
        /// Vertical fraction of the bounding box at which to place the aim point.
        /// 0.0 = top of box, 0.5 = center, 1.0 = bottom.
        /// Full-body models: 0.25 (head/chest). Head-only models: 0.5 (dead center).
        /// </summary>
        public float AimPointFraction { get; set; } = 0.25f;

        private static readonly JsonSerializerOptions SerializerOptions = new()
        {
            PropertyNameCaseInsensitive = true,
            Converters = { new JsonStringEnumConverter() },
        };

        public static string ManifestFileName => "manifest.json";

        /// <summary>
        /// Manifest path for a model living alone in its own folder (e.g. Models/EnemyDetectionV1/model.onnx).
        /// </summary>
        public static string GetManifestPath(string modelDirectory) =>
            Path.Combine(modelDirectory, ManifestFileName);

        /// <summary>
        /// Manifest path for a model sharing a flat folder with other models (e.g. bin/models/*.onnx),
        /// where a single manifest.json per folder would collide across different models. Named after
        /// the model file itself: bin/models/Foo.onnx -> bin/models/Foo.manifest.json.
        /// </summary>
        public static string GetManifestPathForModel(string modelPath) =>
            Path.Combine(
                Path.GetDirectoryName(modelPath) ?? ".",
                Path.GetFileNameWithoutExtension(modelPath) + ".manifest.json");

        public static ModelManifest Load(string manifestPath)
        {
            string json = File.ReadAllText(manifestPath);
            var manifest = JsonSerializer.Deserialize<ModelManifest>(json, ManifestOptionsWithSnakeCase());
            return manifest ?? throw new InvalidDataException($"Manifest at '{manifestPath}' could not be parsed.");
        }

        public static bool TryLoad(string manifestPath, out ModelManifest? manifest)
        {
            try
            {
                if (!File.Exists(manifestPath))
                {
                    manifest = null;
                    return false;
                }

                manifest = Load(manifestPath);
                return true;
            }
            catch
            {
                manifest = null;
                return false;
            }
        }

        public void Save(string manifestPath)
        {
            string json = JsonSerializer.Serialize(this, ManifestOptionsWithSnakeCase(true));
            File.WriteAllText(manifestPath, json);
        }

        public static ModelManifest CreateFallback(string modelId, string modelName, IReadOnlyDictionary<int, string> modelClasses, int inputSize = 640)
        {
            var manifest = new ModelManifest
            {
                SchemaVersion = "1",
                Id = modelId,
                Name = modelName,
                Version = "1.0.0",
                InputWidth = inputSize,
                InputHeight = inputSize,
                DefaultConfidence = 0.5f,
            };

            foreach (var (id, name) in modelClasses)
            {
                manifest.Classes.Add(new ModelClassEntry
                {
                    Id = id,
                    Name = name,
                    SemanticRole = SemanticRole.Enemy,
                });
            }

            return manifest;
        }

        public Dictionary<int, SemanticRole> BuildClassRoleMap() =>
            Classes.ToDictionary(c => c.Id, c => c.SemanticRole);

        public Dictionary<int, string> BuildClassNameMap() =>
            Classes.ToDictionary(c => c.Id, c => c.Name);

        private static JsonSerializerOptions ManifestOptionsWithSnakeCase(bool writeIndented = false)
        {
            var options = new JsonSerializerOptions(SerializerOptions)
            {
                WriteIndented = writeIndented,
                PropertyNamingPolicy = SnakeCaseNamingPolicy.Instance,
            };
            return options;
        }

        private sealed class SnakeCaseNamingPolicy : JsonNamingPolicy
        {
            public static readonly SnakeCaseNamingPolicy Instance = new();

            public override string ConvertName(string name)
            {
                if (string.IsNullOrEmpty(name))
                    return name;

                var sb = new System.Text.StringBuilder(name.Length + 4);
                for (int i = 0; i < name.Length; i++)
                {
                    char c = name[i];
                    if (char.IsUpper(c))
                    {
                        if (i > 0)
                            sb.Append('_');
                        sb.Append(char.ToLowerInvariant(c));
                    }
                    else
                    {
                        sb.Append(c);
                    }
                }
                return sb.ToString();
            }
        }
    }
}
