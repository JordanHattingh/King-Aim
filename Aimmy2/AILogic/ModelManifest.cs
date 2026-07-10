using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Aimmy2.AILogic
{
    public enum SemanticRole
    {
        Unknown,
        Enemy,
        Player,
        Friendly,
        Npc,
        Objective,
        Interactable,
        Ignore
    }

    public class ModelClassEntry
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
        public SemanticRole SemanticRole { get; set; } = SemanticRole.Unknown;
    }

    /// <summary>
    /// Per-feature normalization constants exported alongside the GRU training run.
    /// The GRU was trained on standardized inputs; these constants must match exactly.
    /// </summary>
    public sealed class GruNormConstants
    {
        // Deliberately fail-closed when a field is omitted from the manifest. These values are
        // dataset statistics and must come from training/norm_constants.json; hard-coded guesses
        // create silent train/serve skew.
        public float LogWMean { get; set; } = float.NaN;
        public float LogWStd { get; set; } = float.NaN;
        public float LogHMean { get; set; } = float.NaN;
        public float LogHStd { get; set; } = float.NaN;
        public float DtMean { get; set; } = float.NaN;
        public float DtStd { get; set; } = float.NaN;
        public float AgeMean { get; set; } = float.NaN;
        public float AgeStd { get; set; } = float.NaN;

        public void Validate()
        {
            float[] values =
            [
                LogWMean, LogWStd, LogHMean, LogHStd,
                DtMean, DtStd, AgeMean, AgeStd,
            ];

            if (values.Any(value => !float.IsFinite(value)))
                throw new InvalidDataException(
                    "GRU normalization constants must all be explicitly supplied by the training pipeline.");

            if (LogWStd <= 0 || LogHStd <= 0 || DtStd <= 0 || AgeStd <= 0)
                throw new InvalidDataException("GRU normalization standard deviations must be greater than zero.");
        }
    }

    public class ModelManifest
    {
        public string SchemaVersion { get; set; } = "2";
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Version { get; set; } = "";
        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public List<ModelClassEntry> Classes { get; set; } = new();
        public float DefaultConfidence { get; set; } = 0.5f;

        /// <summary>
        /// Decoder/output contract. Known values: yolo-detect-v1 and yolo-pose-kpt-v1.
        /// </summary>
        public string OutputSchema { get; set; } = "yolo-detect-v1";

        /// <summary>
        /// Vertical box fraction used only as a fallback when a semantic pose point is unavailable.
        /// 0.0 = top, 0.5 = centre, 1.0 = bottom.
        /// </summary>
        public float AimPointFraction { get; set; } = 0.25f;

        // ── Pose model fields ─────────────────────────────────────────────────

        public bool IsPoseModel { get; set; } = false;
        public int KeypointCount { get; set; } = 0;
        public List<string> KeypointNames { get; set; } = new();

        /// <summary>
        /// True only when the ONNX output stores keypoint visibility as raw logits.
        /// Ultralytics pose exports commonly expose activated values, so the safe default is false.
        /// This flag makes the train/export/runtime contract explicit and prevents double-sigmoid bugs.
        /// </summary>
        public bool KeypointVisibilityIsLogit { get; set; } = false;

        // ── Neural bundle contracts ───────────────────────────────────────────

        public string? TemporalModelPath { get; set; }
        public string? CalibrationModelPath { get; set; }
        public string? MovementModelPath { get; set; }
        public GruNormConstants? GruNorm { get; set; }

        public string TemporalFeatureSchema { get; set; } = NeuralFeatureSchemas.TemporalV2;
        public string CalibrationFeatureSchema { get; set; } = NeuralFeatureSchemas.CalibrationV2;
        public string MovementFeatureSchema { get; set; } = NeuralFeatureSchemas.MovementV1;

        private static readonly JsonSerializerOptions SerializerOptions = new()
        {
            PropertyNameCaseInsensitive = true,
            Converters = { new JsonStringEnumConverter() },
        };

        public static string ManifestFileName => "manifest.json";

        public static string GetManifestPath(string modelDirectory) =>
            Path.Combine(modelDirectory, ManifestFileName);

        public static string GetManifestPathForModel(string modelPath) =>
            Path.Combine(
                Path.GetDirectoryName(modelPath) ?? ".",
                Path.GetFileNameWithoutExtension(modelPath) + ".manifest.json");

        public static ModelManifest Load(string manifestPath)
        {
            string json = File.ReadAllText(manifestPath);
            var manifest = JsonSerializer.Deserialize<ModelManifest>(json, ManifestOptionsWithSnakeCase());
            if (manifest == null)
                throw new InvalidDataException($"Manifest at '{manifestPath}' could not be parsed.");

            manifest.Validate();
            return manifest;
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
            Validate();
            string json = JsonSerializer.Serialize(this, ManifestOptionsWithSnakeCase(true));
            string directory = Path.GetDirectoryName(manifestPath) ?? ".";
            Directory.CreateDirectory(directory);

            string tempPath = manifestPath + ".tmp";
            string backupPath = manifestPath + ".bak";
            File.WriteAllText(tempPath, json);

            if (File.Exists(manifestPath))
                File.Copy(manifestPath, backupPath, overwrite: true);

            File.Move(tempPath, manifestPath, overwrite: true);
        }

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(Id))
                throw new InvalidDataException("Model manifest id is required.");
            if (InputWidth < 0 || InputHeight < 0)
                throw new InvalidDataException("Model input dimensions cannot be negative.");
            if (Classes.GroupBy(c => c.Id).Any(g => g.Count() > 1))
                throw new InvalidDataException("Model manifest contains duplicate class ids.");

            if (IsPoseModel)
            {
                if (KeypointCount <= 0)
                    throw new InvalidDataException("Pose manifests require keypoint_count > 0.");
                if (KeypointNames.Count > 0 && KeypointNames.Count != KeypointCount)
                    throw new InvalidDataException("keypoint_names count must match keypoint_count.");
            }

            if (!string.IsNullOrWhiteSpace(TemporalModelPath))
            {
                if (GruNorm == null)
                    throw new InvalidDataException("A temporal model requires gru_norm constants.");
                GruNorm.Validate();
                if (!string.Equals(TemporalFeatureSchema, NeuralFeatureSchemas.TemporalV2, StringComparison.Ordinal))
                    throw new InvalidDataException($"Unsupported temporal feature schema '{TemporalFeatureSchema}'.");
            }

            if (!string.IsNullOrWhiteSpace(CalibrationModelPath) &&
                !string.Equals(CalibrationFeatureSchema, NeuralFeatureSchemas.CalibrationV2, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Unsupported calibration feature schema '{CalibrationFeatureSchema}'.");
            }

            if (!string.IsNullOrWhiteSpace(MovementModelPath) &&
                !string.Equals(MovementFeatureSchema, NeuralFeatureSchemas.MovementV1, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Unsupported movement feature schema '{MovementFeatureSchema}'.");
            }
        }

        public static ModelManifest CreateFallback(
            string modelId,
            string modelName,
            IReadOnlyDictionary<int, string> modelClasses,
            int inputSize = 640)
        {
            var manifest = new ModelManifest
            {
                SchemaVersion = "2",
                Id = modelId,
                Name = modelName,
                Version = "1.0.0",
                InputWidth = inputSize,
                InputHeight = inputSize,
                DefaultConfidence = 0.5f,
                OutputSchema = "yolo-detect-v1",
            };

            foreach (var (id, name) in modelClasses)
            {
                manifest.Classes.Add(new ModelClassEntry
                {
                    Id = id,
                    Name = name,
                    SemanticRole = SemanticRole.Unknown,
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
