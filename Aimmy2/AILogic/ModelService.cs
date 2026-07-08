using Microsoft.ML.OnnxRuntime;
using Other;
using System.IO;
using LogLevel = Other.LogManager.LogLevel;

namespace Aimmy2.AILogic
{
    public sealed class LoadedModel : IDisposable
    {
        public required InferenceSession Session { get; init; }
        public required List<string> OutputNames { get; init; }
        public required ModelManifest Manifest { get; init; }
        public required string ModelPath { get; init; }

        public void Dispose() => Session.Dispose();
    }

    public sealed class ModelService : IDisposable
    {
        private readonly object _lock = new();
        private LoadedModel? _active;
        private bool _disposed;

        public LoadedModel? Active
        {
            get { lock (_lock) { return _active; } }
        }

        public static IReadOnlyList<string> DiscoverModelDirectories(string modelsRoot)
        {
            if (!Directory.Exists(modelsRoot))
                return Array.Empty<string>();

            return Directory.GetDirectories(modelsRoot)
                .Where(dir => File.Exists(Path.Combine(dir, "model.onnx")))
                .ToList();
        }

        public static ModelManifest LoadOrCreateManifest(string modelDirectory, string onnxPath, IReadOnlyDictionary<int, string>? fallbackClasses = null)
        {
            string manifestPath = ModelManifest.GetManifestPath(modelDirectory);

            if (ModelManifest.TryLoad(manifestPath, out var manifest) && manifest != null)
            {
                return manifest;
            }

            string modelId = Path.GetFileName(modelDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            var classes = fallbackClasses ?? new Dictionary<int, string> { { 0, "enemy" } };
            var fallback = ModelManifest.CreateFallback(modelId, modelId, classes);

            try
            {
                fallback.Save(manifestPath);
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Warning, $"Could not persist auto-generated manifest for '{modelId}': {ex.Message}");
            }

            return fallback;
        }

        public bool TryHotSwap(string modelPath, bool useDirectML, out string? error)
        {
            lock (_lock)
            {
                if (_disposed)
                {
                    error = "ModelService has been disposed.";
                    return false;
                }

                LoadedModel? candidate = null;
                try
                {
                    candidate = LoadCandidate(modelPath, useDirectML);

                    var previous = _active;
                    _active = candidate;
                    previous?.Dispose();

                    error = null;
                    return true;
                }
                catch (Exception ex)
                {
                    candidate?.Dispose();
                    error = ex.Message;
                    LogManager.Log(LogLevel.Error, $"Model hot-swap failed, keeping previous model active: {ex.Message}", true, 5000);
                    return false;
                }
            }
        }

        private static LoadedModel LoadCandidate(string modelPath, bool useDirectML)
        {
            string modelDirectory = Path.GetDirectoryName(modelPath) ?? ".";
            OnnxModelLoadResult loaded = OnnxModelSessionFactory.Load(modelPath, useDirectML);

            ModelManifest manifest;
            try
            {
                manifest = LoadOrCreateManifest(modelDirectory, modelPath);
            }
            catch
            {
                loaded.Session.Dispose();
                throw;
            }

            return new LoadedModel
            {
                Session = loaded.Session,
                OutputNames = loaded.OutputNames,
                Manifest = manifest,
                ModelPath = modelPath,
            };
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed)
                    return;

                _disposed = true;
                _active?.Dispose();
                _active = null;
            }
        }
    }
}
