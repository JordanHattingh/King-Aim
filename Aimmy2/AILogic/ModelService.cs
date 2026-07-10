using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json.Linq;
using Other;
using System.IO;
using LogLevel = Other.LogManager.LogLevel;

namespace Aimmy2.AILogic
{
    /// <summary>
    /// Owns one ONNX session and keeps it alive until every acquired lease has been released.
    /// A retired model is no longer leasable, but an inference that already acquired it may finish.
    /// </summary>
    public sealed class LoadedModel : IDisposable
    {
        private readonly object _lifetimeLock = new();
        private int _leaseCount;
        private bool _retired;
        private bool _disposed;

        public required InferenceSession Session { get; init; }
        public required List<string> OutputNames { get; init; }
        public required ModelManifest Manifest { get; init; }
        public required string ModelPath { get; init; }

        internal ModelSessionLease? TryAcquireLease()
        {
            lock (_lifetimeLock)
            {
                if (_retired || _disposed)
                    return null;

                _leaseCount++;
                return new ModelSessionLease(this);
            }
        }

        internal void Retire()
        {
            bool disposeNow;
            lock (_lifetimeLock)
            {
                if (_retired)
                    return;

                _retired = true;
                disposeNow = _leaseCount == 0;
            }

            if (disposeNow)
                DisposeSessionOnce();
        }

        internal void ReleaseLease()
        {
            bool disposeNow;
            lock (_lifetimeLock)
            {
                if (_leaseCount <= 0)
                    throw new InvalidOperationException("Model session lease count underflow.");

                _leaseCount--;
                disposeNow = _retired && _leaseCount == 0;
            }

            if (disposeNow)
                DisposeSessionOnce();
        }

        private void DisposeSessionOnce()
        {
            lock (_lifetimeLock)
            {
                if (_disposed)
                    return;
                _disposed = true;
            }

            Session.Dispose();
        }

        public void Dispose() => Retire();
    }

    /// <summary>
    /// Short-lived ownership token for an active model. Disposing the lease is mandatory.
    /// </summary>
    public sealed class ModelSessionLease : IDisposable
    {
        private LoadedModel? _owner;

        internal ModelSessionLease(LoadedModel owner)
        {
            _owner = owner;
        }

        public InferenceSession Session =>
            _owner?.Session ?? throw new ObjectDisposedException(nameof(ModelSessionLease));

        public IReadOnlyList<string> OutputNames =>
            _owner?.OutputNames ?? throw new ObjectDisposedException(nameof(ModelSessionLease));

        public ModelManifest Manifest =>
            _owner?.Manifest ?? throw new ObjectDisposedException(nameof(ModelSessionLease));

        public string ModelPath =>
            _owner?.ModelPath ?? throw new ObjectDisposedException(nameof(ModelSessionLease));

        public void Dispose()
        {
            LoadedModel? owner = Interlocked.Exchange(ref _owner, null);
            owner?.ReleaseLease();
        }
    }

    public sealed class ModelService : IDisposable
    {
        private readonly object _lock = new();
        private LoadedModel? _active;
        private bool _disposed;

        /// <summary>
        /// Metadata/debug view only. Production inference must use AcquireActive().
        /// </summary>
        public LoadedModel? Active
        {
            get { lock (_lock) { return _active; } }
        }

        public bool HasActiveModel
        {
            get { lock (_lock) { return !_disposed && _active != null; } }
        }

        public ModelSessionLease? AcquireActive()
        {
            lock (_lock)
            {
                if (_disposed)
                    return null;
                return _active?.TryAcquireLease();
            }
        }

        public void ClearActive()
        {
            LoadedModel? active;
            lock (_lock)
            {
                if (_disposed)
                    return;
                active = _active;
                _active = null;
            }

            active?.Retire();
        }

        public static IReadOnlyList<string> DiscoverModelDirectories(string modelsRoot)
        {
            if (!Directory.Exists(modelsRoot))
                return Array.Empty<string>();

            return Directory.GetDirectories(modelsRoot)
                .Where(dir => File.Exists(Path.Combine(dir, "model.onnx")))
                .ToList();
        }

        /// <summary>
        /// Resolves a model's manifest. Checks the per-model path first, then the per-directory
        /// package manifest, and finally generates a compatibility manifest from model class metadata.
        /// </summary>
        public static ModelManifest LoadOrCreateManifest(
            string modelDirectory,
            string onnxPath,
            IReadOnlyDictionary<int, string>? fallbackClasses = null)
        {
            string perModelManifestPath = ModelManifest.GetManifestPathForModel(onnxPath);
            if (ModelManifest.TryLoad(perModelManifestPath, out var perModelManifest) && perModelManifest != null)
                return perModelManifest;

            string perDirectoryManifestPath = ModelManifest.GetManifestPath(modelDirectory);
            if (ModelManifest.TryLoad(perDirectoryManifestPath, out var perDirectoryManifest) && perDirectoryManifest != null)
                return perDirectoryManifest;

            string modelId = Path.GetFileNameWithoutExtension(onnxPath);
            var classes = fallbackClasses ?? new Dictionary<int, string> { { 0, "unknown" } };
            var fallback = ModelManifest.CreateFallback(modelId, modelId, classes);

            try
            {
                fallback.Save(perModelManifestPath);
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Warning, $"Could not persist auto-generated manifest for '{modelId}': {ex.Message}");
            }

            return fallback;
        }

        public bool TryHotSwap(string modelPath, bool useDirectML, out string? error)
        {
            LoadedModel? candidate = null;
            LoadedModel? previous = null;

            try
            {
                candidate = LoadCandidate(modelPath, useDirectML);

                lock (_lock)
                {
                    if (_disposed)
                    {
                        error = "ModelService has been disposed.";
                        candidate.Retire();
                        return false;
                    }

                    previous = _active;
                    _active = candidate;
                    candidate = null; // service owns it now
                }

                // Retire outside the service lock. Existing leases remain valid until released.
                previous?.Retire();
                error = null;
                return true;
            }
            catch (Exception ex)
            {
                candidate?.Retire();
                error = ex.Message;
                LogManager.Log(
                    LogLevel.Error,
                    $"Model hot-swap failed, keeping previous model active: {ex.Message}",
                    true,
                    5000);
                return false;
            }
        }

        private static LoadedModel LoadCandidate(string modelPath, bool useDirectML)
        {
            string modelDirectory = Path.GetDirectoryName(modelPath) ?? ".";
            OnnxModelLoadResult loaded = OnnxModelSessionFactory.Load(modelPath, useDirectML);

            try
            {
                IReadOnlyDictionary<int, string> fallbackClasses = ReadModelClasses(loaded.Session);
                ModelManifest manifest = LoadOrCreateManifest(modelDirectory, modelPath, fallbackClasses);

                return new LoadedModel
                {
                    Session = loaded.Session,
                    OutputNames = loaded.OutputNames,
                    Manifest = manifest,
                    ModelPath = modelPath,
                };
            }
            catch
            {
                loaded.Session.Dispose();
                throw;
            }
        }

        private static IReadOnlyDictionary<int, string> ReadModelClasses(InferenceSession session)
        {
            try
            {
                if (session.ModelMetadata.CustomMetadataMap.TryGetValue("names", out string? value)
                    && !string.IsNullOrWhiteSpace(value))
                {
                    JObject data = JObject.Parse(value);
                    var classes = new Dictionary<int, string>();
                    foreach (var item in data)
                    {
                        if (int.TryParse(item.Key, out int classId)
                            && item.Value?.Type == JTokenType.String)
                        {
                            classes[classId] = item.Value.ToString();
                        }
                    }

                    if (classes.Count > 0)
                        return classes;
                }
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Warning, $"Could not read class metadata while preparing model manifest: {ex.Message}");
            }

            return new Dictionary<int, string> { { 0, "unknown" } };
        }

        public void Dispose()
        {
            LoadedModel? active;
            lock (_lock)
            {
                if (_disposed)
                    return;

                _disposed = true;
                active = _active;
                _active = null;
            }

            active?.Retire();
        }
    }
}
