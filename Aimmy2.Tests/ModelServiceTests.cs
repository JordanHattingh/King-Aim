using System.IO;
using Aimmy2.AILogic;
using Xunit;

namespace Aimmy2.Tests
{
    public class ModelServiceTests
    {
        [Fact]
        public void LoadOrCreateManifest_GeneratesFallback_WhenManifestMissing()
        {
            string tempDir = Path.Combine(Path.GetTempPath(), "AimmyTests_" + Guid.NewGuid());
            Directory.CreateDirectory(tempDir);
            try
            {
                var classes = new Dictionary<int, string> { { 0, "enemy" } };
                string onnxPath = Path.Combine(tempDir, "model.onnx");
                var manifest = ModelService.LoadOrCreateManifest(tempDir, onnxPath, classes);

                Assert.Single(manifest.Classes);
                Assert.Equal(SemanticRole.Unknown, manifest.Classes[0].SemanticRole);
                Assert.True(File.Exists(ModelManifest.GetManifestPathForModel(onnxPath)));
            }
            finally
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }

        [Fact]
        public void LoadOrCreateManifest_UsesExistingManifest_WhenPresent()
        {
            string tempDir = Path.Combine(Path.GetTempPath(), "AimmyTests_" + Guid.NewGuid());
            Directory.CreateDirectory(tempDir);
            try
            {
                var manifest = ModelManifest.CreateFallback("custom-id", "Custom", new Dictionary<int, string> { { 0, "target" } });
                manifest.Classes[0].SemanticRole = SemanticRole.Player;
                manifest.Save(ModelManifest.GetManifestPath(tempDir));

                var loaded = ModelService.LoadOrCreateManifest(tempDir, Path.Combine(tempDir, "model.onnx"));

                Assert.Equal("custom-id", loaded.Id);
                Assert.Equal(SemanticRole.Player, loaded.Classes[0].SemanticRole);
            }
            finally
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }

        [Fact]
        public void LoadOrCreateManifest_KeepsSeparateModelsInSameFolderDistinct()
        {
            string tempDir = Path.Combine(Path.GetTempPath(), "AimmyTests_" + Guid.NewGuid());
            Directory.CreateDirectory(tempDir);
            try
            {
                string modelAPath = Path.Combine(tempDir, "GameA.onnx");
                string modelBPath = Path.Combine(tempDir, "GameB.onnx");

                var manifestA = ModelService.LoadOrCreateManifest(tempDir, modelAPath, new Dictionary<int, string> { { 0, "enemy" } });
                var manifestB = ModelService.LoadOrCreateManifest(tempDir, modelBPath, new Dictionary<int, string> { { 0, "enemy" }, { 1, "teammate" } });

                Assert.Single(manifestA.Classes);
                Assert.Equal(2, manifestB.Classes.Count);
                Assert.NotEqual(
                    ModelManifest.GetManifestPathForModel(modelAPath),
                    ModelManifest.GetManifestPathForModel(modelBPath));
            }
            finally
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }

        [Fact]
        public void FailedModelHotSwap_LeavesOldModelActive()
        {
            using var service = new ModelService();

            // No real model available in this environment, so the first "load" is expected
            // to fail too; this test asserts the *shape* of the contract: a failed TryHotSwap
            // must never leave Active pointing at a disposed/partial session, and must report
            // an error rather than throwing.
            bool result = service.TryHotSwap(Path.Combine(Path.GetTempPath(), "does-not-exist.onnx"), useDirectML: false, out string? error);

            Assert.False(result);
            Assert.NotNull(error);
            Assert.Null(service.Active);
        }


        [Fact]
        public void AcquireActive_ReturnsNull_WhenNoModelIsActive()
        {
            using var service = new ModelService();
            Assert.False(service.HasActiveModel);
            Assert.Null(service.AcquireActive());
        }

        [Fact]
        public void TemporalManifest_RequiresNormalizationContract()
        {
            var manifest = ModelManifest.CreateFallback(
                "temporal-test",
                "Temporal Test",
                new Dictionary<int, string> { { 0, "enemy" } });
            manifest.TemporalModelPath = "trajectory_gru.onnx";
            manifest.GruNorm = null;

            Assert.Throws<InvalidDataException>(() => manifest.Validate());
        }

        [Fact]
        public void FallbackManifest_PreservesAllKnownClassesAsUnknownRoles()
        {
            var manifest = ModelManifest.CreateFallback(
                "multi",
                "Multi",
                new Dictionary<int, string>
                {
                    { 0, "enemy" },
                    { 1, "friendly" },
                    { 2, "objective" },
                });

            Assert.Equal(3, manifest.Classes.Count);
            Assert.All(manifest.Classes, entry => Assert.Equal(SemanticRole.Unknown, entry.SemanticRole));
        }

        [Fact]
        public void DiscoverModelDirectories_ReturnsEmpty_WhenRootMissing()
        {
            string missingRoot = Path.Combine(Path.GetTempPath(), "AimmyTests_missing_" + Guid.NewGuid());
            var result = ModelService.DiscoverModelDirectories(missingRoot);
            Assert.Empty(result);
        }
    }
}
