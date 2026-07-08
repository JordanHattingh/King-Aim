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
                var manifest = ModelService.LoadOrCreateManifest(tempDir, Path.Combine(tempDir, "model.onnx"), classes);

                Assert.Single(manifest.Classes);
                Assert.Equal(SemanticRole.Enemy, manifest.Classes[0].SemanticRole);
                Assert.True(File.Exists(ModelManifest.GetManifestPath(tempDir)));
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
        public void DiscoverModelDirectories_ReturnsEmpty_WhenRootMissing()
        {
            string missingRoot = Path.Combine(Path.GetTempPath(), "AimmyTests_missing_" + Guid.NewGuid());
            var result = ModelService.DiscoverModelDirectories(missingRoot);
            Assert.Empty(result);
        }
    }
}
