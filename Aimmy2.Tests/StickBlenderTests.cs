using Aimmy2.Gamepad;
using Xunit;

namespace Aimmy2.Tests
{
    public class StickBlenderTests
    {
        [Fact]
        public void NoAssist_PassesPlayerInputThrough()
        {
            var (rx, ry) = StickBlender.Blend(playerX: 0.6f, playerY: -0.3f, assistX: 0f, assistY: 0f);

            Assert.Equal(0.6f, rx, 3);
            Assert.Equal(-0.3f, ry, 3);
        }

        [Fact]
        public void StrongAssist_SuppressesPlayerInput()
        {
            // Assist magnitude at or above DominanceThreshold should fully take over.
            var (rx, ry) = StickBlender.Blend(playerX: 1f, playerY: 1f, assistX: 0.8f, assistY: 0f);

            Assert.Equal(0.8f, rx, 2);
            Assert.Equal(0f, ry, 2);
        }

        [Fact]
        public void WeakAssist_PartiallyBlendsWithPlayerInput()
        {
            var (rx, _) = StickBlender.Blend(playerX: 1f, playerY: 0f, assistX: 0.25f, assistY: 0f);

            // Player weight should be partially but not fully suppressed.
            Assert.True(rx > 0.25f);
            Assert.True(rx < 1.25f);
        }

        [Fact]
        public void Output_NeverExceedsPlusMinusOne()
        {
            var (rx, ry) = StickBlender.Blend(playerX: 1f, playerY: 1f, assistX: 1f, assistY: 1f);

            Assert.InRange(rx, -1f, 1f);
            Assert.InRange(ry, -1f, 1f);
        }

        [Fact]
        public void NoPlayerInput_AssistStillApplies()
        {
            var (rx, ry) = StickBlender.Blend(playerX: 0f, playerY: 0f, assistX: 0.4f, assistY: -0.2f);

            Assert.Equal(0.4f, rx, 3);
            Assert.Equal(-0.2f, ry, 3);
        }
    }
}
