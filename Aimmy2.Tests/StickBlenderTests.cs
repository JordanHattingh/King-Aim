using Aimmy2.Gamepad;
using Xunit;

namespace Aimmy2.Tests
{
    public class StickBlenderTests
    {
        [Fact]
        public void NoAssist_PassesPlayerInputThrough()
        {
            var (rx, ry) = StickBlender.Blend(0.6f, -0.3f, 0f, 0f);
            Assert.Equal(0.6f, rx, 3);
            Assert.Equal(-0.3f, ry, 3);
        }

        [Fact]
        public void StrongPlayerInput_RemainsAuthoritative()
        {
            var (rx, ry) = StickBlender.Blend(1f, 0f, -0.8f, 0.5f);
            Assert.Equal(1f, rx, 3);
            Assert.Equal(0f, ry, 3);
        }

        [Fact]
        public void WeakPlayerInput_AllowsCooperativeContribution()
        {
            var (rx, _) = StickBlender.Blend(0.1f, 0f, 0.25f, 0f);
            Assert.Equal(0.35f, rx, 3);
        }

        [Fact]
        public void IncreasingPhysicalIntent_ReducesAssistContribution()
        {
            float weak = StickBlender.Blend(0.2f, 0f, 0.4f, 0f).RX - 0.2f;
            float medium = StickBlender.Blend(0.5f, 0f, 0.4f, 0f).RX - 0.5f;
            float strong = StickBlender.Blend(0.9f, 0f, 0.4f, 0f).RX - 0.9f;

            Assert.True(weak > medium);
            Assert.True(medium > strong);
            Assert.Equal(0f, strong, 3);
        }

        [Fact]
        public void Output_NeverExceedsPlusMinusOne()
        {
            var (rx, ry) = StickBlender.Blend(1f, 1f, 1f, 1f);
            Assert.InRange(rx, -1f, 1f);
            Assert.InRange(ry, -1f, 1f);
        }

        [Fact]
        public void NoPlayerInput_AssistContributionIsPreserved()
        {
            var (rx, ry) = StickBlender.Blend(0f, 0f, 0.4f, -0.2f);
            Assert.Equal(0.4f, rx, 3);
            Assert.Equal(-0.2f, ry, 3);
        }
    }
}
