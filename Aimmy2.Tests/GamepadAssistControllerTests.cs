using Aimmy2.Gamepad;
using Xunit;

namespace Aimmy2.Tests
{
    public class GamepadAssistControllerTests
    {
        private static GamepadAssistController MakeController() => new()
        {
            Gain = 1.5f,
            DeadbandRadius = 0.02f,
            MaxOutput = 1.0f,
            MaxSlewRate = 100f, // large so single-step tests aren't slew-limited
            MaxObservationAge = 10,
        };

        [Fact]
        public void PositiveErrorX_GeneratesPositiveRX()
        {
            var controller = MakeController();
            var (rx, _) = controller.Update(true, 0.5f, 0f, 0f, 0f, 0.9f, 0, 0.016);
            Assert.True(rx > 0);
        }

        [Fact]
        public void NegativeErrorX_GeneratesNegativeRX()
        {
            var controller = MakeController();
            var (rx, _) = controller.Update(true, -0.5f, 0f, 0f, 0f, 0.9f, 0, 0.016);
            Assert.True(rx < 0);
        }

        [Fact]
        public void PositiveErrorY_GeneratesPositiveRY()
        {
            // Convention: positive ErrorY means the target is below screen center
            // (target.Y > screenCenter.Y), so RY should be positive to push the
            // stick "down" and follow the target, matching TargetSelector's error sign.
            var controller = MakeController();
            var (_, ry) = controller.Update(true, 0f, 0.5f, 0f, 0f, 0.9f, 0, 0.016);
            Assert.True(ry > 0);
        }

        [Fact]
        public void NegativeErrorY_GeneratesNegativeRY()
        {
            var controller = MakeController();
            var (_, ry) = controller.Update(true, 0f, -0.5f, 0f, 0f, 0.9f, 0, 0.016);
            Assert.True(ry < 0);
        }

        [Fact]
        public void NoSelectedTarget_ReturnsRXRYToZero()
        {
            var controller = MakeController();
            controller.Update(true, 0.8f, 0.8f, 0f, 0f, 0.9f, 0, 0.016);

            (float rx, float ry) result = (0, 0);
            for (int i = 0; i < 200; i++)
            {
                result = controller.Update(false, 0f, 0f, 0f, 0f, 0f, 0, 0.016);
            }

            Assert.Equal(0f, result.rx, 3);
            Assert.Equal(0f, result.ry, 3);
        }

        [Fact]
        public void StaleObservation_IsRejected()
        {
            var controller = MakeController();
            var (rx, ry) = controller.Update(true, 0.5f, 0.5f, 0f, 0f, 0.9f, observationAgeSamples: 999, dtSeconds: 0.016);

            Assert.Equal(0f, rx, 3);
            Assert.Equal(0f, ry, 3);
        }

        [Fact]
        public void RXRYNeverExceedPlusMinusOne()
        {
            var controller = MakeController();
            controller.Gain = 50f; // deliberately extreme

            for (int i = 0; i < 50; i++)
            {
                var (rx, ry) = controller.Update(true, 1f, -1f, 0f, 0f, 0.9f, 0, 0.016);
                Assert.InRange(rx, -1f, 1f);
                Assert.InRange(ry, -1f, 1f);
            }
        }

        [Fact]
        public void RadialDeadband_SuppressesTinyError()
        {
            var controller = MakeController();
            var (rx, ry) = controller.Update(true, 0.005f, 0.005f, 0f, 0f, 0.9f, 0, 0.016);
            Assert.Equal(0f, rx, 3);
            Assert.Equal(0f, ry, 3);
        }

        [Fact]
        public void SimulatedUpdates_100Hz_And_250Hz_ComparableBehaviour()
        {
            // Same target error sustained for the same wall-clock duration (0.5s) at two
            // different sample rates should converge to a comparable steady-state output,
            // since the controller uses time-based slew limiting rather than per-call limiting.
            var controller100 = MakeController();
            controller100.MaxSlewRate = 4.0f;
            var controller250 = MakeController();
            controller250.MaxSlewRate = 4.0f;

            (float rx, float ry) last100 = (0, 0);
            for (int i = 0; i < 50; i++) // 50 steps * 10ms = 0.5s
            {
                last100 = controller100.Update(true, 0.5f, 0f, 0f, 0f, 0.9f, 0, 0.010);
            }

            (float rx, float ry) last250 = (0, 0);
            for (int i = 0; i < 125; i++) // 125 steps * 4ms = 0.5s
            {
                last250 = controller250.Update(true, 0.5f, 0f, 0f, 0f, 0.9f, 0, 0.004);
            }

            Assert.Equal(last100.rx, last250.rx, 1);
        }

        [Fact]
        public void VisionFailure_DoesNotCrashGamepadManagement()
        {
            var controller = MakeController();

            // Simulate a burst of degenerate/failed vision frames (NaN-free but zeroed/invalid).
            var exception = Record.Exception(() =>
            {
                controller.Update(false, 0f, 0f, 0f, 0f, 0f, int.MaxValue, 0.0);
                controller.Update(true, 0f, 0f, 0f, 0f, 0f, -1, -1.0);
                controller.Update(false, float.NaN, float.NaN, 0f, 0f, 0f, 0, 0.016);
            });

            Assert.Null(exception);
        }
    }
}
