using KingAim.Core.Accessibility.Input;
using Xunit;

namespace Aimmy2.Tests;

public sealed class DriftCompensatorTests
{
    private static ManualDriftCompensator Make() => new();

    private static DriftCompensationProfile Profile(
        float h = 0f, float v = 0f, bool enabled = true)
        => new() { HorizontalOffsetPerSecond = h, VerticalOffsetPerSecond = v, Enabled = enabled };

    // ── Disabled / zero-output cases ─────────────────────────────────────────

    [Fact]
    public void DriftCompensator_ReturnsZeroWhenDisabled()
    {
        var c = Make();
        c.UpdateProfile(new DriftCompensationProfile { Enabled = false,
            HorizontalOffsetPerSecond = 1f, VerticalOffsetPerSecond = 1f });

        var (dx, dy) = c.Compensate(1.0 / 60.0);
        Assert.Equal(0f, dx);
        Assert.Equal(0f, dy);
    }

    [Fact]
    public void DriftCompensator_DefaultProfileIsDisabled()
    {
        var c = Make();
        var (dx, dy) = c.Compensate(1.0 / 60.0);
        Assert.Equal(0f, dx);
        Assert.Equal(0f, dy);
    }

    [Fact]
    public void DriftCompensator_ReturnsZeroForNonPositiveDelta()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: 0.5f, v: 0.5f));

        Assert.Equal((0f, 0f), c.Compensate(0.0));
        Assert.Equal((0f, 0f), c.Compensate(-1.0));
    }

    [Fact]
    public void DriftCompensator_ZeroOffsetProducesZeroCorrection()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: 0f, v: 0f));
        var (dx, dy) = c.Compensate(1.0 / 60.0);
        Assert.Equal(0f, dx);
        Assert.Equal(0f, dy);
    }

    // ── Horizontal axis ───────────────────────────────────────────────────────

    [Fact]
    public void DriftCompensator_AppliesPositiveHorizontalOffset()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: 0.06f));  // 6% screen/s to the right
        var (dx, dy) = c.Compensate(1.0);
        Assert.Equal(0.06f, dx, precision: 5);
        Assert.Equal(0f,    dy);
    }

    [Fact]
    public void DriftCompensator_AppliesNegativeHorizontalOffset()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: -0.03f));  // left drift
        var (dx, _) = c.Compensate(1.0);
        Assert.Equal(-0.03f, dx, precision: 5);
    }

    // ── Vertical axis ─────────────────────────────────────────────────────────

    [Fact]
    public void DriftCompensator_AppliesPositiveVerticalOffset()
    {
        var c = Make();
        c.UpdateProfile(Profile(v: 0.04f));  // downward drift correction
        var (dx, dy) = c.Compensate(1.0);
        Assert.Equal(0f,    dx);
        Assert.Equal(0.04f, dy, precision: 5);
    }

    [Fact]
    public void DriftCompensator_AppliesNegativeVerticalOffset()
    {
        var c = Make();
        c.UpdateProfile(Profile(v: -0.05f));  // upward correction
        var (_, dy) = c.Compensate(1.0);
        Assert.Equal(-0.05f, dy, precision: 5);
    }

    // ── Both axes ─────────────────────────────────────────────────────────────

    [Fact]
    public void DriftCompensator_AppliesBothAxesIndependently()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: 0.02f, v: -0.03f));
        var (dx, dy) = c.Compensate(1.0);
        Assert.Equal( 0.02f, dx, precision: 5);
        Assert.Equal(-0.03f, dy, precision: 5);
    }

    // ── Delta-time scaling ────────────────────────────────────────────────────

    [Fact]
    public void DriftCompensator_ScalesByDeltaTime()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: 0.12f, v: 0.06f));  // per-second offsets

        // At 60 fps, each frame gets 1/60th of the per-second correction
        double dt = 1.0 / 60.0;
        var (dx, dy) = c.Compensate(dt);

        Assert.Equal(0.12f * (float)dt, dx, precision: 5);
        Assert.Equal(0.06f * (float)dt, dy, precision: 5);
    }

    [Fact]
    public void DriftCompensator_SumOverOneSecondEqualsOffsetPerSecond()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: 0.10f, v: -0.08f));

        double dt = 1.0 / 1000.0;  // 1000 Hz polling
        float sumX = 0f, sumY = 0f;
        for (int i = 0; i < 1000; i++)
        {
            var (dx, dy) = c.Compensate(dt);
            sumX += dx;
            sumY += dy;
        }

        Assert.Equal( 0.10f, sumX, precision: 3);
        Assert.Equal(-0.08f, sumY, precision: 3);
    }

    // ── Runtime update ────────────────────────────────────────────────────────

    [Fact]
    public void DriftCompensator_ProfileUpdateTakesEffectImmediately()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: 0.05f));
        var (dx1, _) = c.Compensate(1.0);
        Assert.Equal(0.05f, dx1, precision: 5);

        c.UpdateProfile(Profile(h: 0.10f));
        var (dx2, _) = c.Compensate(1.0);
        Assert.Equal(0.10f, dx2, precision: 5);
    }

    [Fact]
    public void DriftCompensator_CanDisableAtRuntime()
    {
        var c = Make();
        c.UpdateProfile(Profile(h: 0.05f));
        var (dx1, _) = c.Compensate(1.0);
        Assert.Equal(0.05f, dx1, precision: 5);

        c.UpdateProfile(new DriftCompensationProfile { Enabled = false,
            HorizontalOffsetPerSecond = 0.05f });
        var (dx2, _) = c.Compensate(1.0);
        Assert.Equal(0f, dx2);
    }

    [Fact]
    public void DriftCompensator_NullProfileThrows()
    {
        var c = Make();
        Assert.Throws<ArgumentNullException>(() => c.UpdateProfile(null!));
    }

    // ── Profile record equality ───────────────────────────────────────────────

    [Fact]
    public void DriftProfile_DisabledSingletonIsDisabled()
    {
        Assert.False(DriftCompensationProfile.Disabled.Enabled);
        Assert.Equal(0f, DriftCompensationProfile.Disabled.HorizontalOffsetPerSecond);
        Assert.Equal(0f, DriftCompensationProfile.Disabled.VerticalOffsetPerSecond);
    }

    [Fact]
    public void DriftProfile_RecordEqualityWorksByValue()
    {
        var a = Profile(h: 0.1f, v: -0.05f);
        var b = Profile(h: 0.1f, v: -0.05f);
        Assert.Equal(a, b);
    }
}
