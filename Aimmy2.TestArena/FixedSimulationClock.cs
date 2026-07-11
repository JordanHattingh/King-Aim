namespace Aimmy2.TestArena;

public sealed class FixedSimulationClock
{
    public const int FrequencyHz = 60;
    public static readonly DateTime EpochUtc = new(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    public long FrameIndex { get; private set; }
    public double DeltaSeconds => 1.0 / FrequencyHz;
    public DateTime CurrentTimestamp => EpochUtc.AddTicks(
        checked(FrameIndex * TimeSpan.TicksPerSecond / FrequencyHz));

    public void Reset() => FrameIndex = 0;

    public DateTime Advance()
    {
        FrameIndex++;
        return CurrentTimestamp;
    }
}
