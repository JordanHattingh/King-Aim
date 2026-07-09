namespace Aimmy2.Gamepad
{
    public interface IGamepadOutput : IDisposable
    {
        bool IsConnected { get; }

        /// <summary>
        /// Atomically applies the full controller state — passthrough inputs from the physical
        /// pad plus the AI-computed right-stick override — in a single USB report. Use this
        /// instead of calling SetPassthroughState + SetRightStick separately, which produces
        /// two USB packets per frame and can cause button-state jitter between them.
        /// </summary>
        void SetFullState(PhysicalGamepadState physicalState, float rx, float ry, float? rtOverride = null);

        void SetRightStick(float rx, float ry);
        void SetPassthroughState(PhysicalGamepadState physicalState);

        void Connect();
        void Disconnect();
    }
}
