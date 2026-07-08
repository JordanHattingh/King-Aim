namespace Aimmy2.Gamepad
{
    public interface IGamepadOutput : IDisposable
    {
        bool IsConnected { get; }
        void SetRightStick(float rx, float ry);

        /// <summary>
        /// Passes through everything except the right stick from a physical controller reading,
        /// so the game only ever needs to bind to the virtual pad while the player keeps full
        /// control of movement, shooting, and buttons. Call once per frame alongside SetRightStick.
        /// </summary>
        void SetPassthroughState(PhysicalGamepadState physicalState);

        void Connect();
        void Disconnect();
    }
}
