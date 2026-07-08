namespace Aimmy2.Gamepad
{
    public interface IGamepadOutput : IDisposable
    {
        bool IsConnected { get; }
        void SetRightStick(float rx, float ry);
        void Connect();
        void Disconnect();
    }
}
