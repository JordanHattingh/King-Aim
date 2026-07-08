namespace Aimmy2.Gamepad
{
    public readonly record struct PhysicalGamepadState(
        bool Connected,
        float LeftStickX,
        float LeftStickY,
        float RightStickX,
        float RightStickY,
        float LeftTrigger,
        float RightTrigger,
        ushort Buttons)
    {
        public static readonly PhysicalGamepadState Disconnected = new(false, 0, 0, 0, 0, 0, 0, 0);
    }
}
