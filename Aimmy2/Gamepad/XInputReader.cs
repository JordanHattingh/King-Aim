using System.Runtime.InteropServices;

namespace Aimmy2.Gamepad
{
    /// <summary>
    /// Reads a physical XInput-compatible controller (Xbox 360/One/Series pads, and most
    /// third-party pads in XInput mode) directly via XInput1_4.dll. No extra NuGet dependency:
    /// XInput ships with Windows, unlike the unmaintained SharpDX wrapper.
    /// </summary>
    public sealed class XInputReader
    {
        private const int ERROR_SUCCESS = 0;
        private const short AxisMax = short.MaxValue;
        private const short AxisMin = short.MinValue;
        private const byte TriggerMax = byte.MaxValue;

        [StructLayout(LayoutKind.Sequential)]
        private struct XInputGamepad
        {
            public ushort wButtons;
            public byte bLeftTrigger;
            public byte bRightTrigger;
            public short sThumbLX;
            public short sThumbLY;
            public short sThumbRX;
            public short sThumbRY;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct XInputState
        {
            public uint dwPacketNumber;
            public XInputGamepad Gamepad;
        }

        [DllImport("xinput1_4.dll", EntryPoint = "XInputGetState")]
        private static extern int XInputGetState(uint dwUserIndex, out XInputState pState);

        /// <summary>
        /// Reads the given controller slot (0-3). Returns PhysicalGamepadState.Disconnected
        /// if no controller is present at that index — never throws, since this is polled
        /// every frame on the AI loop and a missing controller is an expected, common state.
        /// </summary>
        public PhysicalGamepadState Read(uint userIndex = 0)
        {
            try
            {
                int result = XInputGetState(userIndex, out XInputState state);
                if (result != ERROR_SUCCESS)
                    return PhysicalGamepadState.Disconnected;

                var pad = state.Gamepad;
                return new PhysicalGamepadState(
                    Connected: true,
                    LeftStickX: NormalizeAxis(pad.sThumbLX),
                    LeftStickY: NormalizeAxis(pad.sThumbLY),
                    RightStickX: NormalizeAxis(pad.sThumbRX),
                    RightStickY: NormalizeAxis(pad.sThumbRY),
                    LeftTrigger: pad.bLeftTrigger / (float)TriggerMax,
                    RightTrigger: pad.bRightTrigger / (float)TriggerMax,
                    Buttons: pad.wButtons);
            }
            catch (DllNotFoundException)
            {
                return PhysicalGamepadState.Disconnected;
            }
        }

        /// <summary>
        /// Scans all 4 XInput slots and returns the index of the first connected controller,
        /// or 0 if none found. Use this at startup to auto-detect the active pad.
        /// </summary>
        public uint FindFirstConnectedIndex()
        {
            for (uint i = 0; i < 4; i++)
            {
                try
                {
                    if (XInputGetState(i, out _) == ERROR_SUCCESS)
                        return i;
                }
                catch (DllNotFoundException) { break; }
            }
            return 0;
        }

        /// <summary>Returns true if any XInput slot has a controller connected.</summary>
        public bool AnyConnected()
        {
            for (uint i = 0; i < 4; i++)
            {
                try { if (XInputGetState(i, out _) == ERROR_SUCCESS) return true; }
                catch (DllNotFoundException) { break; }
            }
            return false;
        }

        private static float NormalizeAxis(short raw) =>
            raw >= 0 ? raw / (float)AxisMax : raw / -(float)AxisMin;
    }
}
