using Nefarius.ViGEm.Client;
using Nefarius.ViGEm.Client.Targets;
using Nefarius.ViGEm.Client.Exceptions;
using Other;
using LogLevel = Other.LogManager.LogLevel;

namespace Aimmy2.Gamepad
{
    public sealed class VirtualGamepadOutput : IGamepadOutput
    {
        private ViGEmClient? _client;
        private IXbox360Controller? _controller;
        private bool _disposed;

        public bool IsConnected { get; private set; }

        public VirtualGamepadOutput()
        {
            Connect();
        }

        public void Connect()
        {
            if (IsConnected || _disposed)
                return;

            try
            {
                _client = new ViGEmClient();
                _controller = _client.CreateXbox360Controller();
                _controller.Connect();
                IsConnected = true;
            }
            catch (VigemBusNotFoundException ex)
            {
                LogManager.Log(LogLevel.Warning, $"ViGEm bus driver not found; gamepad assist output disabled: {ex.Message}");
                IsConnected = false;
                CleanUp();
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Warning, $"Could not initialize virtual gamepad output: {ex.Message}");
                IsConnected = false;
                CleanUp();
            }
        }

        public void SetRightStick(float rx, float ry)
        {
            if (!IsConnected || _controller == null)
                return;

            try
            {
                short x = MapAxis(rx);
                short y = MapAxis(ry);
                _controller.SetAxisValue(Xbox360Axis.RightThumbX, x);
                _controller.SetAxisValue(Xbox360Axis.RightThumbY, y);
                _controller.SubmitReport();
            }
            catch
            {
                // Never throw from the vision/output hot path if the virtual device drops.
                IsConnected = false;
            }
        }

        private static short MapAxis(float value)
        {
            float clamped = Math.Clamp(value, -1f, 1f);
            return clamped >= 0
                ? (short)Math.Round(clamped * short.MaxValue)
                : (short)Math.Round(clamped * -(float)short.MinValue);
        }

        public void Disconnect()
        {
            if (!IsConnected)
                return;

            try
            {
                _controller?.Disconnect();
            }
            catch
            {
            }
            finally
            {
                IsConnected = false;
            }
        }

        private void CleanUp()
        {
            try
            {
                _controller?.Disconnect();
            }
            catch
            {
            }

            _controller = null;
            _client?.Dispose();
            _client = null;
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;
            Disconnect();
            CleanUp();
        }
    }
}
