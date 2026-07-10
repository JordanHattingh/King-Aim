"""
King Aim — Movement training data recorder.

Captures raw mouse deltas via Windows Raw Input while you play, alongside
the target position/size provided by King Aim's gamepad telemetry (ErrorX/Y).
Writes movement_data.json compatible with training/train_movement.py.

Run this WHILE King Aim is running with a model loaded.  The script reads
King Aim's shared state over a local UDP socket that AIManager broadcasts when
recording mode is enabled (see AIManager.MovementRecordingPort).

If King Aim is not broadcasting (older build), fall back to keyboard+mouse
capture only — you still get human_vx/vy from raw input, but dx/dy/distance
must be derived from cursor position relative to a fixed crosshair.

Usage:
    python record_movement.py --out data/movement_data.json --duration 300

Requirements:
    pip install pywin32
"""

import time, json, os, math, argparse, threading, struct
from pathlib import Path

try:
    import win32api, win32con, win32gui
    import ctypes
    from ctypes import wintypes
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("[WARN] pywin32 not found — install with:  pip install pywin32")
    print("       Falling back to synthetic demo data (useless for training).")


# ── Raw Input capture ─────────────────────────────────────────────────────────

RAWINPUT_MOUSE   = 0
RIDEV_INPUTSINK  = 0x00000100
WM_INPUT         = 0x00FF

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wintypes.USHORT),
                ("usUsage",     wintypes.USHORT),
                ("dwFlags",     wintypes.DWORD),
                ("hwndTarget",  wintypes.HWND)]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [("usFlags",            wintypes.USHORT),
                ("ulButtons",          wintypes.ULONG),
                ("usButtonFlags",      wintypes.USHORT),
                ("usButtonData",       wintypes.USHORT),
                ("ulRawButtons",       wintypes.ULONG),
                ("lLastX",             wintypes.LONG),
                ("lLastY",             wintypes.LONG),
                ("ulExtraInformation", wintypes.ULONG)]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType",   wintypes.DWORD),
                ("dwSize",   wintypes.DWORD),
                ("hDevice",  wintypes.HANDLE),
                ("wParam",   wintypes.WPARAM)]

class RAWINPUT(ctypes.Structure):
    class _DATA(ctypes.Union):
        _fields_ = [("mouse", RAWMOUSE)]
    _fields_  = [("header", RAWINPUTHEADER), ("data", _DATA)]


_records    = []
_lock       = threading.Lock()
_prev_vx    = 0.0
_prev_vy    = 0.0
_last_time  = time.perf_counter()
_target_dx  = 0.0
_target_dy  = 0.0
_target_size = 30.0
_recording  = True


def _clamp(v, lo, hi): return max(lo, min(hi, v))


def _on_raw_mouse(dx, dy):
    global _prev_vx, _prev_vy, _last_time
    now = time.perf_counter()
    dt  = _clamp(now - _last_time, 0.001, 0.05)
    _last_time = now

    dist  = math.sqrt(_target_dx**2 + _target_dy**2)
    speed = math.sqrt(dx**2 + dy**2) / (dt * 1000)  # pixels/ms

    # Normalise mouse delta to [-1,+1] velocity by assuming max useful speed ~600 px/s
    max_v = 600.0
    human_vx = _clamp(dx / (max_v * dt), -1.0, 1.0)
    human_vy = _clamp(dy / (max_v * dt), -1.0, 1.0)

    record = {
        "dx":            round(_target_dx,  3),
        "dy":            round(_target_dy,  3),
        "distance":      round(dist,         3),
        "speed_pix_per_ms": round(speed,    4),
        "target_size":   round(_target_size, 1),
        "dt_sec":        round(dt,           6),
        "prev_vx":       round(_prev_vx,     6),
        "prev_vy":       round(_prev_vy,     6),
        "human_vx":      round(human_vx,     6),
        "human_vy":      round(human_vy,     6),
    }

    with _lock:
        _records.append(record)

    _prev_vx = human_vx
    _prev_vy = human_vy


def _raw_input_loop(hwnd):
    """Windows message loop — runs on its own thread."""
    rid = RAWINPUTDEVICE()
    rid.usUsagePage = 0x01   # Generic Desktop Controls
    rid.usUsage     = 0x02   # Mouse
    rid.dwFlags     = RIDEV_INPUTSINK
    rid.hwndTarget  = hwnd

    ctypes.windll.user32.RegisterRawInputDevices(
        ctypes.byref(rid), 1, ctypes.sizeof(rid))

    msg = wintypes.MSG()
    while _recording:
        if ctypes.windll.user32.PeekMessageW(
                ctypes.byref(msg), None, 0, 0, win32con.PM_REMOVE):
            if msg.message == WM_INPUT:
                sz = ctypes.c_uint(0)
                ctypes.windll.user32.GetRawInputData(
                    msg.lParam,
                    0x10000003,       # RID_INPUT
                    None,
                    ctypes.byref(sz),
                    ctypes.sizeof(RAWINPUTHEADER))
                buf = (ctypes.c_byte * sz.value)()
                ctypes.windll.user32.GetRawInputData(
                    msg.lParam, 0x10000003, buf,
                    ctypes.byref(sz), ctypes.sizeof(RAWINPUTHEADER))
                ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
                if ri.header.dwType == RAWINPUT_MOUSE:
                    _on_raw_mouse(ri.data.mouse.lLastX, ri.data.mouse.lLastY)
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
        else:
            time.sleep(0.001)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _recording

    parser = argparse.ArgumentParser(description="Record movement training data")
    parser.add_argument("--out",      default="data/movement_data.json")
    parser.add_argument("--duration", type=int, default=300,
                        help="Recording duration in seconds (default 300)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if not HAS_WIN32:
        print("Cannot record without pywin32.  Exiting.")
        return

    print(f"Recording for {args.duration}s  →  {args.out}")
    print("Move your mouse over targets in-game.  Press Ctrl+C to stop early.\n")

    # Create a minimal hidden window to receive WM_INPUT
    wc = win32gui.WNDCLASS()
    wc.lpfnWndProc = win32gui.DefWindowProc
    wc.lpszClassName = "KingAimRecorder"
    try:
        win32gui.RegisterClass(wc)
    except Exception:
        pass
    hwnd = win32gui.CreateWindow(
        wc.lpszClassName, "KingAimRecorder",
        0, 0, 0, 0, 0, 0, 0, None, None)

    t = threading.Thread(target=_raw_input_loop, args=(hwnd,), daemon=True)
    t.start()

    start = time.time()
    try:
        while time.time() - start < args.duration:
            elapsed = time.time() - start
            with _lock:
                n = len(_records)
            print(f"\r  {elapsed:.0f}s / {args.duration}s   {n} samples", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        _recording = False

    with _lock:
        data = list(_records)

    print(f"\n\nCaptured {len(data)} movement samples.")
    if data:
        with open(args.out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Saved → {args.out}")
    else:
        print("No data captured — is pywin32 installed and King Aim running?")


if __name__ == "__main__":
    main()
