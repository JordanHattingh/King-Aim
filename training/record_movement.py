"""
King Aim — controlled TestArena/accessibility pointing recorder.

The previous recorder claimed to consume King Aim UDP telemetry but never created a
socket or updated target_dx/target_dy. It therefore logged zero target errors and
produced corrupt training data. This v2 recorder intentionally FAILS CLOSED until
it receives explicit TestArena target telemetry over UDP.

Telemetry endpoint (default 127.0.0.1:28761) JSON datagram:
    {
      "source": "testarena_pointing",
      "session_id": "session-...",
      "task_id": "horizontal-01",
      "dx": 45.2,
      "dy": -12.1,
      "target_size": 38.0
    }

Only source="testarena_pointing" is accepted. The recorder captures Windows Raw
Input mouse deltas and writes generic pointing samples. It is not a live game aim
recorder.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import socket
import threading
import time
from ctypes import wintypes
from pathlib import Path

try:
    import win32con
    import win32gui

    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


RAWINPUT_MOUSE = 0
RIDEV_INPUTSINK = 0x00000100
WM_INPUT = 0x00FF


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("ulButtons", wintypes.ULONG),
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWINPUT(ctypes.Structure):
    class _DATA(ctypes.Union):
        _fields_ = [("mouse", RAWMOUSE)]

    _fields_ = [("header", RAWINPUTHEADER), ("data", _DATA)]


records: list[dict] = []
lock = threading.Lock()
recording = True
prev_vx = 0.0
prev_vy = 0.0
last_time = time.perf_counter()
telemetry: dict | None = None
telemetry_received_at = 0.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def telemetry_loop(host: str, port: int) -> None:
    global telemetry, telemetry_received_at
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.25)
    while recording:
        try:
            payload, _ = sock.recvfrom(4096)
            message = json.loads(payload.decode("utf-8"))
            if message.get("source") != "testarena_pointing":
                continue
            required = {"session_id", "task_id", "dx", "dy", "target_size"}
            if not required.issubset(message):
                continue
            with lock:
                telemetry = message
                telemetry_received_at = time.perf_counter()
        except socket.timeout:
            continue
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    sock.close()


def on_raw_mouse(dx: int, dy: int) -> None:
    global prev_vx, prev_vy, last_time
    now = time.perf_counter()
    dt = clamp(now - last_time, 0.001, 0.05)
    last_time = now

    with lock:
        target = dict(telemetry) if telemetry is not None else None
        age = now - telemetry_received_at
    if target is None or age > 0.100:
        return

    target_dx = float(target["dx"])
    target_dy = float(target["dy"])
    target_size = float(target["target_size"])
    distance = math.hypot(target_dx, target_dy)
    speed = math.hypot(dx, dy) / (dt * 1000.0)
    max_velocity = 600.0
    human_vx = clamp(dx / (max_velocity * dt), -1.0, 1.0)
    human_vy = clamp(dy / (max_velocity * dt), -1.0, 1.0)

    record = {
        "source": "testarena_pointing",
        "session_id": str(target["session_id"]),
        "task_id": str(target["task_id"]),
        "timestamp_us": int(time.perf_counter_ns() // 1000),
        "dx": round(target_dx, 3),
        "dy": round(target_dy, 3),
        "distance": round(distance, 3),
        "speed_pix_per_ms": round(speed, 4),
        "target_size": round(target_size, 3),
        "dt_sec": round(dt, 6),
        "prev_vx": round(prev_vx, 6),
        "prev_vy": round(prev_vy, 6),
        "human_vx": round(human_vx, 6),
        "human_vy": round(human_vy, 6),
    }
    with lock:
        records.append(record)
    prev_vx = human_vx
    prev_vy = human_vy


def raw_input_loop(hwnd) -> None:
    device = RAWINPUTDEVICE()
    device.usUsagePage = 0x01
    device.usUsage = 0x02
    device.dwFlags = RIDEV_INPUTSINK
    device.hwndTarget = hwnd
    ctypes.windll.user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device))

    msg = wintypes.MSG()
    while recording:
        if ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, win32con.PM_REMOVE):
            if msg.message == WM_INPUT:
                size = ctypes.c_uint(0)
                ctypes.windll.user32.GetRawInputData(
                    msg.lParam, 0x10000003, None, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER)
                )
                buffer = (ctypes.c_byte * size.value)()
                ctypes.windll.user32.GetRawInputData(
                    msg.lParam, 0x10000003, buffer, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER)
                )
                raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
                if raw.header.dwType == RAWINPUT_MOUSE:
                    on_raw_mouse(raw.data.mouse.lLastX, raw.data.mouse.lLastY)
            ctypes.windll.user32.TranslateMessage(ctypes.byref(msg))
            ctypes.windll.user32.DispatchMessageW(ctypes.byref(msg))
        else:
            time.sleep(0.001)


def main() -> None:
    global recording
    parser = argparse.ArgumentParser(description="Record controlled TestArena pointing data")
    parser.add_argument("--out", default="data/movement_data.json")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--telemetry-host", default="127.0.0.1")
    parser.add_argument("--telemetry-port", type=int, default=28761)
    args = parser.parse_args()

    if not HAS_WIN32:
        raise SystemExit("pywin32 is required: pip install pywin32")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    window_class = win32gui.WNDCLASS()
    window_class.lpfnWndProc = win32gui.DefWindowProc
    window_class.lpszClassName = "KingAimPointingRecorder"
    try:
        win32gui.RegisterClass(window_class)
    except Exception:
        pass
    hwnd = win32gui.CreateWindow(
        window_class.lpszClassName,
        "KingAimPointingRecorder",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
    )

    threading.Thread(target=telemetry_loop, args=(args.telemetry_host, args.telemetry_port), daemon=True).start()
    threading.Thread(target=raw_input_loop, args=(hwnd,), daemon=True).start()

    print(f"Waiting for TestArena telemetry on udp://{args.telemetry_host}:{args.telemetry_port}")
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < args.duration:
            with lock:
                sample_count = len(records)
                has_telemetry = telemetry is not None and time.perf_counter() - telemetry_received_at <= 0.100
            print(
                f"\r{time.perf_counter()-start:6.1f}s/{args.duration}s "
                f"samples={sample_count} telemetry={'OK' if has_telemetry else 'WAIT'}",
                end="",
                flush=True,
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        recording = False

    with lock:
        data = list(records)
    print()
    if not data:
        raise SystemExit(
            "No valid TestArena pointing samples were captured. The recorder will not write zero-target data."
        )
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))
    print(f"Saved {len(data)} samples -> {out_path}")


if __name__ == "__main__":
    main()
