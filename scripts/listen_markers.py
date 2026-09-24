"""Zero-overhead global hotkey listener & anonymous input logger for screencasting.

Features:
1. Time Synchronization:
   - Automatic via OBS WebSocket (port 4455, RecordStateChanged).
   - Handles OBS Pause & Resume without timeline drift.
   - Manual fallback via [F8] (hit F8 when starting recording).
2. Configurable Bad Take Marking:
   - Default: [Pause] key (VK_PAUSE = 0x13, no conflict with VS Code/IntelliJ).
   - Configurable to F9, F12, etc.
   - Hotkey and modifier keys (Ctrl, Alt, Shift) are NEVER logged as typing.
3. Privacy-Guaranteed Input Logging:
   - Logs timestamps of typing, clicks, scroll, and foreground PROCESS name (e.g. Code.exe).
   - NEVER logs window titles (no file paths, URLs, or secrets).
   - NEVER logs key characters or content (not a keylogger).

Usage:
    python scripts/listen_markers.py [output_json_path] [--key pause] [--port 4455] [--password secret]
"""

import argparse
import ctypes
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

try:
    import pynput
    from pynput.keyboard import Key
except ImportError:
    pynput = None
    Key = None

# Virtual key map for Windows
VK_MAP = {
    "pause": 0x13,
    "scroll_lock": 0x91,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}

MODIFIER_KEYS = {
    Key.ctrl, Key.ctrl_l, Key.ctrl_r,
    Key.alt, Key.alt_l, Key.alt_r, Key.alt_gr,
    Key.shift, Key.shift_l, Key.shift_r,
    Key.cmd, Key.cmd_l, Key.cmd_r,
    Key.caps_lock, Key.num_lock, Key.scroll_lock,
    Key.pause,
} if Key else set()


def _play_beep(beep_type: int = 0) -> None:
    try:
        ctypes.windll.user32.MessageBeep(beep_type)
    except Exception:
        pass


def _get_active_process_name() -> str:
    """Return process executable name (e.g. Code.exe), never full window title."""
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return ""
        hproc = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not hproc:
            return ""
        buf = ctypes.create_unicode_buffer(512)
        size = ctypes.c_ulong(512)
        name = ""
        if kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(size)):
            name = Path(buf.value).name
        kernel32.CloseHandle(hproc)
        return name
    except Exception:
        return ""


class ObsSyncHandler:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.record_start_monotonic: float | None = None
        self.is_recording = False
        self.is_paused = False
        self.paused_at_monotonic: float | None = None
        self.total_paused_duration: float = 0.0
        self.recording_file: str | None = None
        self.client = None

    def get_elapsed(self) -> float | None:
        if self.record_start_monotonic is None or not self.is_recording:
            return None
        now = time.monotonic()
        current_pause = (now - self.paused_at_monotonic) if (self.is_paused and self.paused_at_monotonic) else 0.0
        return max(0.0, (now - self.record_start_monotonic) - (self.total_paused_duration + current_pause))

    def try_connect(self, host: str = "localhost", port: int = 4455, password: str = "") -> bool:
        try:
            import obsws_python as obs
            client = obs.EventClient(host=host, port=port, password=password, timeout=2)

            def on_record_state_changed(data):
                state = getattr(data, "output_state", None) or getattr(data, "outputState", None)
                active = getattr(data, "output_active", None)
                if active is None:
                    active = getattr(data, "outputActive", False)
                path = getattr(data, "output_path", None) or getattr(data, "outputPath", None)

                if active and not self.is_recording:
                    self.record_start_monotonic = time.monotonic()
                    self.total_paused_duration = 0.0
                    self.is_recording = True
                    self.is_paused = False
                    self.recording_file = path
                    print(f"\n[OBS] >>> RECORDING STARTED! <<< (File: {path})")
                    print("[OBS] Markers & input events are synchronized with video start.")
                    _play_beep(0x00000040)
                elif state == "OBS_WEBSOCKET_OUTPUT_PAUSED" and not self.is_paused:
                    self.is_paused = True
                    self.paused_at_monotonic = time.monotonic()
                    print("\n[OBS] >>> RECORDING PAUSED <<<")
                elif state == "OBS_WEBSOCKET_OUTPUT_RESUMED" and self.is_paused:
                    if self.paused_at_monotonic:
                        self.total_paused_duration += time.monotonic() - self.paused_at_monotonic
                    self.is_paused = False
                    self.paused_at_monotonic = None
                    print("\n[OBS] >>> RECORDING RESUMED <<< (Compensated pause duration)")
                elif not active and self.is_recording:
                    self.is_recording = False
                    self.is_paused = False
                    print(f"\n[OBS] >>> RECORDING STOPPED <<< (Net recorded: {self.get_elapsed():.2f}s)")
                    _play_beep(0x00000040)

            client.callback.register(on_record_state_changed)
            self.client = client
            return True
        except Exception:
            return False


def listen(
    output_path: Path,
    bad_take_key: str = "f12",
    sync_key: str = "scroll_lock",
    obs_port: int = 4455,
    obs_password: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    if output_path.exists():
        try:
            events = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            events = []

    vk_bad_take = VK_MAP.get(bad_take_key.lower(), 0x7B)  # default F12
    vk_sync = VK_MAP.get(sync_key.lower(), 0x91)          # default ScrollLock
    user32 = ctypes.windll.user32
    handler = ObsSyncHandler(output_path)
    has_obs = handler.try_connect(port=obs_port, password=obs_password)

    script_launch_time = time.monotonic()
    manual_start_time: float | None = None
    lock = threading.Lock()

    def get_elapsed_seconds() -> tuple[float, str]:
        obs_elapsed = handler.get_elapsed()
        if obs_elapsed is not None:
            return obs_elapsed, "OBS"
        elif manual_start_time is not None:
            return max(0.0, time.monotonic() - manual_start_time), f"Manual-{sync_key.upper()}"
        return max(0.0, time.monotonic() - script_launch_time), "Unsynced"

    def record_input_event(event_type: str, extra: dict | None = None) -> None:
        elapsed, sync_source = get_elapsed_seconds()
        item = {"time": round(elapsed, 3), "type": event_type}
        if extra:
            item.update(extra)
        with lock:
            events.append(item)

    # Setup background pynput hooks for anonymous typing and clicks
    last_key_time = 0.0
    last_scroll_time = 0.0
    last_marker_monotonic = 0.0

    def on_key_press(key):
        nonlocal last_key_time
        # Filter out modifiers, bad_take key, and manual sync key
        if key in MODIFIER_KEYS:
            return
        if hasattr(key, "name") and key.name and key.name.lower() in (bad_take_key.lower(), sync_key.lower()):
            return
        if hasattr(key, "char") and key.char and key.char.lower() in (bad_take_key.lower(), sync_key.lower()):
            return
        # Suppress key events for only 0.15s after marker to prevent modifier tails without swallowing typing
        if time.monotonic() - last_marker_monotonic < 0.15:
            return

        now = time.monotonic()
        if now - last_key_time >= 0.1:  # Throttle typing to 10/s
            last_key_time = now
            record_input_event("key")

    def on_click(x, y, button, pressed):
        if pressed:
            record_input_event("click")

    def on_scroll(x, y, dx, dy):
        nonlocal last_scroll_time
        now = time.monotonic()
        if now - last_scroll_time >= 0.2:
            last_scroll_time = now
            record_input_event("scroll")

    if pynput:
        k_listener = pynput.keyboard.Listener(on_press=on_key_press)
        m_listener = pynput.mouse.Listener(on_click=on_click, on_scroll=on_scroll)
        k_listener.daemon = True
        m_listener.daemon = True
        k_listener.start()
        m_listener.start()

    print("=" * 65)
    print(f"[*] Autocut Input Logger & Marker Listener Active")
    print(f"[*] Bad take hotkey: [{bad_take_key.upper()}] (Safe from IDE conflicts & conhost)")
    print(f"[*] Manual sync key: [{sync_key.upper()}] (Hit when starting OBS recording)")
    print(f"[*] Target output file: {output_path.resolve()}")
    if has_obs:
        print("[+] CONNECTED TO OBS WEBSOCKET!")
        print("    -> Time zero & pause/resume synchronize automatically with OBS.")
    else:
        print("[-] OBS WebSocket not connected on localhost:4455.")
        print(f"    -> Press [{sync_key.upper()}] when you click 'Start Recording' in OBS.")
    print(f"    -> Press [{bad_take_key.upper()}] whenever you make a mistake / bad take.")
    print("[*] Privacy: Timestamps and process names only (NO window titles or key text).")
    print("[*] Press [Ctrl+C] to exit listener.")
    print("=" * 65)

    was_sync_pressed = False
    was_marker_pressed = False
    last_save_time = time.monotonic()
    last_proc_name = ""

    try:
        while True:
            # Check manual sync key
            sync_pressed = bool(user32.GetAsyncKeyState(vk_sync) & 0x8000)
            if sync_pressed and not was_sync_pressed:
                manual_start_time = time.monotonic()
                print(f"\n[!] [{sync_key.upper()}] RECORDING START SYNCED! Time zero set to now.")
                _play_beep(0x00000040)
            was_sync_pressed = sync_pressed

            # Check Bad Take key
            marker_pressed = bool(user32.GetAsyncKeyState(vk_bad_take) & 0x8000)
            if marker_pressed and not was_marker_pressed:
                last_marker_monotonic = time.monotonic()
                elapsed, sync_src = get_elapsed_seconds()
                marker = {
                    "time": round(elapsed, 3),
                    "type": "bad_take",
                    "sync": sync_src,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                with lock:
                    events.append(marker)
                print(f"[!] BAD TAKE marker recorded at {elapsed:.2f}s ({sync_src}) -> saved to {output_path.name}")
                _play_beep(0x00000030)
            was_marker_pressed = marker_pressed

            # Check active process name every 1s
            now = time.monotonic()
            if now - last_save_time >= 1.0:
                cur_proc = _get_active_process_name()
                if cur_proc and cur_proc != last_proc_name:
                    last_proc_name = cur_proc
                    record_input_event("process", {"process": cur_proc})

                with lock:
                    output_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
                last_save_time = now

            time.sleep(0.05)
    except KeyboardInterrupt:
        with lock:
            output_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
        bad_takes = sum(1 for e in events if e.get("type") == "bad_take")
        keys = sum(1 for e in events if e.get("type") == "key")
        clicks = sum(1 for e in events if e.get("type") == "click")
        print(f"\n[*] Stopped. Total events: {len(events)} (Bad takes: {bad_takes}, Keys: {keys}, Clicks: {clicks})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBS Hotkey & input logger for autocut")
    parser.add_argument("output", nargs="?", default="work/markers.json", help="Path to markers.json")
    parser.add_argument("--key", default="f12", help="Hotkey for bad take marker (f12, scroll_lock, pause, etc.)")
    parser.add_argument("--sync-key", default="scroll_lock", help="Hotkey for manual start sync (scroll_lock, f11, etc.)")
    parser.add_argument("--port", type=int, default=4455, help="OBS WebSocket port")
    parser.add_argument("--password", default="", help="OBS WebSocket password")
    args = parser.parse_args()

    listen(Path(args.output), bad_take_key=args.key, sync_key=args.sync_key, obs_port=args.port, obs_password=args.password)
