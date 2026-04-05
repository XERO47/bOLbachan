"""
DeepFace Live — Webcam Tray Utility
=====================================
Sits in the Windows system tray.
Right-click → Start Streaming  : captures webcam, sends frames to server,
                                  feeds processed frames into a virtual camera
Right-click → Stop Streaming   : stops
Right-click → Quit

Install:
    pip install pystray pillow opencv-python websockets numpy pyvirtualcam

Virtual camera (pick one driver — install once):
    OBS Studio  https://obsproject.com/  (use its built-in virtual camera driver)
    OR OBS-Camera plugin for just the driver without full OBS

Run:
    python tray.py
    python tray.py --server ws://205.147.101.226:8000/ws/deepfake --start
"""

import argparse
import asyncio
import io
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("pip install pystray pillow")

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets")

try:
    import pyvirtualcam
    _VCAM_AVAILABLE = True
except ImportError:
    _VCAM_AVAILABLE = False
    print("[tray] pyvirtualcam not installed — virtual camera disabled"
          " (pip install pyvirtualcam)")

# ── Config ────────────────────────────────────────────────────────────────────

SETTINGS_FILE = Path.home() / ".deepface_tray.json"
DEFAULT_SERVER = "ws://205.147.102.96:8000/ws/deepfake"
DEFAULT_CAM    = 0
FPS            = 25
JPEG_QUALITY   = 85
CAP_W, CAP_H   = 640, 480

# ── Settings persistence ──────────────────────────────────────────────────────

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {"server": DEFAULT_SERVER, "cam": DEFAULT_CAM}

def save_settings(s):
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))

# ── Icons ─────────────────────────────────────────────────────────────────────

def make_icon(streaming: bool) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (34, 197, 94) if streaming else (100, 100, 100)  # green / grey
    d.ellipse([4, 4, size-4, size-4], fill=color)
    if streaming:
        # White pause bars
        d.rectangle([20, 18, 28, 46], fill=(255, 255, 255))
        d.rectangle([36, 18, 44, 46], fill=(255, 255, 255))
    else:
        # White play triangle
        d.polygon([(20, 16), (20, 48), (48, 32)], fill=(255, 255, 255))
    return img

# ── Streaming loop ────────────────────────────────────────────────────────────

_stop_evt = threading.Event()
_stream_thread = None

async def _ws_loop(server_url: str, cam_idx: int):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    interval = 1.0 / FPS

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print(f"[tray] Cannot open camera {cam_idx}")
        return

    print(f"[tray] Connecting to {server_url}")
    try:
        async with websockets.connect(
            server_url,
            max_size=20_000_000,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print("[tray] Connected — streaming")
            async def _recv_to_vcam(vcam):
                """Receive processed frames from server → push to virtual camera."""
                async for data in ws:
                    if _stop_evt.is_set():
                        break
                    if not isinstance(data, bytes) or vcam is None:
                        continue
                    arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    # pyvirtualcam expects RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (CAP_H, CAP_W):
                        rgb = cv2.resize(rgb, (CAP_W, CAP_H))
                    vcam.send(rgb)
                    vcam.sleep_until_next_frame()

            vcam = None
            if _VCAM_AVAILABLE:
                try:
                    vcam = pyvirtualcam.Camera(
                        width=CAP_W, height=CAP_H, fps=FPS,
                        fmt=pyvirtualcam.PixelFormat.RGB,
                    )
                    print(f"[tray] Virtual camera: {vcam.device}")
                except Exception as e:
                    print(f"[tray] Virtual camera failed ({e}) — frames will be discarded")
                    vcam = None

            drain_task = asyncio.create_task(_recv_to_vcam(vcam))
            try:
                while not _stop_evt.is_set():
                    t0 = time.monotonic()
                    ret, frame = cap.read()
                    if ret:
                        ok, jpg = cv2.imencode(".jpg", frame, encode_params)
                        if ok:
                            await ws.send(jpg.tobytes())
                    elapsed = time.monotonic() - t0
                    await asyncio.sleep(max(0.001, interval - elapsed))
            finally:
                drain_task.cancel()
                if vcam is not None:
                    vcam.close()
    except Exception as e:
        print(f"[tray] Connection error: {e}")
    finally:
        cap.release()
        print("[tray] Stopped")

def _run_stream(server_url, cam_idx):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_ws_loop(server_url, cam_idx))
    finally:
        loop.close()

# ── Tray app ──────────────────────────────────────────────────────────────────

class TrayApp:
    def __init__(self, server_url: str):
        self.settings = load_settings()
        if server_url != DEFAULT_SERVER:
            self.settings["server"] = server_url
            save_settings(self.settings)

        self.streaming = False
        self._icon = pystray.Icon(
            "DeepFace Live",
            make_icon(False),
            "DeepFace Live — Idle",
            menu=pystray.Menu(
                pystray.MenuItem("Start Streaming", self.start, default=True),
                pystray.MenuItem("Stop Streaming",  self.stop),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    lambda item: f"Server: {self.settings['server'][:40]}...",
                    None, enabled=False
                ),
                pystray.MenuItem(
                    lambda item: f"Camera: {self.settings['cam']}",
                    None, enabled=False
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self.quit),
            )
        )

    def start(self):
        global _stream_thread, _stop_evt
        if self.streaming:
            return
        _stop_evt.clear()
        _stream_thread = threading.Thread(
            target=_run_stream,
            args=(self.settings["server"], self.settings["cam"]),
            daemon=True,
        )
        _stream_thread.start()
        self.streaming = True
        self._icon.icon  = make_icon(True)
        self._icon.title = "DeepFace Live — Streaming"
        print("[tray] Streaming started")

    def stop(self):
        global _stop_evt
        if not self.streaming:
            return
        _stop_evt.set()
        self.streaming = False
        self._icon.icon  = make_icon(False)
        self._icon.title = "DeepFace Live — Idle"
        print("[tray] Streaming stopped")

    def quit(self):
        self.stop()
        self._icon.stop()

    def run(self):
        print(f"[tray] Starting — server: {self.settings['server']}")
        print("[tray] Right-click the tray icon to start streaming")
        self._icon.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--cam",    type=int, default=None)
    parser.add_argument("--start",  action="store_true", help="Start streaming immediately")
    args = parser.parse_args()

    app = TrayApp(args.server)

    if args.cam is not None:
        app.settings["cam"] = args.cam
        save_settings(app.settings)

    if args.start:
        # Start streaming immediately when tray opens
        threading.Timer(1.0, app.start).start()

    app.run()


if __name__ == "__main__":
    main()
