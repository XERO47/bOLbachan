"""
DeepFace Tray — Virtual Camera Client
======================================
Captures webcam → sends to deepfake server → receives processed frames
→ pushes to OBS Virtual Camera (visible in Meet/Zoom/Teams).

Config: edit config.json (created by install.bat)
  server   : WebSocket URL of the deepfake server
  cam      : camera index (0 = default webcam)
  fps      : capture FPS (default 25)
  quality  : JPEG quality sent to server (default 85)

Run:
  start_tray.bat
  python tray.py
"""

import argparse
import asyncio
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
    raise SystemExit("Run install.bat first  (pip install pystray pillow)")

try:
    import websockets
except ImportError:
    raise SystemExit("Run install.bat first  (pip install websockets)")

try:
    import pyvirtualcam
    _VCAM_AVAILABLE = True
except ImportError:
    _VCAM_AVAILABLE = False
    print("[tray] pyvirtualcam not installed — run install.bat")

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_FILE = Path(__file__).parent / "config.json"
_DEFAULTS = {
    "server":  "ws://205.147.102.96:8000/ws/deepfake",
    "cam":     0,
    "fps":     25,
    "quality": 85,
}

def load_config() -> dict:
    if _CONFIG_FILE.exists():
        try:
            data = json.loads(_CONFIG_FILE.read_text())
            return {**_DEFAULTS, **data}
        except Exception:
            pass
    return dict(_DEFAULTS)

def save_config(cfg: dict):
    _CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

# ── Icons ─────────────────────────────────────────────────────────────────────

def _make_icon(streaming: bool) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (34, 197, 94) if streaming else (100, 100, 100)
    d.ellipse([4, 4, 60, 60], fill=color)
    if streaming:
        d.rectangle([20, 18, 28, 46], fill=(255, 255, 255))
        d.rectangle([36, 18, 44, 46], fill=(255, 255, 255))
    else:
        d.polygon([(20, 16), (20, 48), (48, 32)], fill=(255, 255, 255))
    return img

# ── Streaming ─────────────────────────────────────────────────────────────────

_stop_evt   = threading.Event()
_stream_thr = None

async def _ws_loop(cfg: dict):
    server  = cfg["server"]
    cam_idx = cfg["cam"]
    fps     = cfg["fps"]
    quality = cfg["quality"]
    w, h    = 640, 480
    interval = 1.0 / fps
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    # Try backends in order — external USB webcams often need MSMF on Windows
    cap = None
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, -1]:
        try:
            c = cv2.VideoCapture(cam_idx, backend) if backend != -1 else cv2.VideoCapture(cam_idx)
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    cap = c
                    print(f"[tray] Camera {cam_idx} opened")
                    break
                c.release()
        except Exception:
            pass

    if cap is None:
        print(f"[tray] Cannot open camera {cam_idx} — try a different index in config.json")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # ── Virtual camera ────────────────────────────────────────────────────────
    vcam = None
    if _VCAM_AVAILABLE:
        try:
            vcam = pyvirtualcam.Camera(
                width=w, height=h, fps=fps,
                fmt=pyvirtualcam.PixelFormat.RGB,
            )
            print(f"[tray] Virtual camera ready: {vcam.device}")
            print(f"[tray] Select '{vcam.device}' in Meet/Zoom/Teams")
        except Exception as e:
            print(f"[tray] Virtual camera unavailable: {e}")
            print("[tray] Make sure OBS is installed and Start Virtual Camera was clicked once")

    print(f"[tray] Connecting to {server}")
    try:
        async with websockets.connect(
            server,
            max_size=20_000_000,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print("[tray] Connected — streaming started")

            # ── Receive loop: server frames → virtual camera ──────────────────
            async def _recv():
                async for data in ws:
                    if _stop_evt.is_set():
                        break
                    if not isinstance(data, bytes) or vcam is None:
                        continue
                    arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if rgb.shape[:2] != (h, w):
                        rgb = cv2.resize(rgb, (w, h))
                    vcam.send(rgb)

            recv_task = asyncio.create_task(_recv())

            # ── Send loop: webcam → server ────────────────────────────────────
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
                recv_task.cancel()

    except Exception as e:
        print(f"[tray] Connection error: {e}")
    finally:
        cap.release()
        if vcam:
            vcam.close()
        print("[tray] Stopped")

def _run_stream(cfg: dict):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_ws_loop(cfg))
    finally:
        loop.close()

# ── Tray App ──────────────────────────────────────────────────────────────────

class TrayApp:
    def __init__(self):
        self.cfg = load_config()
        self.streaming = False
        self._icon = pystray.Icon(
            "DeepFace",
            _make_icon(False),
            "DeepFace — Idle",
            menu=pystray.Menu(
                pystray.MenuItem("Start", self.start, default=True),
                pystray.MenuItem("Stop",  self.stop),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    lambda _: f"Server: {self.cfg['server'].split('//')[1].split('/')[0]}",
                    None, enabled=False,
                ),
                pystray.MenuItem(
                    lambda _: f"Camera: {self.cfg['cam']}",
                    None, enabled=False,
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self.quit),
            ),
        )

    def start(self):
        global _stream_thr, _stop_evt
        if self.streaming:
            return
        _stop_evt.clear()
        _stream_thr = threading.Thread(
            target=_run_stream,
            args=(dict(self.cfg),),
            daemon=True,
        )
        _stream_thr.start()
        self.streaming = True
        self._icon.icon  = _make_icon(True)
        self._icon.title = "DeepFace — Streaming"

    def stop(self):
        if not self.streaming:
            return
        _stop_evt.set()
        self.streaming = False
        self._icon.icon  = _make_icon(False)
        self._icon.title = "DeepFace — Idle"

    def quit(self):
        self.stop()
        self._icon.stop()

    def run(self, autostart=False):
        print(f"[tray] Server : {self.cfg['server']}")
        print(f"[tray] Camera : {self.cfg['cam']}")
        if autostart:
            threading.Timer(1.0, self.start).start()
        self._icon.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", help="Override WebSocket server URL")
    parser.add_argument("--cam",    type=int, help="Override camera index")
    parser.add_argument("--start",  action="store_true", help="Auto-start on launch")
    args = parser.parse_args()

    app = TrayApp()

    # CLI overrides (temporary — don't save to config)
    if args.server:
        app.cfg["server"] = args.server
    if args.cam is not None:
        app.cfg["cam"] = args.cam

    app.run(autostart=args.start)


if __name__ == "__main__":
    main()
