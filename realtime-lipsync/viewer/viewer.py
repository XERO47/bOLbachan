"""
DeepFace Live — Windows Viewer
===============================
Captures your webcam, streams frames to the GPU server, shows the deepfake
in a native OpenCV window.  OBS captures that window via Window Capture.

Optional --vcam flag: also pushes frames to an OBS Virtual Camera device
(requires pyvirtualcam + OBS with Virtual Camera plugin installed).

Usage:
    python viewer.py
    python viewer.py --server ws://205.147.101.238:8000/ws/deepfake
    python viewer.py --server ws://205.147.101.238:8000/ws/deepfake --vcam
    python viewer.py --cam 1 --fps 30

Press Q or Esc to quit.
"""

import argparse
import asyncio
import queue
import threading
import time

import cv2
import numpy as np

try:
    import websockets
except ImportError:
    raise SystemExit("Missing dependency: pip install websockets")

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_SERVER = "ws://205.147.101.238:8000/ws/deepfake"
DEFAULT_CAM    = 0
DEFAULT_FPS    = 25
DEFAULT_W      = 640
DEFAULT_H      = 480
JPEG_QUALITY   = 85
WINDOW_TITLE   = "DeepFace Live"

# ── Async send/receive loop ────────────────────────────────────────────────────

async def ws_loop(server_url: str, cam_idx: int, fps: int,
                  rx_q: queue.Queue, stop_evt: threading.Event):
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DEFAULT_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_H)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"[viewer] ERROR: cannot open camera {cam_idx}")
        stop_evt.set()
        return

    interval = 1.0 / fps
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    print(f"[viewer] Connecting to {server_url} …")
    try:
        async with websockets.connect(
            server_url,
            max_size=20_000_000,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print("[viewer] Connected.")

            async def recv_loop():
                async for msg in ws:
                    arr = np.frombuffer(msg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        if rx_q.full():
                            try: rx_q.get_nowait()
                            except queue.Empty: pass
                        rx_q.put_nowait(frame)

            recv_task = asyncio.create_task(recv_loop())

            try:
                while not stop_evt.is_set():
                    t0 = time.monotonic()
                    ret, frame = cap.read()
                    if not ret:
                        print("[viewer] Camera read failed.")
                        break

                    ok, jpg = cv2.imencode(".jpg", frame, encode_params)
                    if ok:
                        await ws.send(jpg.tobytes())

                    elapsed = time.monotonic() - t0
                    await asyncio.sleep(max(0.001, interval - elapsed))
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
    except Exception as e:
        print(f"[viewer] WebSocket error: {e}")
    finally:
        cap.release()
        stop_evt.set()


def run_ws(server_url, cam_idx, fps, rx_q, stop_evt):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(ws_loop(server_url, cam_idx, fps, rx_q, stop_evt))
    finally:
        loop.close()


# ── Main display loop ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepFace Live — Windows Viewer")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"WebSocket URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--cam",    type=int, default=DEFAULT_CAM,
                        help="Camera index (default: 0)")
    parser.add_argument("--fps",    type=int, default=DEFAULT_FPS,
                        help="Target FPS (default: 25)")
    parser.add_argument("--vcam",      action="store_true",
                        help="Push to OBS Virtual Camera via pyvirtualcam")
    parser.add_argument("--no-window", action="store_true",
                        help="Headless mode — no OpenCV window (use with OBS /stream.mjpeg)")
    args = parser.parse_args()

    # Virtual camera setup
    vcam = None
    if args.vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(
                width=DEFAULT_W, height=DEFAULT_H, fps=args.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            print(f"[viewer] Virtual camera: {vcam.device}")
        except Exception as e:
            print(f"[viewer] pyvirtualcam unavailable ({e}) — window-only mode")

    # Start background WS thread
    rx_q     = queue.Queue(maxsize=3)
    stop_evt = threading.Event()
    ws_thread = threading.Thread(
        target=run_ws,
        args=(args.server, args.cam, args.fps, rx_q, stop_evt),
        daemon=True,
    )
    ws_thread.start()

    # OpenCV window
    no_window = args.no_window
    if not no_window:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 960, 720)
        print(f"[viewer] Window '{WINDOW_TITLE}' open. Press Q or Esc to quit.")
    else:
        print("[viewer] Headless mode — no window. OBS: Media Source → http://<server>:8000/stream.mjpeg")
        print("[viewer] Press Ctrl+C to quit.")

    fps_t   = time.time()
    fps_n   = 0
    last_frame = None

    while not stop_evt.is_set():
        try:
            frame = rx_q.get(timeout=0.05)
            last_frame = frame
        except queue.Empty:
            frame = last_frame  # hold last frame

        if frame is not None:
            fps_n += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps_actual = fps_n / (now - fps_t)
                if not no_window:
                    cv2.setWindowTitle(WINDOW_TITLE, f"{WINDOW_TITLE}  {fps_actual:.0f} fps  [Q=quit]")
                else:
                    print(f"\r[viewer] {fps_actual:.0f} fps", end="", flush=True)
                fps_n = 0
                fps_t = now

            if not no_window:
                cv2.imshow(WINDOW_TITLE, frame)

            if vcam:
                f = cv2.resize(frame, (DEFAULT_W, DEFAULT_H))
                vcam.send(f)
                vcam.sleep_until_next_frame()

        if no_window:
            time.sleep(0.001)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or Esc
            break

    stop_evt.set()
    cv2.destroyAllWindows()
    if vcam:
        vcam.close()
    print("[viewer] Stopped.")


if __name__ == "__main__":
    main()
