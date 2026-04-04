"""
Real-time DeepFace WebSocket server.

Wire protocol (client → server):  raw JPEG bytes
Wire protocol (server → client):  raw JPEG bytes

REST:
  GET  /                      → index.html (management dashboard)
  GET  /health
  GET  /stream.mjpeg          → MJPEG stream (add directly in OBS as Media Source)
  POST /api/source-face       → multipart image upload, sets swap target
  POST /api/settings          → {"enhance": bool, "color_correct": bool, "mouth_scale": float}
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from deepfake_pipeline import DeepFakePipeline

_SERVER_DIR = Path(__file__).parent.resolve()
_PROJECT_DIR = _SERVER_DIR.parent
_CLIENT_DIR = Path(os.environ.get("CLIENT_DIR", str(_PROJECT_DIR / "client")))

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepFace Live")
app.mount("/static", StaticFiles(directory=str(_CLIENT_DIR)), name="static")

_pipeline: DeepFakePipeline | None = None
_executor = ThreadPoolExecutor(max_workers=2)
_active: int = 0

# Broadcast: last processed frame + subscriber queues (MJPEG + WS display)
_mjpeg_frame: bytes | None = None
_mjpeg_subs: list[asyncio.Queue] = []
_display_subs: list[asyncio.Queue] = []

MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS", "5"))
JPEG_QUALITY    = int(os.environ.get("JPEG_QUALITY", "85"))
TARGET_FPS      = int(os.environ.get("TARGET_FPS", "25"))


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _pipeline
    loop = asyncio.get_event_loop()
    logger.info("Loading DeepFakePipeline…")
    _pipeline = await loop.run_in_executor(_executor, DeepFakePipeline)

    # Auto-load source face if set via env var
    src = os.environ.get("SOURCE_FACE", "")
    if src and Path(src).exists():
        img = cv2.imread(src)
        if img is not None:
            status = await loop.run_in_executor(_executor, _pipeline.set_source, img)
            logger.info("Source face pre-loaded from %s: %s", src, status)

    logger.info("Server ready.")


@app.on_event("shutdown")
async def shutdown():
    _executor.shutdown(wait=False)


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_ready": _pipeline is not None,
        "source_face_set": _pipeline is not None and _pipeline._source_face is not None,
    }


@app.get("/")
async def index():
    with open(_CLIENT_DIR / "index.html") as f:
        return HTMLResponse(f.read())


@app.get("/frame.jpg")
async def latest_frame():
    """Returns the most recent processed frame as a JPEG."""
    if not _mjpeg_frame:
        raise HTTPException(status_code=503, detail="no frame yet")
    return Response(_mjpeg_frame, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/stream")
async def stream_page():
    """
    Full self-contained stream page — captures webcam AND displays deepfake.
    OBS: Add Source → Browser Source → http://<server>:8000/stream
    Width: 640, Height: 480. Allow camera when prompted.
    """
    html = open(_CLIENT_DIR / "stream.html").read()
    return HTMLResponse(html)


@app.get("/obs")
async def obs_page():
    """
    OBS Browser Source — display only, no camera needed.
    Connects to /ws/display and renders processed frames.
    In OBS: Add Source → Browser Source → URL: http://<server>:8000/obs
    Width: 640, Height: 480. No special permissions needed.
    Keep the dashboard tab open with streaming active to push webcam frames.
    """
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* { margin:0; padding:0; }
body { background:#000; width:640px; height:480px; overflow:hidden; }
img { display:block; width:640px; height:480px; }
</style>
</head>
<body>
<img id="f" src="/frame.jpg">
<script>
const f = document.getElementById('f');
setInterval(() => { f.src = '/frame.jpg?t=' + Date.now(); }, 40);
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/stream.mjpeg")
async def mjpeg_stream():
    """
    MJPEG stream of processed deepfake frames.
    In OBS: Add Source → Media Source → check 'Local File' OFF → URL:
      http://<server-ip>:8000/stream.mjpeg
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=4)
    _mjpeg_subs.append(q)

    def _mjpeg_part(data: bytes) -> bytes:
        return (
            b"--jpgboundary\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(data)).encode() + b"\r\n"
            b"\r\n" +
            data + b"\r\n"
        )

    async def generate():
        try:
            if _mjpeg_frame:
                yield _mjpeg_part(_mjpeg_frame)
            while True:
                try:
                    frame_bytes = await asyncio.wait_for(q.get(), timeout=3.0)
                except asyncio.TimeoutError:
                    if _mjpeg_frame:
                        yield _mjpeg_part(_mjpeg_frame)
                    continue
                yield _mjpeg_part(frame_bytes)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _mjpeg_subs.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=jpgboundary",
        headers={"Connection": "keep-alive"},
    )


@app.post("/api/settings")
async def update_settings(body: dict):
    if _pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    if "mouth_scale" in body:
        _pipeline.mouth_scale = max(0.0, min(3.0, float(body["mouth_scale"])))
    return {"mouth_scale": _pipeline.mouth_scale}


@app.post("/api/source-face")
async def set_source_face(file: UploadFile = File(...)):
    """
    Upload a photo — the server detects the face and stores it as the swap target.
    Accepts image/jpeg, image/png, etc.
    """
    if _pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)

    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "cannot decode image"}, status_code=400)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _pipeline.set_source, img)

    if result == "no_face":
        return JSONResponse({"error": "no face detected in uploaded image"}, status_code=422)

    bbox = _pipeline._source_face.bbox.astype(int).tolist()
    logger.info("Source face updated via /api/source-face  bbox=%s", bbox)
    return {"status": "ok", "bbox": bbox}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/deepfake")
async def ws_deepfake(ws: WebSocket):
    global _active

    if _active >= MAX_CONNECTIONS:
        await ws.close(code=1008)
        return

    await ws.accept()
    _active += 1
    logger.info("Client connected  total=%d", _active)

    frames_in = frames_out = 0
    total_lat = 0.0
    running = False

    try:
        while True:
            jpeg_bytes = await ws.receive_bytes()
            recv_ms = time.time() * 1000

            # Decode JPEG
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frames_in += 1

            # Skip if previous inference still in flight
            if running:
                continue

            running = True
            loop = asyncio.get_event_loop()
            try:
                processed = await loop.run_in_executor(
                    _executor, _pipeline.process, frame
                )
            except Exception as e:
                logger.exception("Inference error: %s", e)
                running = False
                continue
            finally:
                running = False

            # Encode and send
            ok, jpg = cv2.imencode(
                ".jpg", processed,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                continue

            frame_bytes = jpg.tobytes()

            # Broadcast to all passive subscribers (MJPEG + WS display)
            global _mjpeg_frame
            _mjpeg_frame = frame_bytes
            for sub_q in list(_mjpeg_subs) + list(_display_subs):
                if not sub_q.full():
                    sub_q.put_nowait(frame_bytes)

            await ws.send_bytes(frame_bytes)
            frames_out += 1
            total_lat += time.time() * 1000 - recv_ms

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WS error: %s", e)
    finally:
        _active -= 1
        avg = total_lat / frames_out if frames_out else 0
        logger.info("Disconnected  in=%d out=%d avg_lat=%.0fms", frames_in, frames_out, avg)


# ── Display WebSocket (passive — no camera, just receives frames) ──────────────

@app.websocket("/ws/display")
async def ws_display(ws: WebSocket):
    """
    Passive subscriber: receives every processed frame as raw JPEG bytes.
    Used by the /obs Browser Source page — no camera permission needed.
    """
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=4)
    _display_subs.append(q)
    logger.info("Display subscriber connected  total=%d", len(_display_subs))
    try:
        while True:
            frame_bytes = await asyncio.wait_for(q.get(), timeout=10.0)
            await ws.send_bytes(frame_bytes)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as e:
        logger.warning("Display WS error: %s", e)
    finally:
        try:
            _display_subs.remove(q)
        except ValueError:
            pass
        logger.info("Display subscriber disconnected  total=%d", len(_display_subs))


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        ws_ping_interval=20,
        ws_ping_timeout=20,
        log_level=LOG_LEVEL,
    )
