"""
Real-time DeepFace WebSocket server.

Wire protocol (client → server):  raw JPEG bytes
Wire protocol (server → client):  raw JPEG bytes

REST:
  GET  /                      → index.html
  GET  /health
  POST /api/source-face       → multipart image upload, sets swap target
  POST /api/settings          → {"enhance": bool, "color_correct": bool}
"""

import asyncio
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
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


@app.post("/api/settings")
async def update_settings(body: dict):
    """Toggle enhance / color_correct at runtime without restarting."""
    if _pipeline is None:
        return JSONResponse({"error": "pipeline not ready"}, status_code=503)
    if "enhance" in body:
        val = bool(body["enhance"])
        if val and not _pipeline._use_enhance:
            # Load GFPGAN on demand
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_executor, _pipeline._load_gfpgan)
        _pipeline._use_enhance = val and _pipeline.enhancer is not None
    if "color_correct" in body:
        _pipeline._use_color = bool(body["color_correct"])
    if "mouth_scale" in body:
        _pipeline.mouth_scale = max(0.0, min(3.0, float(body["mouth_scale"])))
    logger.info("Settings updated  enhance=%s  color=%s  mouth_scale=%.2f",
                _pipeline._use_enhance, _pipeline._use_color, _pipeline.mouth_scale)
    return {
        "enhance": _pipeline._use_enhance,
        "color_correct": _pipeline._use_color,
        "mouth_scale": _pipeline.mouth_scale,
    }


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

            await ws.send_bytes(jpg.tobytes())
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
