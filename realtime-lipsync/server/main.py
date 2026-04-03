"""
Real-time lip sync WebSocket server.

Binary wire protocol (client → server):
  [4B msg_type][4B audio_size][4B video_size][8B timestamp_ms]
  [audio_size bytes PCM int16][video_size bytes JPEG]

Binary wire protocol (server → client):
  [4B msg_type][4B video_size][8B timestamp_ms][4B latency_ms]
  [video_size bytes JPEG]
"""

import asyncio
import logging
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

# Resolve paths relative to this file so it works both in Docker (/app) and bare metal
_SERVER_DIR = Path(__file__).parent.resolve()
_PROJECT_DIR = _SERVER_DIR.parent
_CLIENT_DIR = Path(os.environ.get("CLIENT_DIR", str(_PROJECT_DIR / "client")))

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from frame_buffer import FrameAudioBuffer
from pipeline import LipSyncPipeline

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-time Lip Sync", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(_CLIENT_DIR)), name="static")

# Global pipeline (loaded once at startup)
_pipeline: LipSyncPipeline | None = None
# Thread pool for running blocking inference without blocking asyncio
_executor = ThreadPoolExecutor(max_workers=2)
# Active connections counter
_active_connections: int = 0

MSG_FRAME = b"FRAME"[:4]
MSG_CTRL  = b"CTRL"

# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _pipeline
    logger.info("Loading lip sync pipeline …")
    loop = asyncio.get_event_loop()
    _pipeline = await loop.run_in_executor(
        _executor,
        lambda: LipSyncPipeline(
            model_type=settings.MODEL_TYPE,
            model_dir=settings.MODEL_DIR,
            device=settings.DEVICE,
            half=settings.HALF_PRECISION,
            face_det_interval=settings.FACE_DET_INTERVAL,
            target_face_size=settings.FRAME_WIDTH,
        )
    )
    await loop.run_in_executor(_executor, _pipeline.warmup)
    logger.info("Server ready.")


@app.on_event("shutdown")
async def shutdown():
    _executor.shutdown(wait=False)


# ──────────────────────────────────────────────────────────────────────────────
# Health + info
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.get("/info")
async def info():
    import torch
    return {
        "model": settings.MODEL_TYPE,
        "device": settings.DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "fp16": settings.HALF_PRECISION,
        "target_fps": settings.TARGET_FPS,
        "active_connections": _active_connections,
    }


@app.get("/")
async def index():
    with open(_CLIENT_DIR / "index.html") as f:
        return HTMLResponse(f.read())


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/lipsync")
async def ws_lipsync(ws: WebSocket):
    global _active_connections

    if _active_connections >= settings.MAX_CONNECTIONS:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()
    _active_connections += 1
    client = ws.client
    logger.info("Client connected: %s (total=%d)", client, _active_connections)

    buffer = FrameAudioBuffer(
        target_fps=settings.TARGET_FPS,
        audio_sample_rate=settings.AUDIO_SAMPLE_RATE,
        audio_context_ms=200,
    )

    stats = {"frames_in": 0, "frames_out": 0, "total_latency_ms": 0.0}

    try:
        while True:
            raw = await ws.receive_bytes()
            recv_time_ms = int(time.time() * 1000)

            # ── Parse header ──────────────────────────────────────────────────
            # [4B type][4B audio_sz][4B video_sz][8B ts_ms]
            if len(raw) < 20:
                continue

            audio_sz = struct.unpack_from(">I", raw, 4)[0]
            video_sz = struct.unpack_from(">I", raw, 8)[0]
            ts_ms    = struct.unpack_from(">Q", raw, 12)[0]

            if len(raw) < 20 + audio_sz + video_sz:
                logger.warning("Short message, skipping")
                continue

            audio_bytes = raw[20 : 20 + audio_sz]
            video_bytes = raw[20 + audio_sz : 20 + audio_sz + video_sz]
            stats["frames_in"] += 1

            # ── Decode audio (int16 PCM → float32) ───────────────────────────
            if audio_sz > 0:
                pcm_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                await buffer.add_audio(pcm_float, ts_ms)

            # ── Decode video (JPEG → BGR) ─────────────────────────────────────
            if video_sz > 0:
                jpg_arr = np.frombuffer(video_bytes, dtype=np.uint8)
                frame = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    await buffer.add_video(frame, ts_ms)

            # ── Run inference if we have aligned data ─────────────────────────
            pair = await buffer.get_aligned_pair()
            if pair is None:
                continue

            frame_bgr, audio_pcm = pair

            # Off-load blocking inference to thread pool
            loop = asyncio.get_event_loop()
            processed = await loop.run_in_executor(
                _executor, _pipeline.process, frame_bgr, audio_pcm
            )

            # ── Encode and send ───────────────────────────────────────────────
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
            ok, jpg = cv2.imencode(".jpg", processed, encode_params)
            if not ok:
                continue

            out_bytes = jpg.tobytes()
            latency_ms = int(time.time() * 1000) - recv_time_ms
            stats["frames_out"] += 1
            stats["total_latency_ms"] += latency_ms

            # [4B "FRME"][4B video_sz][8B ts_ms][4B latency_ms][video bytes]
            header = struct.pack(">4sIQI", b"FRME", len(out_bytes), ts_ms, latency_ms)
            await ws.send_bytes(header + out_bytes)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WS error: %s", e)
    finally:
        _active_connections -= 1
        avg_lat = (
            stats["total_latency_ms"] / stats["frames_out"]
            if stats["frames_out"] > 0 else 0
        )
        logger.info(
            "Client disconnected: %s | in=%d out=%d avg_latency=%.0fms",
            client, stats["frames_in"], stats["frames_out"], avg_lat,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        log_level=settings.LOG_LEVEL,
    )
