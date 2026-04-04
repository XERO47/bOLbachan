"""
DeepFaceLive-style virtual avatar pipeline.

Flow per frame:
  1. Detect face in live webcam frame
  2. Swap live face with reference avatar using InsightFace inswapper
  3. Lip-sync the swapped face with live audio via Wav2Lip
  4. Paste composited result back onto frame

Models:
  - InsightFace buffalo_l (face analysis + inswapper_128)
  - Wav2Lip (wav2lip_gan.pth)

All models are cached in <project_root>/models/
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_SERVER_DIR  = Path(__file__).parent.resolve()
_PROJECT_DIR = _SERVER_DIR.parent
_MODELS_DIR  = _PROJECT_DIR / "models"
_WAV2LIP_DIR = _PROJECT_DIR / "Wav2Lip"

WAV2LIP_CKPT      = _MODELS_DIR / "wav2lip_gan.pth"
INSWAPPER_CKPT    = _MODELS_DIR / "inswapper_128.onnx"
INSIGHTFACE_DIR   = _MODELS_DIR / "buffalo_l"

FACE_SIZE    = 96   # Wav2Lip input size
SAMPLE_RATE  = 16000
MEL_STEP     = 16
HOP_LENGTH   = 200
MEL_WINDOW   = MEL_STEP * HOP_LENGTH   # samples per mel chunk


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_wav2lip(ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(_WAV2LIP_DIR))
    from models import Wav2Lip
    model = Wav2Lip()
    ckpt = torch.load(str(ckpt_path), map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    logger.info("Wav2Lip loaded from %s", ckpt_path)
    return model


def _build_face_app():
    """Build InsightFace FaceAnalysis app (buffalo_l)."""
    import insightface
    app = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(_MODELS_DIR),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace FaceAnalysis ready")
    return app


def _load_inswapper(app):
    """Load inswapper_128 ONNX model."""
    import insightface
    swapper = insightface.model_zoo.get_model(
        str(INSWAPPER_CKPT),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    logger.info("Inswapper loaded from %s", INSWAPPER_CKPT)
    return swapper


def _mel_from_pcm(pcm: np.ndarray) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(
        y=pcm, sr=SAMPLE_RATE, n_mels=80,
        hop_length=HOP_LENGTH, win_length=800, fmax=8000,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)


def _prepare_face_tensor(face_bgr: np.ndarray):
    """
    face_bgr: (96,96,3) uint8 — the cropped/resized avatar face region.
    Returns (6,96,96) float32 tensor suitable for Wav2Lip.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB) / 255.0
    masked = face_rgb.copy()
    masked[FACE_SIZE // 2:] = 0           # mask lower half
    img_t = np.concatenate([
        np.transpose(masked, (2, 0, 1)),
        np.transpose(face_rgb, (2, 0, 1)),
    ], axis=0)
    return img_t.astype(np.float32)


@torch.inference_mode()
def _wav2lip_infer(model, face_tensor: np.ndarray, mel_chunk: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """Returns (96,96,3) BGR uint8."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    img_t = torch.from_numpy(face_tensor).unsqueeze(0).to(device, dtype=dtype)
    mel_t = torch.from_numpy(mel_chunk).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    pred  = model(mel_t, img_t)
    pred  = pred.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    pred  = (pred * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)


def _paste_back(canvas: np.ndarray, insert: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return canvas
    resized = cv2.resize(insert, (w, h))
    out = canvas.copy()
    # Feathered blend
    mask = np.ones((h, w), dtype=np.float32)
    k = max(3, min(h, w) // 6) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, None]
    out[y1:y2, x1:x2] = (
        out[y1:y2, x1:x2].astype(np.float32) * (1 - mask) +
        resized.astype(np.float32) * mask
    ).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AvatarPipeline:
    """
    Virtual avatar pipeline:
      process(frame_bgr, audio_pcm) → output_bgr
    """

    def __init__(
        self,
        avatar_image_path: str,
        device_str: str = "cuda",
    ):
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        logger.info("AvatarPipeline device: %s", self.device)

        # ── Face analysis (InsightFace) ──────────────────────────────────────
        self.face_app  = _build_face_app()
        self.swapper   = _load_inswapper(self.face_app)

        # ── Wav2Lip ──────────────────────────────────────────────────────────
        if not WAV2LIP_CKPT.exists():
            raise FileNotFoundError(
                f"Wav2Lip checkpoint not found at {WAV2LIP_CKPT}\n"
                "Download wav2lip_gan.pth from https://github.com/Rudrabha/Wav2Lip"
            )
        self.wav2lip = _load_wav2lip(WAV2LIP_CKPT, self.device)

        # ── Reference avatar face ─────────────────────────────────────────────
        avatar_img = cv2.imread(avatar_image_path)
        if avatar_img is None:
            raise FileNotFoundError(f"Avatar image not found: {avatar_image_path}")
        faces = self.face_app.get(avatar_img)
        if not faces:
            raise RuntimeError(f"No face detected in avatar image: {avatar_image_path}")
        self.avatar_face = faces[0]       # InsightFace Face object
        self.avatar_img  = avatar_img     # keep reference for Wav2Lip crop
        logger.info("Avatar face loaded from %s  embedding=%s",
                    avatar_image_path, self.avatar_face.embedding.shape)

        # ── Wav2Lip crop from avatar ─────────────────────────────────────────
        bbox = self.avatar_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        # Margin 20%
        mx = int((x2 - x1) * 0.2); my = int((y2 - y1) * 0.2)
        h, w = avatar_img.shape[:2]
        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
        self._avatar_bbox = (x1, y1, x2, y2)
        face_crop = avatar_img[y1:y2, x1:x2]
        face_96   = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
        self._avatar_face_tensor = _prepare_face_tensor(face_96)

        # ── Audio rolling buffer ─────────────────────────────────────────────
        self._mel_buffer = np.zeros(MEL_WINDOW * 2, dtype=np.float32)

        logger.info("AvatarPipeline ready.")

    def warmup(self):
        """Run one dummy inference to pre-load CUDA kernels."""
        dummy_face  = np.zeros((6, FACE_SIZE, FACE_SIZE), dtype=np.float32)
        dummy_mel   = np.zeros((80, MEL_STEP), dtype=np.float32)
        _wav2lip_infer(self.wav2lip, dummy_face, dummy_mel, self.device)
        logger.info("Wav2Lip warmup done.")

    def process(self, frame_bgr: np.ndarray, audio_pcm: np.ndarray) -> np.ndarray:
        """
        Main entry point called from the WebSocket handler (in executor thread).

        frame_bgr  : live webcam frame (H×W×3 BGR uint8)
        audio_pcm  : float32 PCM [-1,1], any length
        returns    : processed frame (same size, BGR uint8)
        """
        t0 = time.perf_counter()

        # ── 1. Detect live face ──────────────────────────────────────────────
        live_faces = self.face_app.get(frame_bgr)
        if not live_faces:
            # No face detected — return passthrough
            return frame_bgr

        # ── 2. Face swap: live → avatar ──────────────────────────────────────
        result = frame_bgr.copy()
        for face in live_faces:
            result = self.swapper.get(result, face, self.avatar_face, paste_back=True)

        # ── 3. Update audio mel buffer ───────────────────────────────────────
        if len(audio_pcm) > 0:
            self._mel_buffer = np.roll(self._mel_buffer, -len(audio_pcm))
            self._mel_buffer[-len(audio_pcm):] = audio_pcm[-len(self._mel_buffer):]

        audio_window = self._mel_buffer[-MEL_WINDOW:]

        # ── 4. Wav2Lip on swapped avatar face region ─────────────────────────
        mel = _mel_from_pcm(audio_window)
        if mel.shape[1] >= MEL_STEP:
            mel_chunk = mel[:, -MEL_STEP:]

            # Use avatar face region on the *swapped* result for Wav2Lip
            x1, y1, x2, y2 = self._avatar_bbox
            # Clamp to actual result dims
            rh, rw = result.shape[:2]
            x1c = max(0, min(x1, rw - 1))
            y1c = max(0, min(y1, rh - 1))
            x2c = max(0, min(x2, rw))
            y2c = max(0, min(y2, rh))

            if x2c > x1c and y2c > y1c:
                face_region = result[y1c:y2c, x1c:x2c]
                face_96 = cv2.resize(face_region, (FACE_SIZE, FACE_SIZE))
                face_tensor = _prepare_face_tensor(face_96)

                lip_synced = _wav2lip_infer(self.wav2lip, face_tensor, mel_chunk, self.device)
                result = _paste_back(result, lip_synced, (x1c, y1c, x2c, y2c))

        dt_ms = (time.perf_counter() - t0) * 1000
        logger.debug("AvatarPipeline.process %.0fms", dt_ms)
        return result
