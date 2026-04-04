"""
Real-time DeepFace pipeline.

Per-frame:
  1. Detect face with InsightFace
  2. Swap with inswapper_128
  3. (Optional) Enhance with GFPGAN — sharpens and makes lips/expression coherent
  4. Color-correct the swapped region to match live skin tone
  5. Smooth bbox between frames (EMA) to kill jitter

Set env ENHANCE=1 to enable GFPGAN (better quality, +20ms on L40S).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_DIR   = Path(__file__).parent.parent.resolve()
_MODELS_DIR    = _PROJECT_DIR / "models"
INSWAPPER_CKPT = _MODELS_DIR / "inswapper_128.onnx"
GFPGAN_CKPT    = _MODELS_DIR / "GFPGANv1.4.pth"

_ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Exponential moving average weight for bbox smoothing (higher = snappier)
_EMA_ALPHA = 0.35


def _color_correct(swapped: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Histogram-match the swapped face region to the original face region
    so skin tone and lighting blend naturally.

    swapped, original: BGR uint8, same shape
    mask: uint8 single-channel, 0-255 (face region = 255)
    """
    result = swapped.copy()
    m = mask > 128
    if not m.any():
        return result

    for c in range(3):
        src = original[:, :, c][m].astype(np.float32)
        dst = swapped[:, :, c][m].astype(np.float32)
        if src.std() < 1e-3 or dst.std() < 1e-3:
            continue
        # Scale dst mean/std to match src
        corrected = (dst - dst.mean()) / dst.std() * src.std() + src.mean()
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        ch = result[:, :, c].copy()
        ch[m] = corrected
        result[:, :, c] = ch
    return result


def _face_ellipse_mask(shape, bbox) -> np.ndarray:
    """Soft ellipse mask covering the face bbox (no sharp edges)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx = max(1, (x2 - x1) // 2)
    ry = max(1, (y2 - y1) // 2)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    # Feather edges
    k = max(3, min(rx, ry) // 3) | 1
    mask = cv2.GaussianBlur(mask, (k * 4 + 1, k * 4 + 1), k)
    return mask


class DeepFakePipeline:
    """
    Swap every detected face in a live frame with a stored source face.

    env vars:
      ENHANCE=1          enable GFPGAN face enhancer
      COLOR_CORRECT=1    enable histogram color correction (default on)
    """

    def __init__(self):
        import insightface

        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=str(_PROJECT_DIR),
            providers=_ORT_PROVIDERS,
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace FaceAnalysis ready")

        if not INSWAPPER_CKPT.exists():
            raise FileNotFoundError(
                f"inswapper_128.onnx not found at {INSWAPPER_CKPT}"
            )
        self.swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_CKPT), providers=_ORT_PROVIDERS
        )
        logger.info("Inswapper ready")

        # ── GFPGAN enhancer (optional) ────────────────────────────────────────
        self.enhancer = None
        self._use_enhance = os.environ.get("ENHANCE", "0") == "1"
        self._use_color   = os.environ.get("COLOR_CORRECT", "1") == "1"

        if self._use_enhance:
            self._load_gfpgan()

        # ── State ─────────────────────────────────────────────────────────────
        self._source_face = None
        self._smooth_bbox = None   # EMA-smoothed bbox

        logger.info("DeepFakePipeline ready  enhance=%s  color_correct=%s",
                    self._use_enhance, self._use_color)

    def _load_gfpgan(self):
        try:
            from gfpgan import GFPGANer
            if not GFPGAN_CKPT.exists():
                logger.info("Downloading GFPGANv1.4.pth…")
                from huggingface_hub import hf_hub_download
                import shutil
                path = hf_hub_download(
                    repo_id="TencentARC/GFPGAN",
                    filename="GFPGANv1.4.pth",
                    cache_dir="/tmp/hf_cache",
                )
                GFPGAN_CKPT.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, GFPGAN_CKPT)
                logger.info("GFPGANv1.4.pth saved to %s", GFPGAN_CKPT)

            self.enhancer = GFPGANer(
                model_path=str(GFPGAN_CKPT),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            logger.info("GFPGAN enhancer loaded")
        except Exception as e:
            logger.warning("GFPGAN load failed (%s) — running without enhancement", e)
            self.enhancer = None
            self._use_enhance = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_source(self, image_bgr: np.ndarray) -> str:
        faces = self.face_app.get(image_bgr)
        if not faces:
            return "no_face"
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        self._source_face = faces[0]
        self._smooth_bbox = None   # reset smoothing on new source
        logger.info("Source face set  bbox=%s", self._source_face.bbox)
        return "ok"

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._source_face is None:
            return frame_bgr

        live_faces = self.face_app.get(frame_bgr)
        if not live_faces:
            self._smooth_bbox = None
            return frame_bgr

        # ── Swap ──────────────────────────────────────────────────────────────
        result = frame_bgr.copy()
        for face in live_faces:
            result = self.swapper.get(result, face, self._source_face, paste_back=True)

        # ── Color correction ──────────────────────────────────────────────────
        if self._use_color:
            # Use the largest live face for color reference
            live_faces.sort(
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            ref_face = live_faces[0]
            bbox = ref_face.bbox.astype(float)

            # EMA smooth the bbox
            if self._smooth_bbox is None:
                self._smooth_bbox = bbox.copy()
            else:
                self._smooth_bbox = (
                    _EMA_ALPHA * bbox + (1 - _EMA_ALPHA) * self._smooth_bbox
                )

            mask = _face_ellipse_mask(frame_bgr.shape, self._smooth_bbox)
            result = _color_correct(result, frame_bgr, mask)

        # ── GFPGAN face enhancement ───────────────────────────────────────────
        if self._use_enhance and self.enhancer is not None:
            try:
                _, _, result = self.enhancer.enhance(
                    result,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
            except Exception as e:
                logger.debug("GFPGAN enhance error: %s", e)

        return result
