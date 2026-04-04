"""
Real-time DeepFace pipeline.

Per-frame processing:
  1. EMA-smoothed face detection (kills jitter between frames)
  2. InsightFace inswapper_128 face swap
  3. Landmark-guided soft mask → alpha blend at face boundary
     (makes the swap edge invisible — no "pasted face" look, ~2ms)
  4. Color correction — histogram-match swap to live skin tone
  5. (Optional) Sharpening on the face crop to counteract swap softness

All steps run on GPU / CPU via OpenCV + InsightFace.
No heavy models needed for enhancement — total budget <45ms at 25fps.

Env:
  COLOR_CORRECT=0   disable color correction
  SHARPEN=0         disable face sharpening
  ENHANCE=1         enable GFPGAN (~80ms, ~12fps, best quality)
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

# EMA weight for bbox smoothing: higher = snappier tracking, lower = smoother
_EMA = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kps_mask(shape, kps: np.ndarray, bbox) -> np.ndarray:
    """
    Build a soft face mask from InsightFace 5-point kps (eyes, nose, mouth corners).
    The mask is an ellipse fitted to the face bounding box, feathered at edges.
    kps shape: (5,2) — [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Expand box slightly to cover chin and forehead
    expand_y = int((y2 - y1) * 0.10)
    y2 = min(h, y2 + expand_y)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx = max(1, (x2 - x1) // 2)
    ry = max(1, (y2 - y1) // 2)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    # Feather: big kernel for smooth fade
    k = max(5, min(rx, ry) // 2) | 1
    mask = cv2.GaussianBlur(mask, (k * 6 + 1, k * 6 + 1), k)
    return mask


def _color_correct(swapped: np.ndarray, reference: np.ndarray,
                   mask: np.ndarray) -> np.ndarray:
    """
    Scale mean+std of each channel in the swapped face to match the live face.
    Applied only within the mask region.
    """
    result = swapped.copy().astype(np.float32)
    m = mask > 64
    if not m.any():
        return swapped

    for c in range(3):
        src = reference[:, :, c][m].astype(np.float32)
        dst = swapped[:, :, c][m].astype(np.float32)
        if src.std() < 1 or dst.std() < 1:
            continue
        corrected = (dst - dst.mean()) / dst.std() * src.std() + src.mean()
        result[:, :, c][m] = corrected

    return np.clip(result, 0, 255).astype(np.uint8)


def _sharpen_region(img: np.ndarray, bbox, strength: float = 0.5) -> np.ndarray:
    """
    Unsharp-mask on the face region only.
    Counteracts the slight softening that inswapper introduces.
    strength: 0=none, 1=strong
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return img

    region = img[y1:y2, x1:x2].astype(np.float32)
    blurred = cv2.GaussianBlur(region, (0, 0), 2)
    sharpened = region + strength * (region - blurred)
    result = img.copy()
    result[y1:y2, x1:x2] = np.clip(sharpened, 0, 255).astype(np.uint8)
    return result


def _soft_blend(swapped: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Alpha-blend swapped frame over the original using the feathered face mask.
    Brings the swap boundary to zero at face edges — no hard "paste" line.
    ~2ms on CPU vs ~60ms for Poisson clone.
    """
    alpha = mask[:, :, None].astype(np.float32) / 255.0
    return (swapped.astype(np.float32) * alpha +
            original.astype(np.float32) * (1 - alpha)).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DeepFakePipeline:

    def __init__(self):
        import insightface

        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=str(_PROJECT_DIR),
            providers=_ORT_PROVIDERS,
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self._full_app = self.face_app   # same app used for both source and live
        logger.info("InsightFace FaceAnalysis ready")

        if not INSWAPPER_CKPT.exists():
            raise FileNotFoundError(f"inswapper_128.onnx not found at {INSWAPPER_CKPT}")
        self.swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_CKPT), providers=_ORT_PROVIDERS
        )
        logger.info("Inswapper ready")

        self._use_color   = os.environ.get("COLOR_CORRECT", "1") == "1"
        self._use_sharpen = os.environ.get("SHARPEN", "1") == "1"
        self._use_enhance = False
        self.enhancer     = None

        if os.environ.get("ENHANCE", "0") == "1":
            self._load_gfpgan()

        self._source_face = None
        self._smooth_bbox = None
        self._smooth_kps  = None

        logger.info("DeepFakePipeline ready  color=%s  sharpen=%s  enhance=%s",
                    self._use_color, self._use_sharpen, self._use_enhance)

    def _load_gfpgan(self):
        try:
            from gfpgan import GFPGANer
            if not GFPGAN_CKPT.exists():
                import shutil
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id="TencentARC/GFPGAN",
                                       filename="GFPGANv1.4.pth",
                                       cache_dir="/tmp/hf_cache")
                shutil.copy(path, GFPGAN_CKPT)

            self.enhancer = GFPGANer(
                model_path=str(GFPGAN_CKPT),
                upscale=1, arch="clean",
                channel_multiplier=2, bg_upsampler=None,
            )
            self._use_enhance = True
            logger.info("GFPGAN enhancer loaded (~80ms/frame → ~12fps)")
        except Exception as e:
            logger.warning("GFPGAN unavailable: %s", e)
            self._use_enhance = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_source(self, image_bgr: np.ndarray) -> str:
        # Use full analyzer for source — we need the embedding here
        faces = self._full_app.get(image_bgr)
        if not faces:
            return "no_face"
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        self._source_face = faces[0]
        self._smooth_bbox = None
        self._smooth_kps  = None
        logger.info("Source face set  bbox=%s", self._source_face.bbox)
        return "ok"

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._source_face is None:
            return frame_bgr

        live_faces = self.face_app.get(frame_bgr)
        if not live_faces:
            self._smooth_bbox = None
            return frame_bgr

        # Sort by size — use the dominant (largest) face
        live_faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        ref_face = live_faces[0]

        # ── EMA smooth bbox + kps ────────────────────────────────────────────
        bbox = ref_face.bbox.astype(float)
        kps  = ref_face.kps.astype(float)
        if self._smooth_bbox is None:
            self._smooth_bbox = bbox.copy()
            self._smooth_kps  = kps.copy()
        else:
            self._smooth_bbox = _EMA * bbox + (1 - _EMA) * self._smooth_bbox
            self._smooth_kps  = _EMA * kps  + (1 - _EMA) * self._smooth_kps

        # ── Swap all faces ───────────────────────────────────────────────────
        swapped = frame_bgr.copy()
        for face in live_faces:
            swapped = self.swapper.get(swapped, face, self._source_face, paste_back=True)

        # ── Build face mask from smoothed bbox + kps ─────────────────────────
        mask = _kps_mask(frame_bgr.shape, self._smooth_kps, self._smooth_bbox)

        # ── Color correction ─────────────────────────────────────────────────
        if self._use_color:
            swapped = _color_correct(swapped, frame_bgr, mask)

        # ── Soft blend: fade swap to original at face boundary ───────────────
        result = _soft_blend(swapped, frame_bgr, mask)

        # ── Sharpening on face crop ──────────────────────────────────────────
        if self._use_sharpen:
            result = _sharpen_region(result, self._smooth_bbox, strength=0.4)

        # ── Optional GFPGAN (reduces fps to ~12 but looks best) ──────────────
        if self._use_enhance and self.enhancer is not None:
            try:
                _, _, result = self.enhancer.enhance(
                    result, has_aligned=False, only_center_face=False, paste_back=True
                )
            except Exception as e:
                logger.debug("GFPGAN error: %s", e)

        return result
