"""
Real-time DeepFace pipeline — InsightFace inswapper_128.

DeepFaceLive-style masking (two-layer):

  Layer 1 — Face mask (106 landmarks convex hull):
    After the swap, blend the result onto the original using a precise
    face-polygon mask.  Natural boundary at chin/jaw/cheeks/forehead.

  Layer 2 — Mouth cutout (real mouth shows through):
    Cut the mouth region out of the swap and fill it with the ORIGINAL
    live frame.  Your real lips are always visible — expression and
    lip movement come directly from the webcam, not the swap model.
    Exactly how DeepFaceLive's mouth mask works.

Face cache:
  If detection drops a frame the last known face is reused so the
  swap holds instead of snapping back to raw webcam.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_DIR   = Path(__file__).parent.parent.resolve()
INSWAPPER_CKPT = _PROJECT_DIR / "models" / "inswapper_128.onnx"
_MAX_CACHE     = 5

_ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ── Masks ─────────────────────────────────────────────────────────────────────

def _face_mask_106(shape, lm106: np.ndarray) -> np.ndarray:
    """Convex hull of 106 landmarks → soft face boundary mask."""
    h, w = shape[:2]
    hull = cv2.convexHull(lm106.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.GaussianBlur(mask, (31, 31), 11)


def _mouth_mask(shape, kps: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Soft ellipse over the mouth + chin region, derived from the 5-point kps:
      kps[2] = nose tip  (top boundary of mouth region)
      kps[3] = left mouth corner
      kps[4] = right mouth corner

    scale: multiplier for the ellipse radii (0 = no cutout, 1 = default, 2 = large)
    Returns mask where 255 = show original live mouth, 0 = show swap.
    """
    h, w = shape[:2]

    nose       = kps[2]
    lm_corner  = kps[3]
    rm_corner  = kps[4]

    # Centre of mouth
    cx = int((lm_corner[0] + rm_corner[0]) / 2)
    cy = int((lm_corner[1] + rm_corner[1]) / 2)

    # Horizontal radius: slightly wider than mouth corner span
    rx = int(np.linalg.norm(rm_corner - lm_corner) * 0.75 * scale)

    # Vertical radius: from above mouth corners down to estimated chin
    # chin ≈ mouth_y + (mouth_y − nose_y) * 0.9
    chin_y  = cy + int((cy - nose[1]) * 0.9)
    ry = max(rx // 2, int((chin_y - cy) * scale))

    if rx <= 0 or ry <= 0:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (31, 31), 11)


def _blend(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """src * mask + dst * (1-mask)"""
    a = mask[:, :, None].astype(np.float32) / 255.0
    return (src.astype(np.float32) * a +
            dst.astype(np.float32) * (1.0 - a)).astype(np.uint8)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class DeepFakePipeline:

    def __init__(self):
        import insightface

        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=str(_PROJECT_DIR),
            providers=_ORT_PROVIDERS,
            allowed_modules=["detection", "landmark_2d_106", "recognition"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("InsightFace FaceAnalysis ready")

        if not INSWAPPER_CKPT.exists():
            raise FileNotFoundError(f"inswapper_128.onnx not found at {INSWAPPER_CKPT}")
        self.swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_CKPT), providers=_ORT_PROVIDERS
        )
        logger.info("Inswapper ready")

        self._source_face  = None
        self._cached_faces = None
        self._missed       = 0
        self.mouth_scale   = 0.15  # tunable: 0=no cutout, 0.15=default, 2=large

        self._warmup()
        logger.info("DeepFakePipeline ready")

    def _warmup(self):
        """Warm up the full pipeline (detector + inswapper) using avatar against itself."""
        import time, os
        t0 = time.time()
        src = os.environ.get("SOURCE_FACE", "")
        avatar = None
        if src:
            avatar = cv2.imread(src)
        if avatar is None:
            # Make a synthetic face-like image (skin tone rectangle)
            avatar = np.full((480, 640, 3), (120, 100, 80), dtype=np.uint8)

        logger.info("Warming up CUDA kernels (full pipeline)…")
        faces = self.face_app.get(avatar)
        if faces:
            faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            src_face = faces[0]
            self._source_face = src_face          # pre-set so process() is live immediately
            self._warmup_inswapper(avatar, src_face)
            logger.info("Full warmup done in %.1fs", time.time() - t0)
        else:
            logger.info("Warmup: no face in avatar image — inswapper will warm up on first set_source call")

    def _warmup_inswapper(self, image_bgr: np.ndarray, face) -> None:
        """Run inswapper to compile TRT engine (first call) or warm CUDA (subsequent)."""
        import time
        t0 = time.time()
        logger.info("Running inswapper warmup (may take several minutes on first TRT compile)…")
        for _ in range(5):
            try:
                self.swapper.get(image_bgr.copy(), face, face, paste_back=True)
            except Exception:
                pass
        logger.info("Inswapper warmup done in %.1fs", time.time() - t0)

    def set_source(self, image_bgr: np.ndarray) -> str:
        faces = self.face_app.get(image_bgr)
        if not faces:
            return "no_face"
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        self._source_face  = faces[0]
        self._cached_faces = None
        self._missed       = 0
        logger.info("Source face set  bbox=%s", self._source_face.bbox)

        # Compile CUDA kernels using the just-uploaded face (first call will be slow
        # on a fresh instance; this warmup prevents that hitting live frames).
        logger.info("Warming up inswapper with source face…")
        self._warmup_inswapper(image_bgr, self._source_face)

        return "ok"

    _perf_n = 0
    _perf_det = 0.0
    _perf_swap = 0.0
    _perf_mask = 0.0
    _PERF_EVERY = 100   # log timing every N frames

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        import time as _time
        if self._source_face is None:
            return frame_bgr

        t_det = _time.perf_counter()
        live_faces = self.face_app.get(frame_bgr)
        self._perf_det += _time.perf_counter() - t_det

        if live_faces:
            live_faces.sort(
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                reverse=True,
            )
            self._cached_faces = live_faces
            self._missed = 0
        else:
            self._missed += 1
            if self._missed > _MAX_CACHE or self._cached_faces is None:
                return frame_bgr
            live_faces = self._cached_faces

        # ── Inswapper ────────────────────────────────────────────────────────
        t_swap = _time.perf_counter()
        swapped = frame_bgr.copy()
        for face in live_faces:
            swapped = self.swapper.get(swapped, face, self._source_face, paste_back=True)
        self._perf_swap += _time.perf_counter() - t_swap

        # ── 106-landmark face mask → natural blend ────────────────────────────
        t_mask = _time.perf_counter()
        # Use the primary (largest) face's landmarks for the mask.
        # If landmarks aren't available (shouldn't happen with buffalo_l), fall back.
        ref = live_faces[0]
        lm = getattr(ref, "landmark_2d_106", None)
        if lm is not None:
            # Layer 1: blend swap onto original using face boundary mask
            face_mask = _face_mask_106(frame_bgr.shape, lm)
            result = _blend(swapped, frame_bgr, face_mask)

            # Layer 2: cut mouth hole — show real live lips on top
            kps = getattr(ref, "kps", None)
            if kps is not None and len(kps) >= 5 and self.mouth_scale > 0:
                mm = _mouth_mask(frame_bgr.shape, kps, self.mouth_scale)
                result = _blend(frame_bgr, result, mm)

            self._perf_mask += _time.perf_counter() - t_mask
            self._perf_n += 1
            if self._perf_n % self._PERF_EVERY == 0:
                n = self._PERF_EVERY
                logger.info(
                    "PERF[%d] det=%.1fms swap=%.1fms mask=%.1fms total=%.1fms",
                    self._perf_n,
                    self._perf_det / n * 1000,
                    self._perf_swap / n * 1000,
                    self._perf_mask / n * 1000,
                    (self._perf_det + self._perf_swap + self._perf_mask) / n * 1000,
                )
                self._perf_det = self._perf_swap = self._perf_mask = 0.0
            return result

        self._perf_mask += _time.perf_counter() - t_mask
        # Fallback: return raw inswapper result
        return swapped
