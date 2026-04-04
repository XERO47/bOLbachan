"""
Real-time DeepFace pipeline — InsightFace inswapper_128.

Kept lean so inswapper runs at full GPU speed (~30ms, 25fps+).

The only extra logic:
  - Face cache: if detection drops a frame, reuse the last known face
    object so the swap holds instead of snapping back to raw webcam.
    This is what makes lips look "connected" — no flicker on missed frames.
  - EMA on bbox only (for the _visual_ display, not the swap alignment).
    The swap itself always uses the real detected kps so lips track exactly.
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

_ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# How many consecutive missed-detection frames to keep using the cached face
# before giving up and showing the raw frame
_MAX_CACHE_FRAMES = 5


class DeepFakePipeline:

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
            raise FileNotFoundError(f"inswapper_128.onnx not found at {INSWAPPER_CKPT}")
        self.swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_CKPT), providers=_ORT_PROVIDERS
        )
        logger.info("Inswapper ready")

        self._source_face  = None   # InsightFace Face object for the avatar
        self._cached_faces = None   # last successfully detected live faces
        self._missed       = 0      # consecutive frames with no detection

        logger.info("DeepFakePipeline ready")

    def set_source(self, image_bgr: np.ndarray) -> str:
        faces = self.face_app.get(image_bgr)
        if not faces:
            return "no_face"
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        self._source_face  = faces[0]
        self._cached_faces = None
        self._missed       = 0
        logger.info("Source face set  bbox=%s", self._source_face.bbox)
        return "ok"

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._source_face is None:
            return frame_bgr

        live_faces = self.face_app.get(frame_bgr)

        if live_faces:
            # Sort largest face first
            live_faces.sort(
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                reverse=True,
            )
            self._cached_faces = live_faces
            self._missed = 0
        else:
            self._missed += 1
            if self._missed > _MAX_CACHE_FRAMES or self._cached_faces is None:
                # Too many misses — show raw frame rather than a stale swap
                return frame_bgr
            # Reuse last known face positions so the swap holds on this frame
            live_faces = self._cached_faces

        result = frame_bgr.copy()
        for face in live_faces:
            result = self.swapper.get(result, face, self._source_face, paste_back=True)
        return result
