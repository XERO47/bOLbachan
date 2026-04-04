"""
Real-time DeepFace pipeline — InsightFace inswapper only.
No audio, no lip sync. Just live face → source face swap.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_DIR   = Path(__file__).parent.parent.resolve()
_MODELS_DIR    = _PROJECT_DIR / "models"
INSWAPPER_CKPT = _MODELS_DIR / "inswapper_128.onnx"

_ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class DeepFakePipeline:
    """
    Swap every detected face in a live frame with a stored source face.

    Usage:
        pipe = DeepFakePipeline()
        pipe.set_source(image_bgr)   # call once (or whenever user changes avatar)
        out  = pipe.process(frame)   # call per webcam frame
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
                f"inswapper_128.onnx not found at {INSWAPPER_CKPT}\n"
                "Run scripts/setup_avatar.sh to download it."
            )
        self.swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_CKPT), providers=_ORT_PROVIDERS
        )
        logger.info("Inswapper ready")

        self._source_face = None   # InsightFace Face object
        logger.info("DeepFakePipeline ready (no source face set yet)")

    def set_source(self, image_bgr: np.ndarray) -> str:
        """
        Set the source (avatar) face from a BGR image.
        Returns a status string.
        """
        faces = self.face_app.get(image_bgr)
        if not faces:
            return "no_face"
        # Pick the largest face by bounding box area
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
        self._source_face = faces[0]
        logger.info("Source face set  embedding=%s  bbox=%s",
                    self._source_face.embedding.shape, self._source_face.bbox)
        return "ok"

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Swap all detected faces in frame with the source face.
        Returns frame unchanged if no source face is set or no face detected.
        """
        if self._source_face is None:
            return frame_bgr

        live_faces = self.face_app.get(frame_bgr)
        if not live_faces:
            return frame_bgr

        result = frame_bgr.copy()
        for face in live_faces:
            result = self.swapper.get(result, face, self._source_face, paste_back=True)
        return result
