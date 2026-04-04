"""
Real-time DeepFace pipeline — InsightFace inswapper_128.

DeepFaceLive-style face mask:
  After the swap, use the 106 facial landmarks to build a precise
  face-polygon mask, then alpha-blend the swap result onto the original
  frame.  This gives a natural face-shaped boundary with soft edges —
  no hard rectangular paste, no visible seam around the mouth/chin.

  The mask is feathered with a small kernel (fast) so it adds ~2ms
  while keeping the output at 25fps+.

Face cache:
  If detection drops a frame (side angle, occlusion) the last known
  face is reused so the swap holds instead of snapping to raw webcam.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_DIR   = Path(__file__).parent.parent.resolve()
INSWAPPER_CKPT = _PROJECT_DIR / "models" / "inswapper_128.onnx"
_ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_MAX_CACHE     = 5   # max frames to reuse cached face when detection drops


# ── Face mask from 106 landmarks ─────────────────────────────────────────────

def _face_mask_106(shape, landmark_2d_106: np.ndarray) -> np.ndarray:
    """
    Build a soft face polygon mask from InsightFace 106 2D landmarks.

    The convex hull of all 106 points tightly wraps the face including
    the chin, cheeks, and forehead.  We feather the edges with a small
    GaussianBlur so the blend fades naturally into the neck/hair.

    Returns uint8 mask (0-255), same H×W as `shape`.
    """
    h, w = shape[:2]
    pts = landmark_2d_106.astype(np.int32)

    # Convex hull of all 106 points → precise face boundary
    hull = cv2.convexHull(pts)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Small feather — 31px is fast and gives a smooth edge
    mask = cv2.GaussianBlur(mask, (31, 31), 11)
    return mask


def _blend(swapped: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha-blend swapped frame onto original using the face mask."""
    a = mask[:, :, None].astype(np.float32) / 255.0
    return (swapped.astype(np.float32) * a +
            original.astype(np.float32) * (1.0 - a)).astype(np.uint8)


# ── Pipeline ──────────────────────────────────────────────────────────────────

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

        self._source_face  = None
        self._cached_faces = None
        self._missed       = 0
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
        swapped = frame_bgr.copy()
        for face in live_faces:
            swapped = self.swapper.get(swapped, face, self._source_face, paste_back=True)

        # ── 106-landmark face mask → natural blend ────────────────────────────
        # Use the primary (largest) face's landmarks for the mask.
        # If landmarks aren't available (shouldn't happen with buffalo_l), fall back.
        ref = live_faces[0]
        lm = getattr(ref, "landmark_2d_106", None)
        if lm is not None:
            mask = _face_mask_106(frame_bgr.shape, lm)
            return _blend(swapped, frame_bgr, mask)

        # Fallback: return raw inswapper result
        return swapped
