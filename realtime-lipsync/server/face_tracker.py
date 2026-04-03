"""
Fast face detection and ROI tracking.

Strategy:
- Run full MediaPipe face mesh detection every DETECT_INTERVAL frames.
- Between detections, track the bounding box with a lightweight optical
  flow tracker (much faster than running the full detector every frame).
- Returns a normalized face crop + the paste-back affine transform.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import mediapipe as mp


# Lip landmark indices from MediaPipe face mesh (468-point model)
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
]


class FaceTracker:
    def __init__(self, detect_interval: int = 5, target_size: int = 256):
        self.detect_interval = detect_interval
        self.target_size = target_size
        self._frame_count = 0
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None  # x1,y1,x2,y2
        self._prev_gray: Optional[np.ndarray] = None
        self._track_pts: Optional[np.ndarray] = None

        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_face_crop(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple]]:
        """
        Returns:
          - face_crop: (target_size x target_size) BGR
          - M_inv: affine matrix to paste crop back into frame
          - bbox: (x1, y1, x2, y2) in original frame coords
        Returns None if no face found.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        do_detect = (
            self._frame_count % self.detect_interval == 0
            or self._last_bbox is None
        )

        if do_detect:
            bbox = self._detect_face(frame_bgr)
            if bbox is None:
                self._last_bbox = None
                self._frame_count += 1
                return None
            self._last_bbox = bbox
            # Seed optical flow tracker with corner points inside bbox
            x1, y1, x2, y2 = bbox
            self._track_pts = np.array([
                [x1, y1], [x2, y1], [x1, y2], [x2, y2]
            ], dtype=np.float32).reshape(-1, 1, 2)
        else:
            # Track with optical flow
            if self._prev_gray is not None and self._track_pts is not None:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, self._track_pts, None
                )
                if new_pts is not None and status.sum() >= 2:
                    self._track_pts = new_pts
                    pts = new_pts.reshape(-1, 2)
                    x1 = int(pts[:, 0].min())
                    y1 = int(pts[:, 1].min())
                    x2 = int(pts[:, 0].max())
                    y2 = int(pts[:, 1].max())
                    self._last_bbox = (x1, y1, x2, y2)

        self._prev_gray = gray
        self._frame_count += 1

        if self._last_bbox is None:
            return None

        # Expand bbox with margin and extract crop
        x1, y1, x2, y2 = self._last_bbox
        margin = int((x2 - x1) * 0.2)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_resized = cv2.resize(face_crop, (self.target_size, self.target_size))

        # Affine matrix to paste back (simple scale + offset)
        M_inv = np.float32([
            [(x2 - x1) / self.target_size, 0, x1],
            [0, (y2 - y1) / self.target_size, y1],
        ])

        return face_resized, M_inv, (x1, y1, x2, y2)

    def paste_back(
        self,
        frame_bgr: np.ndarray,
        face_result: np.ndarray,
        M_inv: np.ndarray,
        bbox: Tuple,
    ) -> np.ndarray:
        """Blend the lip-synced face crop back into the full frame."""
        x1, y1, x2, y2 = bbox
        h_crop = y2 - y1
        w_crop = x2 - x1

        if h_crop <= 0 or w_crop <= 0:
            return frame_bgr

        resized = cv2.resize(face_result, (w_crop, h_crop))

        result = frame_bgr.copy()

        # Soft-blend at edges to avoid hard seams
        mask = np.ones((h_crop, w_crop), dtype=np.float32)
        ksize = max(3, min(h_crop, w_crop) // 8) | 1   # must be odd
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        mask = mask[:, :, np.newaxis]

        roi = result[y1:y2, x1:x2].astype(np.float32)
        blended = roi * (1 - mask) + resized.astype(np.float32) * mask
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result

    def _detect_face(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lms = results.multi_face_landmarks[0].landmark
        xs = [int(lm.x * w) for lm in lms]
        ys = [int(lm.y * h) for lm in lms]
        return min(xs), min(ys), max(xs), max(ys)
