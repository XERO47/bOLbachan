"""
Lip sync inference pipeline.

Supports two backends:
  - musetalk : TMElyralab/MuseTalk  (recommended, real-time capable)
  - wav2lip  : Wav2Lip              (fallback, well-tested)

The public interface is:
    pipeline = LipSyncPipeline(model_type="musetalk")
    out_frame = pipeline.process(frame_bgr, audio_pcm_float32)
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from config import settings
from face_tracker import FaceTracker

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MuseTalk wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MuseTalkBackend:
    """
    Thin wrapper around MuseTalk's inference stack.
    MuseTalk repo must be on PYTHONPATH (handled by Dockerfile).
    """

    def __init__(self, model_dir: str, device: str, half: bool = True):
        self.device = torch.device(device)
        self.half = half
        self._load(model_dir)

    def _load(self, model_dir: str):
        logger.info("Loading MuseTalk models from %s …", model_dir)
        try:
            # MuseTalk exposes these after you clone the repo
            from musetalk.whisper.audio2feature import Audio2Feature
            from musetalk.models.vae import AutoencoderKL
            from musetalk.models.unet import UNet2DConditionModel
            from musetalk.utils.utils import load_all_model

            self.audio_processor = Audio2Feature(
                model_path=str(Path(model_dir) / "whisper" / "tiny.pt")
            )
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=str(Path(model_dir) / "musetalk" / "pytorch_model.bin"),
                vae_type="sd-vae-ft-mse",
                unet_config=str(Path(model_dir) / "musetalk" / "musetalk.json"),
            )

            self.vae = self.vae.to(self.device)
            self.unet = self.unet.to(self.device)
            self.pe = self.pe.to(self.device)

            if self.half:
                self.vae.half()
                self.unet.half()
                self.pe.half()

            self.vae.eval()
            self.unet.eval()
            logger.info("MuseTalk loaded OK (fp16=%s)", self.half)

        except ImportError as e:
            raise RuntimeError(
                "MuseTalk not found on PYTHONPATH. "
                "Make sure /app/MuseTalk is in PYTHONPATH inside the container."
            ) from e

    @torch.inference_mode()
    def infer(self, face_crop: np.ndarray, audio_pcm: np.ndarray) -> np.ndarray:
        """
        face_crop : (H, W, 3) BGR uint8, H==W==256
        audio_pcm : float32 PCM, sample_rate=16000, ~200ms context
        Returns   : (H, W, 3) BGR uint8 with animated lips
        """
        dtype = torch.float16 if self.half else torch.float32

        # 1. Audio → whisper features
        audio_feat = self.audio_processor.get_audio_feature(audio_pcm, weight_dtype=dtype)
        audio_feat = audio_feat.to(self.device)

        # 2. Encode face with VAE
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_t = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0)
        face_t = face_t.to(self.device, dtype=dtype) / 127.5 - 1.0
        latents = self.vae.encode(face_t).latent_dist.sample()

        # 3. Mask lower-half of face (MuseTalk convention)
        mask = torch.ones_like(latents)
        mask[:, :, latents.shape[2] // 2 :, :] = 0
        masked_latents = latents * mask

        # 4. UNet denoising pass (single step, not diffusion — MuseTalk shortcut)
        timesteps = torch.zeros(1, dtype=torch.long, device=self.device)
        noise_pred = self.unet(
            torch.cat([masked_latents, mask], dim=1),
            timesteps,
            encoder_hidden_states=audio_feat.unsqueeze(0),
        ).sample

        # 5. Decode
        decoded = self.vae.decode(noise_pred).sample
        decoded = (decoded.clamp(-1, 1) + 1) / 2 * 255
        decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        result = cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Wav2Lip wrapper (fallback)
# ──────────────────────────────────────────────────────────────────────────────

class Wav2LipBackend:
    """
    Minimal Wav2Lip wrapper. Expects wav2lip_gan.pth in model_dir.
    Clone https://github.com/Rudrabha/Wav2Lip alongside this project.
    """

    def __init__(self, model_dir: str, device: str, half: bool = True):
        self.device = torch.device(device)
        self.half = half
        self._load(model_dir)

    def _load(self, model_dir: str):
        logger.info("Loading Wav2Lip model …")
        try:
            sys.path.insert(0, "/app/Wav2Lip")
            from models import Wav2Lip

            ckpt_path = Path(model_dir) / "wav2lip_gan.pth"
            checkpoint = torch.load(str(ckpt_path), map_location=self.device)
            s = checkpoint["state_dict"]
            new_s = {k.replace("module.", ""): v for k, v in s.items()}
            self.model = Wav2Lip()
            self.model.load_state_dict(new_s)
            self.model = self.model.to(self.device).eval()
            if self.half:
                self.model.half()
            logger.info("Wav2Lip loaded OK")
        except ImportError as e:
            raise RuntimeError("Wav2Lip not found on PYTHONPATH.") from e

    @torch.inference_mode()
    def infer(self, face_crop: np.ndarray, audio_pcm: np.ndarray) -> np.ndarray:
        import librosa

        dtype = torch.float16 if self.half else torch.float32

        # Mel spectrogram  (Wav2Lip uses 80-band mel, 16kHz, hop 200)
        mel = librosa.feature.melspectrogram(
            y=audio_pcm, sr=16000, n_mels=80,
            hop_length=200, win_length=800, fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_t = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0)
        mel_t = mel_t.to(self.device, dtype=dtype)

        # Face tensor
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_t = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0)
        face_t = face_t.to(self.device, dtype=dtype) / 255.0

        out = self.model(mel_t, face_t)
        out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Public pipeline
# ──────────────────────────────────────────────────────────────────────────────

class LipSyncPipeline:
    def __init__(
        self,
        model_type: str = "musetalk",
        model_dir: str = "/app/models/musetalk",
        device: str = "cuda",
        half: bool = True,
        face_det_interval: int = 5,
        target_face_size: int = 256,
    ):
        self.device = device
        self.target_face_size = target_face_size

        if model_type == "musetalk":
            self.backend = MuseTalkBackend(model_dir, device, half)
        elif model_type == "wav2lip":
            self.backend = Wav2LipBackend(model_dir, device, half)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.face_tracker = FaceTracker(
            detect_interval=face_det_interval,
            target_size=target_face_size,
        )
        logger.info("LipSyncPipeline ready (%s, device=%s, fp16=%s)", model_type, device, half)

    def process(self, frame_bgr: np.ndarray, audio_pcm: np.ndarray) -> np.ndarray:
        """
        frame_bgr : (H, W, 3) BGR uint8 — original webcam frame
        audio_pcm : float32 PCM at 16kHz — ~200ms window
        Returns   : (H, W, 3) BGR uint8 — frame with animated lips
        """
        t0 = time.perf_counter()

        result = self.face_tracker.get_face_crop(frame_bgr)
        if result is None:
            return frame_bgr   # No face found — pass through unchanged

        face_crop, M_inv, bbox = result

        # Resize to model's expected input size if needed
        if face_crop.shape[0] != self.target_face_size:
            face_crop = cv2.resize(face_crop, (self.target_face_size, self.target_face_size))

        # Run inference
        animated_face = self.backend.infer(face_crop, audio_pcm)

        # Paste result back into original frame
        output = self.face_tracker.paste_back(frame_bgr, animated_face, M_inv, bbox)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("process() %.1f ms", elapsed_ms)

        return output

    def warmup(self):
        """Run one dummy forward pass so the first real frame isn't slow."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_audio = np.zeros(3200, dtype=np.float32)   # 200ms @ 16kHz
        try:
            self.process(dummy_frame, dummy_audio)
            logger.info("Pipeline warmup done")
        except Exception:
            pass  # warmup failure is non-fatal
