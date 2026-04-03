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
    Wrapper around MuseTalk's actual inference API (V1.5).

    MuseTalk's VAE and UNet are plain Python wrapper classes, NOT nn.Modules,
    so they cannot be moved with .to(). They handle device placement internally.

    Audio pipeline: PCM float32 → temp WAV → audio2feat() → per-frame whisper features.
    Whisper needs ≥1 s of audio to return non-empty segments; we buffer accordingly.
    """

    def __init__(self, model_dir: str, device: str, half: bool = True):
        self.device = torch.device(device)
        self.half = half
        self._models_root = Path(model_dir)
        self._load(model_dir)

    def _load(self, model_dir: str):
        import os
        logger.info("Loading MuseTalk models from %s …", model_dir)

        project_root = Path(model_dir).parent
        models_root  = Path(model_dir)

        # load_all_model resolves VAE path as CWD-relative "models/sd-vae-ft-mse"
        _prev_cwd = os.getcwd()
        os.chdir(str(project_root))

        try:
            from musetalk.whisper.audio2feature import Audio2Feature
            from musetalk.utils.utils import load_all_model

            self.audio_processor = Audio2Feature(
                model_path=str(models_root / "whisper" / "tiny.pt")
            )

            # Returns (VAE wrapper, UNet wrapper, PositionalEncoding nn.Module)
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=str(models_root / "musetalkV15" / "unet.pth"),
                vae_type="sd-vae-ft-mse",
                unet_config=str(models_root / "musetalkV15" / "musetalk.json"),
            )
        finally:
            os.chdir(_prev_cwd)

        # vae and unet are plain wrappers — their inner .vae / .model are nn.Modules
        # that already moved to CUDA in their __init__. Only pe needs manual placement.
        dtype = torch.float16 if self.half else torch.float32
        self.pe = self.pe.to(self.device)
        if self.half:
            self.unet.model = self.unet.model.half()
            self.vae.vae    = self.vae.vae.half()
            self.vae._use_float16 = True
            self.pe         = self.pe.half()

        self.unet.model.eval()
        self.pe.eval()
        logger.info("MuseTalk loaded OK (fp16=%s)", self.half)

    @torch.inference_mode()
    def infer(self, face_crop: np.ndarray, audio_pcm: np.ndarray) -> np.ndarray:
        """
        face_crop : (H, W, 3) BGR uint8, resized to 256×256
        audio_pcm : float32 PCM at 16kHz, ≥1 second recommended
        Returns   : (H, W, 3) BGR uint8 with animated lips
        """
        import os
        import tempfile
        import soundfile as sf

        dtype = torch.float16 if self.half else torch.float32

        # ── Audio: PCM → temp WAV → whisper features ─────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio_pcm, 16000, subtype="PCM_16")
            feat_array = self.audio_processor.audio2feat(tmp_path)
        finally:
            os.unlink(tmp_path)

        # get_sliced_feature returns (50, 384) for vid_idx=0, fps=25
        audio_feat_np, _ = self.audio_processor.get_sliced_feature(
            feat_array, vid_idx=0, fps=25
        )
        audio_feat = torch.from_numpy(audio_feat_np).unsqueeze(0).to(self.device, dtype=dtype)
        audio_feat = self.pe(audio_feat)  # positional encoding → (1, 50, 384)

        # ── Face: VAE encode (masked + reference latents) ─────────────────────
        # vae.get_latents_for_unet accepts BGR numpy (H,W,3)
        latent_input = self.vae.get_latents_for_unet(face_crop)  # (1, 8, 32, 32)

        # ── UNet inference ────────────────────────────────────────────────────
        timestep = torch.tensor([0], device=self.device, dtype=torch.long)
        pred_latents = self.unet.model(
            latent_input,
            timestep,
            encoder_hidden_states=audio_feat,
        ).sample  # (1, 4, 32, 32)

        # ── VAE decode → BGR uint8 ────────────────────────────────────────────
        output_frames = self.vae.decode_latents(pred_latents)  # list of BGR numpy
        return output_frames[0]


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
