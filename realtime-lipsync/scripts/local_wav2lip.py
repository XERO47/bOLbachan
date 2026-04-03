"""
Local real-time Wav2Lip on a single reference photo.

Usage:
    python local_wav2lip.py [--image path/to/face.jpg]

Controls:
    SPACE  - capture reference face from webcam (if no --image given)
    Q      - quit
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import sounddevice as sd
import librosa

# ── Wav2Lip repo must be on path ──────────────────────────────────────────────
WAV2LIP_DIR = Path(__file__).parent.parent / "Wav2Lip"
sys.path.insert(0, str(WAV2LIP_DIR))

SAMPLE_RATE    = 16000
MEL_STEP_SIZE  = 16        # mel frames per video frame (Wav2Lip default)
HOP_LENGTH     = 200       # samples per mel frame @ 16kHz
FPS            = 25
FACE_SIZE      = 96        # Wav2Lip input face size
AUDIO_CHUNK    = int(SAMPLE_RATE / FPS)   # ~640 samples per frame
MEL_WINDOW     = MEL_STEP_SIZE * HOP_LENGTH  # audio samples for one mel chunk


def load_wav2lip(checkpoint_path: str, device: torch.device):
    from models import Wav2Lip
    model = Wav2Lip()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd_ = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd_)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    print(f"[wav2lip] loaded from {checkpoint_path}")
    return model


def detect_face(frame_bgr: np.ndarray):
    """Returns (x1,y1,x2,y2) of biggest face, or None."""
    import mediapipe as mp
    h, w = frame_bgr.shape[:2]
    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = mp_fd.process(rgb)
    mp_fd.close()
    if not res.detections:
        return None
    d = res.detections[0].location_data.relative_bounding_box
    x1 = max(0, int(d.xmin * w))
    y1 = max(0, int(d.ymin * h))
    x2 = min(w, x1 + int(d.width * w))
    y2 = min(h, y1 + int(d.height * h))
    # Add 20% margin
    mx = int((x2 - x1) * 0.2)
    my = int((y2 - y1) * 0.2)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
    return x1, y1, x2, y2


def prepare_face(frame_bgr: np.ndarray, bbox):
    """Crop, resize to 96x96, split into top+bottom halves (Wav2Lip input format)."""
    x1, y1, x2, y2 = bbox
    face = frame_bgr[y1:y2, x1:x2]
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) / 255.0
    # Wav2Lip masks lower half of input face
    masked = face_rgb.copy()
    masked[FACE_SIZE // 2:] = 0
    # Stack: [masked, original] → (2, 3, 96, 96) combined as (6, 96, 96)
    img_t = np.concatenate([
        np.transpose(masked, (2, 0, 1)),
        np.transpose(face_rgb, (2, 0, 1))
    ], axis=0)
    return img_t.astype(np.float32), face, (x1, y1, x2, y2)


def audio_to_mel(pcm_float32: np.ndarray) -> np.ndarray:
    """Convert PCM to 80-band mel spectrogram, return (80, T) float32."""
    mel = librosa.feature.melspectrogram(
        y=pcm_float32, sr=SAMPLE_RATE, n_mels=80,
        hop_length=HOP_LENGTH, win_length=800, fmax=8000
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)


@torch.inference_mode()
def infer(model, face_tensor, mel_chunk, device):
    """
    face_tensor : (6, 96, 96) float32
    mel_chunk   : (80, 16)    float32
    Returns     : (96, 96, 3) BGR uint8
    """
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    img_t = torch.from_numpy(face_tensor).unsqueeze(0).to(device, dtype=dtype)
    mel_t = torch.from_numpy(mel_chunk).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    pred  = model(mel_t, img_t)                          # (1, 3, 96, 96) in [0,1]
    pred  = pred.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    pred  = (pred * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)


def paste_back(canvas: np.ndarray, face_result: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return canvas
    resized = cv2.resize(face_result, (w, h))
    out = canvas.copy()
    # Feathered blend at edges
    mask = np.ones((h, w), dtype=np.float32)
    k = max(3, min(h, w) // 6) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, None]
    out[y1:y2, x1:x2] = (
        out[y1:y2, x1:x2].astype(np.float32) * (1 - mask) +
        resized.astype(np.float32) * mask
    ).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Path to reference face image (skip webcam capture)")
    parser.add_argument("--model", type=str,
                        default=str(WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"),
                        help="Path to wav2lip_gan.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not Path(args.model).exists():
        print(f"ERROR: model not found at {args.model}")
        print("Download from: https://github.com/Rudrabha/Wav2Lip#getting-the-weights")
        sys.exit(1)
    model = load_wav2lip(args.model, device)

    # ── Get reference frame ───────────────────────────────────────────────────
    if args.image:
        ref_frame = cv2.imread(args.image)
        if ref_frame is None:
            print(f"ERROR: cannot read {args.image}"); sys.exit(1)
        print(f"[ref] loaded {args.image}")
    else:
        print("[webcam] press SPACE to capture reference face, Q to quit")
        cap = cv2.VideoCapture(0)
        ref_frame = None
        while True:
            ok, frame = cap.read()
            if not ok: break
            cv2.putText(frame, "Press SPACE to capture face | Q to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Capture Reference Face", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                ref_frame = frame.copy()
                break
            elif k == ord('q'):
                cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        cap.release()
        cv2.destroyAllWindows()
        if ref_frame is None:
            print("No frame captured"); sys.exit(1)

    # ── Detect face in reference ──────────────────────────────────────────────
    print("[face] detecting...")
    bbox = detect_face(ref_frame)
    if bbox is None:
        print("ERROR: no face detected in reference image. Try a clearer photo.")
        sys.exit(1)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(ref_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Reference face (any key to continue)", ref_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    face_tensor, face_crop_orig, bbox = prepare_face(ref_frame, bbox)
    print(f"[face] bbox={bbox}, crop={face_crop_orig.shape}")

    # ── Audio buffer ──────────────────────────────────────────────────────────
    audio_q: queue.Queue = queue.Queue()
    audio_ring = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2s ring buffer

    def audio_callback(indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        audio_q.put(mono)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        dtype="float32", blocksize=AUDIO_CHUNK,
        callback=audio_callback
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    print("[running] Wav2Lip live — press Q in output window to quit")
    output_frame = ref_frame.copy()
    frame_times = []
    mel_buffer = np.zeros(MEL_WINDOW * 2, dtype=np.float32)

    with stream:
        while True:
            # Drain audio queue into ring buffer
            while not audio_q.empty():
                chunk = audio_q.get_nowait()
                mel_buffer = np.roll(mel_buffer, -len(chunk))
                mel_buffer[-len(chunk):] = chunk

            # Compute mel from last MEL_WINDOW samples
            audio_window = mel_buffer[-MEL_WINDOW:]
            mel = audio_to_mel(audio_window)  # (80, T)

            if mel.shape[1] >= MEL_STEP_SIZE:
                mel_chunk = mel[:, -MEL_STEP_SIZE:]  # (80, 16) — last 16 frames

                t0 = time.perf_counter()
                face_result = infer(model, face_tensor, mel_chunk, device)
                dt = time.perf_counter() - t0

                frame_times.append(dt)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times))

                output_frame = paste_back(ref_frame.copy(), face_result, bbox)
                cv2.putText(output_frame, f"FPS: {avg_fps:.1f}  inference: {dt*1000:.0f}ms",
                            (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                output_frame = ref_frame.copy()
                cv2.putText(output_frame, "Listening...",
                            (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("Wav2Lip Live", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
