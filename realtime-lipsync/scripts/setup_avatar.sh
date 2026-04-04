#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup script: DeepFaceLive-style avatar pipeline on L40S
# Run from /workspace/bOLbachan/realtime-lipsync/
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
echo "==> Project dir: $PROJECT_DIR"

MODELS_DIR="$PROJECT_DIR/models"
mkdir -p "$MODELS_DIR"

# ── 1. Fix DNS if needed ─────────────────────────────────────────────────────
if ! curl -s --max-time 5 https://pypi.org > /dev/null; then
  echo "==> Fixing DNS..."
  echo "nameserver 8.8.8.8" >> /etc/resolv.conf
  echo "nameserver 1.1.1.1" >> /etc/resolv.conf
fi

# ── 2. Install Python deps ───────────────────────────────────────────────────
echo "==> Installing Python dependencies..."
pip install -q \
  insightface==0.7.3 \
  onnxruntime-gpu==1.17.1 \
  opencv-python-headless==4.8.1.78 \
  numpy==1.26.4 \
  librosa==0.10.1 \
  soundfile==0.12.1 \
  fastapi==0.110.0 \
  "uvicorn[standard]==0.27.1" \
  websockets==12.0 \
  python-multipart==0.0.9 \
  pydantic==2.6.3 \
  pydantic-settings==2.2.1

# ── 3. Clone Wav2Lip ─────────────────────────────────────────────────────────
if [ ! -d "$PROJECT_DIR/Wav2Lip" ]; then
  echo "==> Cloning Wav2Lip..."
  git clone https://github.com/Rudrabha/Wav2Lip.git "$PROJECT_DIR/Wav2Lip"
else
  echo "==> Wav2Lip already cloned."
fi

# ── 4. Download wav2lip_gan.pth ──────────────────────────────────────────────
WAV2LIP_CKPT="$MODELS_DIR/wav2lip_gan.pth"
if [ ! -f "$WAV2LIP_CKPT" ]; then
  echo "==> Downloading wav2lip_gan.pth (~420MB) from HuggingFace..."
  pip install -q huggingface_hub
  python3 -c "
import shutil, os
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='numz/wav2lip', filename='wav2lip_gan.pth', cache_dir='/tmp/hf_cache')
os.makedirs('$MODELS_DIR', exist_ok=True)
shutil.copy(path, '$WAV2LIP_CKPT')
print('Saved to $WAV2LIP_CKPT')
"
else
  echo "==> wav2lip_gan.pth already present."
fi

# ── 5. Download inswapper_128.onnx ──────────────────────────────────────────
INSWAPPER_CKPT="$MODELS_DIR/inswapper_128.onnx"
if [ ! -f "$INSWAPPER_CKPT" ]; then
  echo "==> Downloading inswapper_128.onnx (~560MB) from HuggingFace..."
  python3 -c "
import shutil, os
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='ezioruan/inswapper_128.onnx', filename='inswapper_128.onnx', cache_dir='/tmp/hf_cache')
os.makedirs('$MODELS_DIR', exist_ok=True)
shutil.copy(path, '$INSWAPPER_CKPT')
print('Saved to $INSWAPPER_CKPT')
"
else
  echo "==> inswapper_128.onnx already present."
fi

# ── 6. InsightFace buffalo_l (auto-downloaded on first use) ─────────────────
echo "==> Pre-downloading InsightFace buffalo_l model..."
python3 -c "
import insightface
app = insightface.app.FaceAnalysis(name='buffalo_l', root='$MODELS_DIR',
    providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
print('InsightFace buffalo_l ready.')
"

# ── 7. Create placeholder avatar image if none exists ──────────────────────
AVATAR_IMG="$MODELS_DIR/avatar.jpg"
if [ ! -f "$AVATAR_IMG" ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  ACTION REQUIRED: Upload your avatar reference photo        ║"
  echo "║                                                              ║"
  echo "║  scp /path/to/your/face.jpg root@<IP>:$AVATAR_IMG  ║"
  echo "║                                                              ║"
  echo "║  Or set AVATAR_IMAGE env var to a different path.           ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
else
  echo "==> Avatar image found: $AVATAR_IMG"
fi

echo ""
echo "==> Setup complete! To start the server:"
echo "    cd $PROJECT_DIR/server"
echo "    MODEL_TYPE=avatar AVATAR_IMAGE=$AVATAR_IMG python main.py"
