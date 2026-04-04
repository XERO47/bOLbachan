#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup script: DeepFaceLive-style avatar pipeline on L40S
# Run once from /workspace/bOLbachan/realtime-lipsync/
# Tested: NGC nvcr.io/nvidia/pytorch:23.07-py3  (CUDA 12.1 + cuDNN 8.9)
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

# Remove any conflicting opencv first
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
rm -rf /usr/local/lib/python3.10/dist-packages/cv2 2>/dev/null || true

pip install -q \
  insightface==0.7.3 \
  opencv-python-headless==4.8.1.78 \
  fastapi==0.110.0 \
  "uvicorn[standard]==0.27.1" \
  websockets==12.0 \
  python-multipart==0.0.9 \
  pydantic==2.6.3 \
  pydantic-settings==2.2.1 \
  librosa==0.10.1 \
  soundfile==0.12.1 \
  huggingface_hub \
  "protobuf<4.0"

# Pin numpy to 1.x (2.x breaks cv2 + torch compiled for numpy 1.x)
pip install -q "numpy==1.26.4" --force-reinstall

# onnxruntime-gpu for CUDA 12 + cuDNN 8 (NGC 23.07 container)
# Must come from the CUDA-12 specific feed to get the cuDNN 8 variant
pip install -q 'onnxruntime-gpu==1.18.0' \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

echo "==> Verifying imports..."
python3 -c "import cv2, numpy, insightface, onnxruntime, fastapi; print('All imports OK')"

# ── 3. Clone Wav2Lip (needed if ENABLE_LIP_SYNC=1) ──────────────────────────
if [ ! -d "$PROJECT_DIR/Wav2Lip" ]; then
  echo "==> Cloning Wav2Lip..."
  git clone https://github.com/Rudrabha/Wav2Lip.git "$PROJECT_DIR/Wav2Lip"
else
  echo "==> Wav2Lip already cloned."
fi

# ── 4. Download inswapper_128.onnx (required) ───────────────────────────────
INSWAPPER_CKPT="$MODELS_DIR/inswapper_128.onnx"
if [ ! -f "$INSWAPPER_CKPT" ] || [ ! -s "$INSWAPPER_CKPT" ]; then
  echo "==> Downloading inswapper_128.onnx (~560MB)..."
  python3 -c "
import shutil, os
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='ezioruan/inswapper_128.onnx', filename='inswapper_128.onnx', cache_dir='/tmp/hf_cache')
os.makedirs('$MODELS_DIR', exist_ok=True)
shutil.copy(path, '$INSWAPPER_CKPT')
print('Saved to $INSWAPPER_CKPT  size=%.1fMB' % (os.path.getsize('$INSWAPPER_CKPT')/1e6))
"
else
  echo "==> inswapper_128.onnx already present."
fi

# ── 5. Download wav2lip_gan.pth (optional, for lip sync) ────────────────────
WAV2LIP_CKPT="$MODELS_DIR/wav2lip_gan.pth"
if [ ! -f "$WAV2LIP_CKPT" ] || [ ! -s "$WAV2LIP_CKPT" ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  OPTIONAL: wav2lip_gan.pth for lip sync                     ║"
  echo "║                                                              ║"
  echo "║  Download from Wav2Lip official page:                        ║"
  echo "║  https://github.com/Rudrabha/Wav2Lip#getting-the-weights    ║"
  echo "║  Then: scp wav2lip_gan.pth root@<IP>:$WAV2LIP_CKPT  ║"
  echo "║                                                              ║"
  echo "║  Without this, face-swap still works (no lip sync).         ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
else
  echo "==> wav2lip_gan.pth already present."
fi

# ── 6. Pre-download InsightFace buffalo_l ────────────────────────────────────
echo "==> Pre-downloading InsightFace buffalo_l model..."
python3 -c "
import insightface
app = insightface.app.FaceAnalysis(name='buffalo_l', root='$PROJECT_DIR',
    providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
print('InsightFace buffalo_l ready.')
"

# ── 7. Avatar image ──────────────────────────────────────────────────────────
AVATAR_IMG="$MODELS_DIR/avatar.jpg"
if [ ! -f "$AVATAR_IMG" ] || [ ! -s "$AVATAR_IMG" ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  ACTION REQUIRED: Upload your avatar reference photo         ║"
  echo "║                                                              ║"
  echo "║  scp /path/to/your/face.jpg root@<IP>:$AVATAR_IMG   ║"
  echo "║                                                              ║"
  echo "║  Or set AVATAR_IMAGE env var to a different path.           ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
else
  echo "==> Avatar image found: $AVATAR_IMG"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Setup complete! Run the server:"
echo ""
echo "   cd $PROJECT_DIR/server"
echo "   MODEL_TYPE=avatar \\"
echo "   AVATAR_IMAGE=$AVATAR_IMG \\"
echo "   PYTHONPATH=. python main.py"
echo ""
echo " SSH tunnel (run on your LOCAL machine):"
echo "   ssh -L 8000:localhost:8000 root@<INSTANCE_IP>"
echo " Then open: http://localhost:8000"
echo "═══════════════════════════════════════════════════════════════"
