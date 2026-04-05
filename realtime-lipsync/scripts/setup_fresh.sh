#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot setup for a FRESH E2E Networks instance (Jupyter py3.10-cuda12.1)
#
# Run once after spinning up the instance:
#   bash scripts/setup_fresh.sh
#
# What it does:
#   1. Installs Python packages (insightface, onnxruntime-gpu 1.18, TRT 10.x)
#   2. Downloads models (inswapper_128.onnx + buffalo_l)
#   3. Verifies TRT provider works
#   4. Starts server in tmux
#   5. Optionally starts Cloudflare tunnel
# ─────────────────────────────────────────────────────────────────────────────
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> Project: $PROJECT_DIR"
MODELS_DIR="$PROJECT_DIR/models"
mkdir -p "$MODELS_DIR"

# ── 1. System deps ────────────────────────────────────────────────────────────
apt-get install -y tmux libgl1 libglib2.0-0 -qq

# ── 2. Python packages ────────────────────────────────────────────────────────
echo "==> Installing Python packages..."
pip install -q \
    "onnxruntime-gpu==1.18.0" \
    "insightface==0.7.3" \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "python-multipart==0.0.9" \
    "websockets==12.0" \
    "opencv-python-headless==4.9.0.80" \
    "numpy==1.26.4"

# TensorRT 10.x via NVIDIA pip index — compatible with ORT 1.18 TRT EP
echo "==> Installing TensorRT 10.x (pip)..."
pip install -q \
    --extra-index-url https://pypi.nvidia.com \
    "tensorrt==10.0.1.6" \
    "tensorrt-cu12-bindings==10.0.1.6" \
    "tensorrt-cu12-libs==10.0.1.6"

# ── 3. Find TRT library path (pip installs .so into site-packages) ────────────
TRT_LIB_PATH=$(python3 -c "
import tensorrt_libs, os
print(os.path.dirname(tensorrt_libs.__file__))
" 2>/dev/null || python3 -c "
import tensorrt, os
print(os.path.dirname(tensorrt.__file__))
")
echo "==> TRT libs at: $TRT_LIB_PATH"

# ── 4. Verify TRT provider loads ──────────────────────────────────────────────
echo "==> Verifying TensorRT provider..."
LD_LIBRARY_PATH="$TRT_LIB_PATH:$LD_LIBRARY_PATH" python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
if 'TensorrtExecutionProvider' in providers:
    print('  TensorRT provider: OK')
else:
    print('  WARNING: TensorRT provider not available:', providers)
"

# ── 5. Download models ────────────────────────────────────────────────────────
# inswapper_128.onnx (~530 MB) — face swap model
if [ ! -f "$MODELS_DIR/inswapper_128.onnx" ]; then
    echo "==> Downloading inswapper_128.onnx..."
    # Download from HuggingFace or your own source
    # Replace this URL with your model source:
    wget -q --show-progress \
        "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx" \
        -O "$MODELS_DIR/inswapper_128.onnx"
else
    echo "==> inswapper_128.onnx already exists"
fi

# buffalo_l — face analysis models (~300 MB)
if [ ! -d "$MODELS_DIR/buffalo_l" ]; then
    echo "==> Downloading buffalo_l..."
    pip install -q huggingface_hub
    python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', root='$MODELS_DIR/..')
print('  buffalo_l downloaded')
"
else
    echo "==> buffalo_l already exists"
fi

# ── 6. Create TRT cache dir ───────────────────────────────────────────────────
mkdir -p "$MODELS_DIR/trt_cache"

# ── 7. Write env file for server startup ──────────────────────────────────────
ENV_FILE="$PROJECT_DIR/server/.env_trt"
cat > "$ENV_FILE" << EOF
export LD_LIBRARY_PATH="$TRT_LIB_PATH:\$LD_LIBRARY_PATH"
export PYTHONPATH="."
export SOURCE_FACE="$MODELS_DIR/avatar.jpg"
EOF
echo "==> Env file written to $ENV_FILE"

# ── 8. Kill old sessions ──────────────────────────────────────────────────────
tmux kill-session -t deepfake 2>/dev/null || true
tmux kill-session -t cftunnel 2>/dev/null || true
pkill -f "python main.py" 2>/dev/null || true
sleep 1

# ── 9. Start server ───────────────────────────────────────────────────────────
echo "==> Starting server in tmux session 'deepfake'..."
echo "    NOTE: First start compiles TRT engines (3-5 min). Upload a face"
echo "    via the dashboard to trigger inswapper TRT compile."
tmux new-session -d -s deepfake -x 220 -y 50
tmux send-keys -t deepfake \
    "source $ENV_FILE && cd $PROJECT_DIR/server && python main.py 2>&1 | tee /var/log/deepfake.log" \
    Enter

echo "==> Waiting 30s for server to start..."
sleep 30
tail -5 /var/log/deepfake.log

# ── 10. Optional Cloudflare tunnel ───────────────────────────────────────────
echo ""
echo "==> Install cloudflared? (y/N)"
read -r ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    if ! command -v cloudflared &>/dev/null; then
        curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
            -o /usr/local/bin/cloudflared
        chmod +x /usr/local/bin/cloudflared
    fi
    tmux new-session -d -s cftunnel -x 220 -y 10
    tmux send-keys -t cftunnel \
        "cloudflared tunnel --url http://localhost:8000 2>&1 | tee /var/log/cloudflared.log" \
        Enter
    sleep 8
    TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /var/log/cloudflared.log 2>/dev/null | tail -1)
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " DeepFace Live (TensorRT) is running!"
echo ""
echo " Dashboard  : http://$(hostname -I | awk '{print $1}'):8000"
if [ -n "$TUNNEL_URL" ]; then
echo " Tunnel     : $TUNNEL_URL"
fi
echo " Attach     : tmux attach -t deepfake"
echo ""
echo " Steps:"
echo "   1. Open dashboard → upload your face photo"
echo "   2. Wait for 'TRT compile done' in logs (first time only ~3 min)"
echo "   3. Start tray app on Windows → streaming begins"
echo "═══════════════════════════════════════════════════════════════"
