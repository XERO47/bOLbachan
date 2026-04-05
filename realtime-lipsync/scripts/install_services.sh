#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot setup: install cloudflared + start server + tunnel via tmux
# Works on E2E Networks (no systemd). Run once after setup_avatar.sh.
# ─────────────────────────────────────────────────────────────────────────────
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> Project: $PROJECT_DIR"

# ── 1. Install tmux if missing ────────────────────────────────────────────────
if ! command -v tmux &>/dev/null; then
  echo "==> Installing tmux..."
  apt-get install -y tmux -qq
fi

# ── 2. Install cloudflared ────────────────────────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
  echo "==> Installing cloudflared..."
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
  echo "==> cloudflared: $(cloudflared --version)"
else
  echo "==> cloudflared already installed"
fi

# ── 3. Kill old sessions ──────────────────────────────────────────────────────
tmux kill-session -t deepfake  2>/dev/null || true
tmux kill-session -t cftunnel  2>/dev/null || true
pkill -f "python main.py"      2>/dev/null || true
pkill -f cloudflared           2>/dev/null || true
sleep 1

# ── 4. Start server in tmux ───────────────────────────────────────────────────
echo "==> Starting DeepFace server in tmux session 'deepfake'..."
tmux new-session -d -s deepfake -x 220 -y 50
tmux send-keys -t deepfake \
  "cd $PROJECT_DIR/server && PYTHONPATH=. LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH SOURCE_FACE=$PROJECT_DIR/models/avatar.jpg python main.py 2>&1 | tee /var/log/deepfake.log" \
  Enter

echo "==> Waiting 25s for models to load..."
sleep 25

# ── 5. Start cloudflare tunnel ────────────────────────────────────────────────
echo "==> Starting Cloudflare tunnel in tmux session 'cftunnel'..."
tmux new-session -d -s cftunnel -x 220 -y 10
tmux send-keys -t cftunnel \
  "cloudflared tunnel --url http://localhost:8000 2>&1 | tee /var/log/cloudflared.log" \
  Enter

echo "==> Waiting for tunnel URL..."
sleep 10

TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /var/log/cloudflared.log 2>/dev/null | tail -1)

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " DeepFace Live is running!"
echo ""
if [ -n "$TUNNEL_URL" ]; then
  echo " Dashboard : $TUNNEL_URL"
  echo " OBS MJPEG : $TUNNEL_URL/stream.mjpeg"
  echo " WebSocket : ${TUNNEL_URL/https:/wss:}/ws/deepfake"
else
  echo " Tunnel URL not ready yet. Check: grep trycloudflare /var/log/cloudflared.log"
fi
echo ""
echo " Attach to sessions:"
echo "   tmux attach -t deepfake   (server logs)"
echo "   tmux attach -t cftunnel   (tunnel logs)"
echo "═══════════════════════════════════════════════════════════════"
