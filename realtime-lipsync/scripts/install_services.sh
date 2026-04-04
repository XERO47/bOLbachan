#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot: install systemd services + Cloudflare tunnel
# Run once after setup_avatar.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Project: $PROJECT_DIR"

# ── 1. Install cloudflared ────────────────────────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
  echo "==> Installing cloudflared..."
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
  echo "==> cloudflared installed: $(cloudflared --version)"
else
  echo "==> cloudflared already installed"
fi

# ── 2. Deepfake server systemd service ───────────────────────────────────────
cat > /etc/systemd/system/deepfake.service <<EOF
[Unit]
Description=DeepFace Live Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR/server
Environment=PYTHONPATH=$PROJECT_DIR/server
Environment=SOURCE_FACE=$PROJECT_DIR/models/avatar.jpg
ExecStart=/usr/bin/python3 $PROJECT_DIR/server/main.py
Restart=always
RestartSec=5
StandardOutput=append:/var/log/deepfake.log
StandardError=append:/var/log/deepfake.log

[Install]
WantedBy=multi-user.target
EOF

# ── 3. Cloudflare tunnel systemd service ─────────────────────────────────────
cat > /etc/systemd/system/deepfake-tunnel.service <<EOF
[Unit]
Description=Cloudflare Tunnel for DeepFace Live
After=deepfake.service
Requires=deepfake.service

[Service]
Type=simple
ExecStart=/usr/local/bin/cloudflared tunnel --url http://localhost:8000 \
  --logfile /var/log/cloudflared.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ── 4. Enable and start ───────────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable deepfake deepfake-tunnel
systemctl restart deepfake

echo ""
echo "==> Waiting for server to start..."
sleep 20
systemctl restart deepfake-tunnel

echo ""
echo "==> Waiting for Cloudflare tunnel URL..."
sleep 8

TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /var/log/cloudflared.log | tail -1)

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " DeepFace Live is running!"
echo ""
echo " Dashboard : $TUNNEL_URL"
echo " OBS MJPEG : $TUNNEL_URL/stream.mjpeg"
echo " WebSocket : wss://$(echo $TUNNEL_URL | sed 's|https://||')/ws/deepfake"
echo ""
echo " Hardcode the OBS MJPEG URL above — it changes each restart."
echo " For a fixed URL: set up a named Cloudflare tunnel."
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo " Status:"
echo "   systemctl status deepfake"
echo "   systemctl status deepfake-tunnel"
echo "   tail -f /var/log/deepfake.log"
echo "   tail -f /var/log/cloudflared.log"
