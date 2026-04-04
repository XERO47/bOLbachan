#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start the DeepFace Live server.
# Usage:
#   bash scripts/start_server.sh
#   SOURCE_FACE=/path/to/face.jpg bash scripts/start_server.sh
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"

# Default source face — upload a different one via /api/source-face anytime
SOURCE_FACE="${SOURCE_FACE:-$MODELS_DIR/avatar.jpg}"

if [ ! -f "$SOURCE_FACE" ]; then
  echo "INFO: No source face at $SOURCE_FACE — upload one via the dashboard after starting."
  SOURCE_FACE=""
fi

# Kill any previous server
pkill -f "python main.py" 2>/dev/null || true
sleep 1

echo "═══════════════════════════════════════════════"
echo " Starting DeepFace Live server"
[ -n "$SOURCE_FACE" ] && echo "  Source face: $SOURCE_FACE"
echo "  Dashboard:   http://localhost:8000"
echo "  WebSocket:   ws://localhost:8000/ws/deepfake"
echo "═══════════════════════════════════════════════"

cd "$PROJECT_DIR/server"
exec env \
  SOURCE_FACE="$SOURCE_FACE" \
  PYTHONPATH="$PROJECT_DIR/server" \
  python main.py
