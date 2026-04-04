#!/usr/bin/env bash
# Start the avatar pipeline server. Run from anywhere.
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"

AVATAR_IMAGE="${AVATAR_IMAGE:-$MODELS_DIR/avatar.jpg}"

if [ ! -f "$AVATAR_IMAGE" ]; then
  echo "ERROR: Avatar image not found at $AVATAR_IMAGE"
  echo "Upload your face photo: scp face.jpg root@<IP>:$AVATAR_IMAGE"
  exit 1
fi

# Kill any previous server
pkill -f "python main.py" 2>/dev/null || true
sleep 1

echo "Starting AvatarPipeline server..."
echo "  Avatar: $AVATAR_IMAGE"
echo "  Lip sync: ${ENABLE_LIP_SYNC:-0}"

cd "$PROJECT_DIR/server"
exec env \
  MODEL_TYPE=avatar \
  AVATAR_IMAGE="$AVATAR_IMAGE" \
  ENABLE_LIP_SYNC="${ENABLE_LIP_SYNC:-0}" \
  PYTHONPATH="$PROJECT_DIR/server" \
  python main.py
