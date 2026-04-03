#!/usr/bin/env bash
# Download MuseTalk model weights from HuggingFace.
# Run this once before building the Docker image, or mount the models/ volume.

set -euo pipefail

MODELS_DIR="${1:-./models}"
MUSETALK_DIR="$MODELS_DIR/musetalk"

echo "Downloading MuseTalk weights to $MUSETALK_DIR ..."
mkdir -p "$MUSETALK_DIR"

pip install -q huggingface_hub

python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download

dest = os.environ.get("MUSETALK_DIR", "./models/musetalk")
snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    local_dir=dest,
    ignore_patterns=["*.md", "*.txt", "samples/*"]
)
print(f"Models saved to {dest}")
EOF

echo "Done. Model dir contents:"
ls -lh "$MUSETALK_DIR"
