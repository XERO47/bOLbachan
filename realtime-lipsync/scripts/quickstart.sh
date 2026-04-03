#!/usr/bin/env bash
# Run this once on the instance after git clone.
# Skips Docker — uses the pre-installed PyTorch 2.1.0 + CUDA 12.1 directly.
set -euo pipefail

echo "==> Cloning MuseTalk..."
git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk
export PYTHONPATH="/app/MuseTalk:$PYTHONPATH"

echo "==> Installing MuseTalk deps..."
pip install -q -r /app/MuseTalk/requirements.txt

echo "==> Installing server deps..."
pip install -q -r server/requirements.txt

echo "==> Downloading MuseTalk model weights (~3 GB)..."
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir models/musetalk \
  --exclude "*.md" "samples/*"

echo "==> Done. Start server with:"
echo "    bash scripts/run.sh"
