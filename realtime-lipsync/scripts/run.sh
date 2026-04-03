#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/app/MuseTalk:$PYTHONPATH"
export MODEL_DIR="$(pwd)/models/musetalk"

cd "$(dirname "$0")/.."
python3 -m uvicorn server.main:app \
  --host 0.0.0.0 --port 8000 \
  --log-level info
