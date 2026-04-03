# Base: NVIDIA NGC PyTorch 23.07 — PyTorch 2.1.0, Python 3.10, CUDA 12.1, Ubuntu 22.04
# Matches the pre-installed environment on the L40S instance exactly.
# PyTorch, torchvision, torchaudio, cuDNN, NCCL are already baked in — do NOT reinstall.
FROM nvcr.io/nvidia/pytorch:23.07-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.9"
# L40S is Ada Lovelace — sm_89

# System dependencies only (Python 3.10 + pip already present in NGC image)
RUN apt-get update && apt-get install -y \
    git git-lfs wget curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libsndfile1 portaudio19-dev \
    v4l2loopback-dkms v4l2loopback-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk
RUN pip install --no-cache-dir -r /app/MuseTalk/requirements.txt

# Install server dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download models (MuseTalk weights via huggingface)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('TMElyralab/MuseTalk', local_dir='/app/models/musetalk')"

# Copy app code
COPY server/ /app/server/
COPY client/ /app/client/

ENV PYTHONPATH="/app/MuseTalk:$PYTHONPATH"
ENV MODEL_DIR="/app/models/musetalk"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
