# L40S is Ada Lovelace (sm_89) — CUDA 12.1+ required
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.9"
# L40S compute capability

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    git git-lfs wget curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libsndfile1 portaudio19-dev \
    v4l2loopback-dkms v4l2loopback-utils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# PyTorch with CUDA 12.1 (optimized for Ada Lovelace)
RUN pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    torchaudio==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

# Clone MuseTalk
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk
RUN pip install --no-cache-dir -r /app/MuseTalk/requirements.txt

# Install server dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download models (MuseTalk weights via huggingface)
RUN pip install --no-cache-dir huggingface_hub
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
