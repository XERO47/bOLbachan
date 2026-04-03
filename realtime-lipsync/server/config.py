from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # Model
    MODEL_TYPE: Literal["musetalk", "wav2lip"] = "musetalk"
    MODEL_DIR: str = "/app/models/musetalk"
    DEVICE: str = "cuda"

    # Stream
    TARGET_FPS: int = 25
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_MS: int = 40        # 40ms audio chunks = 25fps alignment
    FRAME_WIDTH: int = 512
    FRAME_HEIGHT: int = 512
    JPEG_QUALITY: int = 85

    # Pipeline tuning
    FACE_DET_INTERVAL: int = 5      # Re-detect face every N frames
    INFERENCE_BATCH: int = 1        # Batch size (1 for lowest latency)
    HALF_PRECISION: bool = True     # FP16 for L40S (faster, same quality)

    # Server
    MAX_CONNECTIONS: int = 10
    LOG_LEVEL: str = "info"

    class Config:
        env_file = ".env"


settings = Settings()
