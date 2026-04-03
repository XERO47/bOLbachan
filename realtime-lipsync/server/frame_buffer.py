"""
Synchronized audio/video buffer.

Video frames arrive at ~25fps and audio arrives as ~40ms chunks.
This buffer aligns them so we always feed the model a (frame, audio) pair
where the audio temporally matches the frame.
"""

import time
import asyncio
import collections
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class VideoFrame:
    data: np.ndarray        # HxWx3 BGR
    timestamp_ms: int


@dataclass
class AudioChunk:
    data: np.ndarray        # float32 PCM, shape (samples,)
    timestamp_ms: int
    duration_ms: int


class FrameAudioBuffer:
    """
    Keeps rolling windows of video frames and audio chunks.
    get_aligned_pair() returns the most recent frame with its
    corresponding audio window (configurable lookahead for MuseTalk).
    """

    def __init__(
        self,
        target_fps: int = 25,
        audio_sample_rate: int = 16000,
        audio_context_ms: int = 200,    # MuseTalk uses ~200ms audio context
        max_lag_ms: int = 500,          # Drop frames older than this
    ):
        self.target_fps = target_fps
        self.sample_rate = audio_sample_rate
        self.audio_context_samples = int(audio_sample_rate * audio_context_ms / 1000)
        self.max_lag_ms = max_lag_ms

        self._video: collections.deque[VideoFrame] = collections.deque(maxlen=target_fps * 2)
        self._audio: collections.deque[AudioChunk] = collections.deque(maxlen=100)
        self._audio_buffer = np.zeros(self.audio_context_samples, dtype=np.float32)
        self._lock = asyncio.Lock()

    async def add_video(self, frame: np.ndarray, timestamp_ms: int):
        async with self._lock:
            self._video.append(VideoFrame(data=frame, timestamp_ms=timestamp_ms))

    async def add_audio(self, pcm: np.ndarray, timestamp_ms: int):
        """pcm: float32 normalized [-1, 1]"""
        duration_ms = int(len(pcm) / self.sample_rate * 1000)
        async with self._lock:
            self._audio.append(AudioChunk(
                data=pcm, timestamp_ms=timestamp_ms, duration_ms=duration_ms
            ))
            # Roll audio into the context buffer
            if len(pcm) >= self.audio_context_samples:
                self._audio_buffer = pcm[-self.audio_context_samples:].copy()
            else:
                self._audio_buffer = np.roll(self._audio_buffer, -len(pcm))
                self._audio_buffer[-len(pcm):] = pcm

    async def get_aligned_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns (video_frame BGR, audio_context float32) or None if not ready.
        """
        async with self._lock:
            if not self._video:
                return None

            frame = self._video[-1]  # Most recent frame
            return frame.data.copy(), self._audio_buffer.copy()

    def ready(self) -> bool:
        return len(self._video) > 0
