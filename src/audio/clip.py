import wave
import numpy as np
from typing import List
from pathlib import Path
from ..core.models import TimeRange

class AudioClip:
    """Audio clip with processing capabilities"""

    def __init__(self, sample_rate: int, samples: np.ndarray):
        self.sample_rate = sample_rate
        self.samples = samples  # Mono, float32, range [-1, 1]

    @classmethod
    def from_wav(cls, filepath: Path) -> "AudioClip":
        """Load audio from WAV file"""
        with wave.open(str(filepath), "rb") as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            n_frames = wav.getnframes()
            sample_width = wav.getsampwidth()

            raw_data = wav.readframes(n_frames)

            if sample_width == 1:
                samples = np.frombuffer(raw_data, dtype=np.uint8)
                samples = (samples.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16)
                samples = samples.astype(np.float32) / 32768.0
            elif sample_width == 4:
                samples = np.frombuffer(raw_data, dtype=np.int32)
                samples = samples.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels)
                samples = samples.mean(axis=1)

            return cls(sample_rate, samples)

    def get_duration_cs(self) -> int:
        """Get duration in centiseconds"""
        return int(len(self.samples) * 100 / self.sample_rate)

    def detect_voice_activity(self, frame_duration_ms: int = 30) -> List[TimeRange]:
        """Energy-based voice activity detection with smoothing"""
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        energies = []

        for i in range(0, len(self.samples), frame_size):
            frame = self.samples[i : i + frame_size]
            if len(frame) > 0:
                energy = np.sqrt(np.mean(frame**2))
                energies.append(energy)

        if not energies:
            return []

        # Adaptive threshold
        energies = np.array(energies)
        threshold = np.percentile(energies, 40)

        # Smooth with moving average
        window_size = 5
        smoothed = np.convolve(
            energies, np.ones(window_size) / window_size, mode="same"
        )

        voice_segments = []
        in_voice = False
        start_frame = 0

        for i, energy in enumerate(smoothed):
            if energy > threshold and not in_voice:
                start_frame = i
                in_voice = True
            elif energy <= threshold and in_voice:
                start_cs = int(start_frame * frame_duration_ms / 10)
                end_cs = int(i * frame_duration_ms / 10)
                if end_cs - start_cs > 10:  # Minimum 100ms
                    voice_segments.append(TimeRange(start_cs, end_cs))
                in_voice = False

        # Merge close segments
        merged = []
        for segment in voice_segments:
            if merged and segment.start - merged[-1].end < 20:  # < 200ms gap
                merged[-1].end = segment.end
            else:
                merged.append(segment)

        return merged
