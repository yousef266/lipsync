import threading
import queue
import numpy as np
from typing import Callable
from ..core.types import Shape
from ..utils.dependencies import SOUNDDEVICE_AVAILABLE

if SOUNDDEVICE_AVAILABLE:
    import sounddevice as sd

class RealtimeProcessor:
    """Real-time audio processing for live lip sync"""

    def __init__(self, callback: Callable[[Shape], None]):
        self.callback = callback
        self.is_running = False
        self.audio_queue = queue.Queue()

        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not available for real-time processing")

    def start(self, sample_rate: int = 16000):
        """Start real-time processing"""
        self.is_running = True

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_queue.put(indata.copy())

        # Start audio stream
        self.stream = sd.InputStream(
            callback=audio_callback, channels=1, samplerate=sample_rate
        )
        self.stream.start()

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.start()

    def _process_loop(self):
        """Process audio in real-time"""
        buffer = np.array([], dtype=np.float32)

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, chunk.flatten()])

                # Process every 100ms
                if len(buffer) > 1600:  # ~100ms at 16kHz
                    energy = np.sqrt(np.mean(buffer[:1600] ** 2))

                    if energy > 0.01:
                        # Active speech - estimate shape
                        # Simplified: use energy-based heuristic
                        if energy > 0.1:
                            shape = Shape.D
                        elif energy > 0.05:
                            shape = Shape.C
                        else:
                            shape = Shape.B
                    else:
                        shape = Shape.X

                    self.callback(shape)
                    buffer = buffer[800:]  # Slide window

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def stop(self):
        """Stop real-time processing"""
        self.is_running = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, "process_thread"):
            self.process_thread.join()
