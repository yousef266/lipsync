from pathlib import Path
from typing import Tuple, List, Dict
from ..utils.dependencies import WHISPER_AVAILABLE

if WHISPER_AVAILABLE:
    import whisper

class SpeechRecognizer:
    """Speech recognition for Arabic"""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self.device = "cpu"

        if WHISPER_AVAILABLE:
            try:
                # Try to use GPU if available for faster processing
                try:
                    import torch
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    self.device = "cpu"
                
                # Load the model once
                self.model = whisper.load_model(model_name, device=self.device)
            except Exception as e:
                print(f"Error loading whisper model: {e}")
                pass

    def transcribe(self, audio_path: Path) -> Tuple[str, List[Dict]]:
        """
        Transcribe audio and get word-level timestamps

        Returns:
            (transcript, word_segments)
        """
        if self.model is None:
            return "", []

        try:
            # Use faster transcription settings
            result = self.model.transcribe(
                str(audio_path), 
                language="ar", 
                word_timestamps=True,
                beam_size=1,        # Faster processing (default is 5)
                best_of=1,          # Faster processing (default is 5)
                fp16=(self.device == "cuda"), # Use fp16 only on GPU
                condition_on_previous_text=False, # Faster for short clips
                compression_ratio_threshold=None, # Skip some checks
                logprob_threshold=None            # Skip some checks
            )

            transcript = result.get("text", "")
            segments = result.get("segments", [])

            word_segments = []
            for segment in segments:
                for word_info in segment.get("words", []):
                    word_segments.append(
                        {
                            "word": word_info.get("word", ""),
                            "start": word_info.get("start", 0),
                            "end": word_info.get("end", 0),
                        }
                    )

            return transcript, word_segments
        except Exception as e:
            print(f"Recognition error: {e}")
            return "", []
