import json
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from ..core.types import ArabicPhone, Shape, Emotion
from ..core.models import TimeRange
from ..audio.clip import AudioClip
from ..linguistics.g2p import ArabicG2P
from ..linguistics.mapper import ArabicShapeMapper
from ..linguistics.diacritizer import ArabicDiacritizer
from ..recognition.whisper_rec import SpeechRecognizer
from ..engine.timeline import Timeline
from ..utils.dependencies import WHISPER_AVAILABLE


class ArabicLipSyncEngine:
    """Main engine for Arabic lip sync generation"""

    def __init__(self, diacritizer_backend: str = "auto"):
        self.shape_mapper = ArabicShapeMapper()
        self.g2p = ArabicG2P()
        self.diacritizer = ArabicDiacritizer(backend=diacritizer_backend)
        self.recognizer = SpeechRecognizer() if WHISPER_AVAILABLE else None

    def process_audio(
        self,
        audio_path: Path,
        emotion: Emotion = Emotion.NEUTRAL,
        use_recognition: bool = True,
        add_tweening: bool = True,
    ) -> Timeline:
        """
        Process audio file and generate lip sync animation

        Args:
            audio_path: Path to audio file
            emotion: Emotional state
            use_recognition: Use speech recognition
            add_tweening: Add smooth transitions

        Returns:
            Timeline of mouth shapes
        """
        # Load audio
        audio = AudioClip.from_wav(audio_path)

        # Get phoneme timing
        if use_recognition and self.recognizer:
            phones = self._recognize_phones(audio_path, audio)
        else:
            phones = self._estimate_phones(audio)

        # Generate animation timeline
        timeline = Timeline()

        for phone, start_cs, end_cs in phones:
            shape = self.shape_mapper.get_shape(phone, emotion)
            timeline.add(start_cs, end_cs, shape, {"phone": phone.value})

        # Add tweening for smooth transitions
        if add_tweening:
            timeline.add_tweening()

        # Optimize timeline
        timeline.optimize()

        return timeline

    def _recognize_phones(
        self, audio_path: Path, audio: AudioClip
    ) -> List[Tuple[ArabicPhone, int, int]]:
        """Use speech recognition to get phoneme timing.

        Each word from Whisper is first passed through the neural diacritizer
        so that the rule-based G2P receives fully-vocalized text.
        """
        transcript, word_segments = self.recognizer.transcribe(audio_path)

        if not word_segments:
            return self._estimate_phones(audio)

        # Diacritize the full transcript in one pass for best neural context,
        # then align the diacritized tokens back to Whisper word segments.
        diac_words = self._diacritize_segments(
            transcript,
            [seg["word"].strip() for seg in word_segments],
        )

        phones = []
        for idx, segment in enumerate(word_segments):
            word = diac_words[idx]
            start_sec = segment["start"]
            end_sec = segment["end"]

            # Convert diacritized word to phones
            word_phones = self.g2p.text_to_phones(word)

            if word_phones:
                # Distribute time evenly across phones in the word
                duration_cs = int((end_sec - start_sec) * 100)
                phone_duration = max(1, duration_cs // len(word_phones))

                for i, phone in enumerate(word_phones):
                    phone_start = int(start_sec * 100) + i * phone_duration
                    phone_end = phone_start + phone_duration
                    phones.append((phone, phone_start, phone_end))

        return phones

    def _diacritize_segments(
        self, transcript: str, raw_words: List[str]
    ) -> List[str]:
        """Return a diacritized word list aligned 1-to-1 with raw_words.

        Strategy:
          1. Diacritize the full transcript (better context for the model).
          2. If the token count matches, use those tokens directly.
          3. Otherwise fall back to per-word diacritization.
        """
        diac_full = self.diacritizer.diacritize(transcript)
        diac_tokens = diac_full.split()

        if len(diac_tokens) == len(raw_words):
            return diac_tokens

        # Fallback: diacritize each word independently
        return [self.diacritizer.diacritize(w) for w in raw_words]

    def _estimate_phones(self, audio: AudioClip) -> List[Tuple[ArabicPhone, int, int]]:
        """Estimate phonemes from audio energy"""
        voice_segments = audio.detect_voice_activity()

        phones = []
        for segment in voice_segments:
            duration = segment.duration()
            mid_point = segment.start + duration // 2

            # Simple alternating pattern
            phones.append((ArabicPhone.K, segment.start, mid_point))
            phones.append((ArabicPhone.AA, mid_point, segment.end))

        return phones

    def export_json(
        self, timeline: Timeline, output_path: Path, audio_duration_cs: int
    ):
        """Export animation to JSON format"""
        data = {
            "metadata": {
                "duration": audio_duration_cs / 100.0,
                "format": "arabic_lip_sync_v1",
                "timestamp": datetime.now().isoformat(),
            },
            "mouthCues": [
                {
                    "start": elem.start / 100.0,
                    "end": elem.end / 100.0,
                    "value": elem.value.value,
                    "metadata": elem.metadata,
                }
                for elem in timeline.elements
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
