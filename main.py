#!/usr/bin/env python3
"""
Arabic Lip Sync System - Complete Implementation
A comprehensive Python system for automatic lip sync animation with Arabic language support
"""

import wave
import struct
import json
import threading
import queue
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from enum import Enum
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

# Try to import optional dependencies
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# ============================================================================
# Core Data Structures
# ============================================================================


class Shape(Enum):
    """Mouth shapes for animation (adapted for Arabic)"""

    A = "A"  # Closed mouth (م، ب، ف)
    B = "B"  # Slightly open, teeth visible (ت، د، ك)
    C = "C"  # Open mouth (ع، ح، ه)
    D = "D"  # Wide open (آ، أ، ا)
    E = "E"  # Rounded (و، ؤ)
    F = "F"  # Puckered lips (و)
    G = "G"  # F/V sound (ف، ث، ذ)
    H = "H"  # L sound (ل)
    X = "X"  # Idle/rest position


class Emotion(Enum):
    """Emotional states affecting lip sync"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"


class ArabicPhone(Enum):
    """Arabic phonemes"""

    # Short Vowels
    A = "a"  # فتحة
    I = "i"  # كسرة
    U = "u"  # ضمة

    # Long Vowels
    AA = "aa"  # ألف
    II = "ii"  # ياء
    UU = "uu"  # واو

    # Consonants
    B = "b"  # ب
    T = "t"  # ت
    TH = "th"  # ث
    J = "j"  # ج
    H = "h"  # ح
    KH = "kh"  # خ
    D = "d"  # د
    DH = "dh"  # ذ
    R = "r"  # ر
    Z = "z"  # ز
    S = "s"  # س
    SH = "sh"  # ش
    SS = "ss"  # ص
    DD = "dd"  # ض
    TT = "tt"  # ط
    DZ = "dz"  # ظ
    AIN = "ain"  # ع
    GH = "gh"  # غ
    F = "f"  # ف
    Q = "q"  # ق
    K = "k"  # ك
    L = "l"  # ل
    M = "m"  # م
    N = "n"  # ن
    W = "w"  # و
    Y = "y"  # ي
    HAMZA = "'"  # ء

    # Special
    SILENCE = "sil"
    NOISE = "noise"
    BREATH = "breath"


@dataclass
class TimeRange:
    """Time range in centiseconds"""

    start: int
    end: int

    def duration(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "TimeRange") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class TimedValue:
    """A value with associated time range"""

    start: int
    end: int
    value: any
    metadata: Dict = field(default_factory=dict)

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start, self.end)


# ============================================================================
# Audio Processing
# ============================================================================


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


# ============================================================================
# Arabic Text-to-Phoneme Conversion
# ============================================================================


class ArabicG2P:
    """Arabic Grapheme-to-Phoneme converter"""

    # Simplified mapping (in production, use a proper G2P model)
    LETTER_TO_PHONE = {
        "ا": ArabicPhone.AA,
        "أ": ArabicPhone.HAMZA,
        "إ": ArabicPhone.HAMZA,
        "آ": ArabicPhone.AA,
        "ء": ArabicPhone.HAMZA,
        "ب": ArabicPhone.B,
        "ت": ArabicPhone.T,
        "ث": ArabicPhone.TH,
        "ج": ArabicPhone.J,
        "ح": ArabicPhone.H,
        "خ": ArabicPhone.KH,
        "د": ArabicPhone.D,
        "ذ": ArabicPhone.DH,
        "ر": ArabicPhone.R,
        "ز": ArabicPhone.Z,
        "س": ArabicPhone.S,
        "ش": ArabicPhone.SH,
        "ص": ArabicPhone.SS,
        "ض": ArabicPhone.DD,
        "ط": ArabicPhone.TT,
        "ظ": ArabicPhone.DZ,
        "ع": ArabicPhone.AIN,
        "غ": ArabicPhone.GH,
        "ف": ArabicPhone.F,
        "ق": ArabicPhone.Q,
        "ك": ArabicPhone.K,
        "ل": ArabicPhone.L,
        "م": ArabicPhone.M,
        "ن": ArabicPhone.N,
        "ه": ArabicPhone.H,
        "و": ArabicPhone.W,
        "ي": ArabicPhone.Y,
    }

    @classmethod
    def text_to_phones(cls, text: str) -> List[ArabicPhone]:
        """Convert Arabic text to phoneme sequence"""
        phones = []

        for char in text:
            if char in cls.LETTER_TO_PHONE:
                phones.append(cls.LETTER_TO_PHONE[char])
            elif char in "َ":  # Fatha
                phones.append(ArabicPhone.A)
            elif char in "ِ":  # Kasra
                phones.append(ArabicPhone.I)
            elif char in "ُ":  # Damma
                phones.append(ArabicPhone.U)
            elif char == " ":
                phones.append(ArabicPhone.SILENCE)

        return phones


# ============================================================================
# Speech Recognition Integration
# ============================================================================


class SpeechRecognizer:
    """Speech recognition for Arabic"""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None

        if WHISPER_AVAILABLE:
            try:
                self.model = whisper.load_model(model_name)
            except:
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
            result = self.model.transcribe(
                str(audio_path), language="ar", word_timestamps=True
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


# ============================================================================
# Phoneme to Shape Mapping with Emotions
# ============================================================================


class ArabicShapeMapper:
    """Maps Arabic phonemes to mouth shapes with emotion support"""

    BASE_MAPPING = {
        # Vowels
        ArabicPhone.A: Shape.C,
        ArabicPhone.I: Shape.B,
        ArabicPhone.U: Shape.E,
        ArabicPhone.AA: Shape.D,
        ArabicPhone.II: Shape.B,
        ArabicPhone.UU: Shape.F,
        # Bilabials
        ArabicPhone.B: Shape.A,
        ArabicPhone.M: Shape.A,
        ArabicPhone.W: Shape.F,
        # Labiodentals
        ArabicPhone.F: Shape.G,
        # Dentals/Alveolars
        ArabicPhone.T: Shape.B,
        ArabicPhone.TH: Shape.G,
        ArabicPhone.D: Shape.B,
        ArabicPhone.DH: Shape.G,
        ArabicPhone.S: Shape.B,
        ArabicPhone.Z: Shape.B,
        ArabicPhone.SS: Shape.B,
        ArabicPhone.DD: Shape.B,
        ArabicPhone.TT: Shape.B,
        ArabicPhone.DZ: Shape.B,
        ArabicPhone.N: Shape.B,
        ArabicPhone.L: Shape.H,
        ArabicPhone.R: Shape.B,
        # Palatals
        ArabicPhone.J: Shape.B,
        ArabicPhone.SH: Shape.B,
        ArabicPhone.Y: Shape.B,
        # Velars/Uvulars
        ArabicPhone.K: Shape.B,
        ArabicPhone.Q: Shape.B,
        ArabicPhone.KH: Shape.B,
        ArabicPhone.GH: Shape.B,
        # Pharyngeals/Glottals
        ArabicPhone.H: Shape.C,
        ArabicPhone.AIN: Shape.D,
        ArabicPhone.HAMZA: Shape.C,
        # Special
        ArabicPhone.SILENCE: Shape.X,
        ArabicPhone.NOISE: Shape.X,
        ArabicPhone.BREATH: Shape.C,
    }

    # Emotion modifiers (affect shape intensity)
    EMOTION_MODIFIERS = {
        Emotion.HAPPY: {Shape.D: 1.2, Shape.C: 1.1},  # More open
        Emotion.SAD: {Shape.D: 0.8, Shape.C: 0.9},  # Less open
        Emotion.ANGRY: {Shape.B: 1.1, Shape.G: 1.2},  # More tense
        Emotion.SURPRISED: {Shape.D: 1.3},  # Very open
        Emotion.NEUTRAL: {},
    }

    @classmethod
    def get_shape(cls, phone: ArabicPhone, emotion: Emotion = Emotion.NEUTRAL) -> Shape:
        """Get shape for phoneme with emotion"""
        base_shape = cls.BASE_MAPPING.get(phone, Shape.B)
        # In production, apply emotion modifiers
        return base_shape

    @classmethod
    def get_tween_shape(cls, shape1: Shape, shape2: Shape) -> Optional[Shape]:
        """Get intermediate shape for smooth transition"""
        # Transitions that need intermediate shapes
        transitions = {
            (Shape.A, Shape.D): Shape.C,
            (Shape.D, Shape.A): Shape.C,
            (Shape.B, Shape.D): Shape.C,
            (Shape.F, Shape.D): Shape.E,
        }
        return transitions.get((shape1, shape2))


# ============================================================================
# Animation Timeline with Tweening
# ============================================================================


class Timeline:
    """Timeline with smooth transitions"""

    def __init__(self):
        self.elements: List[TimedValue] = []

    def add(self, start: int, end: int, value: any, metadata: Dict = None):
        """Add a timed value"""
        self.elements.append(TimedValue(start, end, value, metadata or {}))

    def get_at(self, time_cs: int) -> Optional[any]:
        """Get value at specific time"""
        for element in self.elements:
            if element.start <= time_cs < element.end:
                return element.value
        return None

    def optimize(self):
        """Merge adjacent elements with same value"""
        if not self.elements:
            return

        self.elements.sort(key=lambda x: x.start)
        optimized = [self.elements[0]]

        for element in self.elements[1:]:
            last = optimized[-1]
            if last.value == element.value and last.end == element.start:
                last.end = element.end
                last.metadata.update(element.metadata)
            else:
                optimized.append(element)

        self.elements = optimized

    def add_tweening(self, min_tween_duration_cs: int = 4):
        """Add smooth transitions between shapes"""
        if len(self.elements) < 2:
            return

        new_elements = [self.elements[0]]

        for i in range(1, len(self.elements)):
            prev = new_elements[-1]
            curr = self.elements[i]

            # Check if we need a tween
            tween_shape = ArabicShapeMapper.get_tween_shape(prev.value, curr.value)

            if tween_shape and curr.start - prev.end < 1:
                # Add tween
                tween_duration = min(
                    min_tween_duration_cs, (curr.end - prev.start) // 3
                )

                if tween_duration >= min_tween_duration_cs:
                    tween_start = curr.start
                    tween_end = curr.start + tween_duration

                    new_elements.append(
                        TimedValue(tween_start, tween_end, tween_shape, {"tween": True})
                    )

                    # Adjust current element
                    curr.start = tween_end

            new_elements.append(curr)

        self.elements = new_elements


# ============================================================================
# Real-time Audio Processing
# ============================================================================


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


# ============================================================================
# Main Lip Sync Engine
# ============================================================================


class ArabicLipSyncEngine:
    """Main engine for Arabic lip sync generation"""

    def __init__(self):
        self.shape_mapper = ArabicShapeMapper()
        self.g2p = ArabicG2P()
        self.recognizer = SpeechRecognizer() if WHISPER_AVAILABLE else None

    def process_audio(
        self,
        audio_path: Path,
        text: Optional[str] = None,
        emotion: Emotion = Emotion.NEUTRAL,
        use_recognition: bool = True,
        add_tweening: bool = True,
    ) -> Timeline:
        """
        Process audio file and generate lip sync animation

        Args:
            audio_path: Path to audio file
            text: Optional Arabic text (if not using recognition)
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
        elif text:
            phones = self._text_to_timed_phones(text, audio)
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
        """Use speech recognition to get phoneme timing"""
        transcript, word_segments = self.recognizer.transcribe(audio_path)

        if not word_segments:
            return self._estimate_phones(audio)

        phones = []
        for segment in word_segments:
            word = segment["word"].strip()
            start_sec = segment["start"]
            end_sec = segment["end"]

            # Convert word to phones
            word_phones = self.g2p.text_to_phones(word)

            if word_phones:
                # Distribute time evenly across phones
                duration_cs = int((end_sec - start_sec) * 100)
                phone_duration = duration_cs // len(word_phones)

                for i, phone in enumerate(word_phones):
                    phone_start = int(start_sec * 100) + i * phone_duration
                    phone_end = phone_start + phone_duration
                    phones.append((phone, phone_start, phone_end))

        return phones

    def _text_to_timed_phones(
        self, text: str, audio: AudioClip
    ) -> List[Tuple[ArabicPhone, int, int]]:
        """Convert text to timed phonemes using voice activity"""
        phones_list = self.g2p.text_to_phones(text)
        voice_segments = audio.detect_voice_activity()

        if not voice_segments or not phones_list:
            return []

        # Distribute phones across voice segments
        total_duration = sum(seg.duration() for seg in voice_segments)

        phones = []
        phone_idx = 0

        for segment in voice_segments:
            seg_phone_count = int(
                len(phones_list) * segment.duration() / total_duration
            )
            if seg_phone_count == 0:
                continue

            phone_duration = segment.duration() // seg_phone_count

            for i in range(seg_phone_count):
                if phone_idx >= len(phones_list):
                    break

                phone_start = segment.start + i * phone_duration
                phone_end = phone_start + phone_duration
                phones.append((phones_list[phone_idx], phone_start, phone_end))
                phone_idx += 1

        return phones

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

    def export_tsv(self, timeline: Timeline, output_path: Path):
        """Export animation to TSV format"""
        with open(output_path, "w", encoding="utf-8") as f:
            for elem in timeline.elements:
                f.write(f"{elem.start / 100.0:.2f}\t{elem.value.value}\n")

            if timeline.elements:
                last = timeline.elements[-1]
                f.write(f"{last.end / 100.0:.2f}\tX\n")


# ============================================================================
# GUI Application
# ============================================================================


class LipSyncGUI:
    """Graphical user interface for Arabic Lip Sync"""

    def __init__(self):
        self.engine = ArabicLipSyncEngine()
        self.audio_path = None
        self.timeline = None

        self.root = tk.Tk()
        self.root.title("Arabic Lip Sync Generator")
        self.root.geometry("800x600")

        self._create_widgets()

    def _create_widgets(self):
        """Create GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File selection
        ttk.Label(main_frame, text="Audio File:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.file_label = ttk.Label(
            main_frame, text="No file selected", foreground="gray"
        )
        self.file_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(main_frame, text="Browse", command=self._browse_file).grid(
            row=0, column=2, padx=5
        )

        # Text input
        ttk.Label(main_frame, text="Arabic Text (optional):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.text_input = tk.Text(main_frame, height=3, width=50, font=("Arial", 12))
        self.text_input.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)

        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.use_recognition = tk.BooleanVar(value=WHISPER_AVAILABLE)
        ttk.Checkbutton(
            options_frame,
            text="Use Speech Recognition (Whisper)",
            variable=self.use_recognition,
            state="normal" if WHISPER_AVAILABLE else "disabled",
        ).grid(row=0, column=0, sticky=tk.W)

        self.use_tweening = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Add Smooth Transitions (Tweening)",
            variable=self.use_tweening,
        ).grid(row=1, column=0, sticky=tk.W)

        ttk.Label(options_frame, text="Emotion:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.emotion_var = tk.StringVar(value="neutral")
        emotion_combo = ttk.Combobox(
            options_frame,
            textvariable=self.emotion_var,
            values=["neutral", "happy", "sad", "angry", "surprised"],
            state="readonly",
            width=15,
        )
        emotion_combo.grid(row=2, column=1, sticky=tk.W, padx=5)

        # Process button
        self.process_btn = ttk.Button(
            main_frame,
            text="Generate Lip Sync",
            command=self._process,
            state="disabled",
        )
        self.process_btn.grid(row=3, column=0, columnspan=3, pady=15)

        # Progress
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Status
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(
            row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10
        )

        self.results_text = tk.Text(
            results_frame, height=10, width=70, state="disabled"
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.results_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Export buttons
        export_frame = ttk.Frame(main_frame)
        export_frame.grid(row=7, column=0, columnspan=3, pady=10)

        self.export_json_btn = ttk.Button(
            export_frame,
            text="Export JSON",
            command=self._export_json,
            state="disabled",
        )
        self.export_json_btn.pack(side=tk.LEFT, padx=5)

        self.export_tsv_btn = ttk.Button(
            export_frame, text="Export TSV", command=self._export_tsv, state="disabled"
        )
        self.export_tsv_btn.pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

    def _browse_file(self):
        """Browse for audio file"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )

        if filename:
            self.audio_path = Path(filename)
            self.file_label.config(text=self.audio_path.name, foreground="black")
            self.process_btn.config(state="normal")

    def _process(self):
        """Process audio file"""
        if not self.audio_path:
            return

        self.process_btn.config(state="disabled")
        self.export_json_btn.config(state="disabled")
        self.export_tsv_btn.config(state="disabled")
        self.progress.start()
        self.status_label.config(text="Processing...", foreground="orange")

        # Run in separate thread
        thread = threading.Thread(target=self._process_thread)
        thread.start()

    def _process_thread(self):
        """Processing thread"""
        try:
            text = self.text_input.get("1.0", tk.END).strip()
            emotion = Emotion[self.emotion_var.get().upper()]

            self.timeline = self.engine.process_audio(
                self.audio_path,
                text=text if text else None,
                emotion=emotion,
                use_recognition=self.use_recognition.get(),
                add_tweening=self.use_tweening.get(),
            )

            # Update UI
            self.root.after(0, self._process_complete)

        except Exception as e:
            self.root.after(0, lambda: self._process_error(str(e)))

    def _process_complete(self):
        """Processing completed"""
        self.progress.stop()
        self.status_label.config(text="Complete!", foreground="green")
        self.process_btn.config(state="normal")
        self.export_json_btn.config(state="normal")
        self.export_tsv_btn.config(state="normal")

        # Display results
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)

        self.results_text.insert(
            tk.END, f"Generated {len(self.timeline.elements)} mouth shape segments:\n\n"
        )

        for elem in self.timeline.elements[:20]:  # Show first 20
            self.results_text.insert(
                tk.END,
                f"{elem.start/100:.2f}s - {elem.end/100:.2f}s: {elem.value.value}\n",
            )

        if len(self.timeline.elements) > 20:
            self.results_text.insert(
                tk.END, f"\n... and {len(self.timeline.elements) - 20} more segments"
            )

        self.results_text.config(state="disabled")

    def _process_error(self, error: str):
        """Processing error"""
        self.progress.stop()
        self.status_label.config(text="Error!", foreground="red")
        self.process_btn.config(state="normal")
        messagebox.showerror("Processing Error", f"Error: {error}")

    def _export_json(self):
        """Export to JSON"""
        if not self.timeline:
            return

        filename = filedialog.asksaveasfilename(
            title="Save JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if filename:
            audio = AudioClip.from_wav(self.audio_path)
            self.engine.export_json(
                self.timeline, Path(filename), audio.get_duration_cs()
            )
            messagebox.showinfo("Success", f"Exported to {filename}")

    def _export_tsv(self):
        """Export to TSV"""
        if not self.timeline:
            return

        filename = filedialog.asksaveasfilename(
            title="Save TSV",
            defaultextension=".tsv",
            filetypes=[("TSV files", "*.tsv"), ("All files", "*.*")],
        )

        if filename:
            self.engine.export_tsv(self.timeline, Path(filename))
            messagebox.showinfo("Success", f"Exported to {filename}")

    def run(self):
        """Run the GUI"""
        self.root.mainloop()


# ============================================================================
# Example Usage
# ============================================================================


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Arabic Lip Sync Generator")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--audio", type=str, help="Input audio file")
    parser.add_argument("--text", type=str, help="Arabic text")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument(
        "--format", choices=["json", "tsv"], default="json", help="Output format"
    )

    args = parser.parse_args()

    if args.gui or (not args.audio):
        # Launch GUI
        app = LipSyncGUI()
        app.run()
    else:
        # Command-line mode
        engine = ArabicLipSyncEngine()
        audio_path = Path(args.audio)

        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            return

        print(f"Processing {audio_path}...")
        timeline = engine.process_audio(
            audio_path, text=args.text, use_recognition=WHISPER_AVAILABLE
        )

        # Export
        output_path = (
            Path(args.output)
            if args.output
            else audio_path.with_suffix(f".{args.format}")
        )

        audio = AudioClip.from_wav(audio_path)
        if args.format == "json":
            engine.export_json(timeline, output_path, audio.get_duration_cs())
        else:
            engine.export_tsv(timeline, output_path)

        print(f"Generated {len(timeline.elements)} mouth shape segments")
        print(f"Exported to {output_path}")


if __name__ == "__main__":
    main()
