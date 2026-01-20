import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from pathlib import Path

from ..engine.lip_sync import ArabicLipSyncEngine
from ..audio.clip import AudioClip
from ..core.types import Emotion
from ..utils.dependencies import WHISPER_AVAILABLE

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

        # Options (Tweening and Whisper are now mandatory and hidden from UI)
        self.use_recognition = tk.BooleanVar(value=WHISPER_AVAILABLE)
        self.use_tweening = tk.BooleanVar(value=True)

        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(options_frame, text="Emotion:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.emotion_var = tk.StringVar(value="neutral")
        emotion_combo = ttk.Combobox(
            options_frame,
            textvariable=self.emotion_var,
            values=["neutral", "happy", "sad", "angry", "surprised"],
            state="readonly",
            width=15,
        )
        emotion_combo.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Process button
        self.process_btn = ttk.Button(
            main_frame,
            text="Generate Lip Sync",
            command=self._process,
            state="disabled",
        )
        self.process_btn.grid(row=2, column=0, columnspan=3, pady=15)

        # Progress
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Status
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(
            row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10
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
        export_frame.grid(row=6, column=0, columnspan=3, pady=10)

        self.export_json_btn = ttk.Button(
            export_frame,
            text="Export JSON",
            command=self._export_json,
            state="disabled",
        )
        self.export_json_btn.pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
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
        self.progress.start()
        self.status_label.config(text="Processing...", foreground="orange")
        self.start_processing_time = time.time()

        # Run in separate thread
        thread = threading.Thread(target=self._process_thread)
        thread.start()

    def _process_thread(self):
        """Processing thread"""
        try:
            emotion = Emotion[self.emotion_var.get().upper()]

            self.timeline = self.engine.process_audio(
                self.audio_path,
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
        duration = time.time() - self.start_processing_time
        self.progress.stop()
        self.status_label.config(
            text=f"Complete! (Took {duration:.2f} seconds)", 
            foreground="green"
        )
        self.process_btn.config(state="normal")
        self.export_json_btn.config(state="normal")

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

    def run(self):
        """Run the GUI"""
        self.root.mainloop()
