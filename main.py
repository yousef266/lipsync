#!/usr/bin/env python3
"""
Arabic Lip Sync System - Entry Point
"""

import argparse
import time
from pathlib import Path
from src.ui.gui import LipSyncGUI
from src.engine.lip_sync import ArabicLipSyncEngine
from src.audio.clip import AudioClip
from src.utils.dependencies import WHISPER_AVAILABLE


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Arabic Lip Sync Generator")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--audio", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, help="Output file")

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
        start_time = time.time()
        
        timeline = engine.process_audio(
            audio_path, 
            use_recognition=WHISPER_AVAILABLE,
            add_tweening=True
        )
        
        duration = time.time() - start_time
        print(f"Done! Generation took {duration:.2f} seconds")

        # Export
        output_path = (
            Path(args.output)
            if args.output
            else audio_path.with_suffix(".json")
        )

        audio = AudioClip.from_wav(audio_path)
        engine.export_json(timeline, output_path, audio.get_duration_cs())

        print(f"Generated {len(timeline.elements)} mouth shape segments")
        print(f"Exported to {output_path}")


if __name__ == "__main__":
    main()
