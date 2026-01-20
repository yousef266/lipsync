import asyncio
import edge_tts
import json
import os
import sys

# === Setup ===
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def generate_tts_with_visemes(text, voice="en-US-AriaNeural"):
    """
    Generate speech and viseme data from text using Edge TTS.
    Compatible with 3D talking head synchronization.
    """

    print(f"\n🗣️ Generating speech using voice: {voice}\n")
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="+0%",
        pitch="+0Hz"
    )

    viseme_data = []
    audio_path = os.path.join(OUTPUT_DIR, "output.mp3")
    viseme_path = os.path.join(OUTPUT_DIR, "visemes.json")

    # Remove any old files
    if os.path.exists(audio_path):
        os.remove(audio_path)

    # === Stream the TTS data ===
    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "viseme":
                viseme_data.append({
                    "viseme_id": chunk["value"],
                    "time_ms": chunk["offset"]
                })
                print(f"Viseme {chunk['value']} at {chunk['offset']} ms")

            elif chunk["type"] == "audio" and chunk["data"]:
                with open(audio_path, "ab") as f:
                    f.write(chunk["data"])

    except Exception as e:
        print(f"\n❌ Error during TTS generation: {e}")
        sys.exit(1)

    # === Save Results ===
    if viseme_data:
        with open(viseme_path, "w", encoding="utf-8") as f:
            json.dump(viseme_data, f, indent=2)
        print(f"\n✅ Viseme data saved to: {viseme_path}")
    else:
        print("   • Temporary Microsoft Edge TTS issue\n")

    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        print(f"✅ Audio saved to: {audio_path}")
    else:
        print("⚠️ No audio data received.")


async def list_arabic_voices():
    """
    Query Edge TTS voices and return `ShortName`s of Arabic voices.
    This keeps the code resilient to available voices on the running machine
    or account/region differences.
    """
    try:
        voices = await edge_tts.list_voices()
    except Exception as e:
        print(f"Could not list voices: {e}")
        return []

    arabic = [v for v in voices if v.get("Locale", "").startswith("ar")]
    if not arabic:
        print("No Arabic voices found via edge_tts.list_voices().")
        return []

    print("\n🔎 Found Arabic voices:")
    for v in arabic:
        print(f"  {v.get('ShortName')}  —  {v.get('Locale')}  —  {v.get('Name')}")

    return [v.get("ShortName") for v in arabic]

async def main():
    # Arabic test text
    text = "مرحبا! كيف حالك اليوم؟ أتمنى أن تكون بخير."

    # Discover Arabic voices at runtime
    arabic_voices = await list_arabic_voices()
    if not arabic_voices:
        # Fallback: keep original known voices if no Arabic voices are available
        print("Falling back to English demo voices")
        voices = ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural"]
    else:
        # Use up to the first 3 Arabic voices found (adjust as needed)
        voices = arabic_voices[:3]

    for voice in voices:
        await generate_tts_with_visemes(text, voice)
        print("\n" + "="*50 + "\n")

asyncio.run(main())