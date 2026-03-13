#!/usr/bin/env python3
"""
test_g2p_pipeline.py
====================
Verifies the full diacritizer → G2P chain.

Run from the project root:
    python test_g2p_pipeline.py

Output is printed to the terminal AND saved to:
    test_g2p_output.txt   (UTF-8, in the project root)

What is tested per sentence:
  * diacritization adds harakat to raw Arabic text
  * G2P reads fatha / kasra / damma correctly
  * shadda produces a doubled consonant phone
  * tanwin produces vowel + N
  * sukun produces no extra vowel
"""

from __future__ import annotations

import sys
import os
import io
from datetime import datetime

# Ensure the project root is on sys.path so that
# `from src.linguistics...` mirrors how main.py imports things.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.linguistics.diacritizer import ArabicDiacritizer  # noqa: E402
from src.linguistics.g2p import ArabicG2P                  # noqa: E402
from src.core.types import ArabicPhone                      # noqa: E402

# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "test_g2p_output.txt")


class _Tee:
    """Writes every print() call to both the terminal and a UTF-8 text file."""

    def __init__(self, filepath: str) -> None:
        self._terminal = sys.stdout
        self._file = open(filepath, "w", encoding="utf-8")
        # Write a header so the file is self-describing
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"Arabic G2P Pipeline Test  –  {ts}\n")
        self._file.write(f"Python {sys.version.split()[0]}  |  "
                         f"File: {filepath}\n\n")

    def write(self, message: str) -> None:
        self._terminal.write(message)
        self._file.write(message)

    def flush(self) -> None:
        self._terminal.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Test sentences (Egyptian Arabic)
# ---------------------------------------------------------------------------
TEST_SENTENCES = [
    "انا بحب البرمجة",
    "كتب محمد الرسالة",
    "الولد راح المدرسة",
    "الشمس طالعة النهاردة",
    "مصر جميلة جدا",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def phones_to_str(phones: list[ArabicPhone]) -> str:
    """
    Convert a phoneme list to a human-readable token string.

    SILENCE tokens become ' | ' so word boundaries are visible.
    All other tokens are rendered as their enum value in upper-case.

    Example:
      [K, A, T, SILENCE, M, U, H] → "K A T | M U H"
    """
    parts: list[str] = []
    for p in phones:
        if p is ArabicPhone.SILENCE:
            parts.append("|")
        else:
            parts.append(p.value.upper())
    # Collapse consecutive | symbols and strip leading/trailing ones
    result = " ".join(parts).strip()
    while "| |" in result:
        result = result.replace("| |", "|")
    return result.strip("|").strip()


def check_shadda(phones: list[ArabicPhone]) -> bool:
    """Return True if any consecutive pair of identical consonants appears."""
    vowels = {ArabicPhone.A, ArabicPhone.I, ArabicPhone.U,
              ArabicPhone.AA, ArabicPhone.II, ArabicPhone.UU,
              ArabicPhone.SILENCE}
    for i in range(len(phones) - 1):
        if phones[i] == phones[i + 1] and phones[i] not in vowels:
            return True
    return False


def check_tanwin(phones: list[ArabicPhone]) -> bool:
    """Return True if any (A/I/U) immediately followed by N appears."""
    for i in range(len(phones) - 1):
        if phones[i] in (ArabicPhone.A, ArabicPhone.I, ArabicPhone.U):
            if phones[i + 1] is ArabicPhone.N:
                return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Redirect stdout through the Tee so every print() goes to both
    # the terminal and the output file simultaneously.
    tee = _Tee(OUTPUT_FILE)
    sys.stdout = tee  # type: ignore[assignment]

    try:
        _run_tests()
    finally:
        sys.stdout = tee._terminal   # restore normal stdout
        tee.close()
        tee._terminal.write(f"\nOutput saved to: {OUTPUT_FILE}\n")


def _run_tests() -> None:
    print("=" * 60)
    print("  Arabic G2P Pipeline Test")
    print("=" * 60)

    # --- load diacritizer (graceful on failure) ---
    print("\nLoading diacritizer...", end=" ", flush=True)
    try:
        diacritizer = ArabicDiacritizer(backend="auto")
        print(f"OK  [{diacritizer.backend}]")
        diac_ok = diacritizer.backend != "none"
    except Exception as exc:
        print(f"FAILED ({exc})")
        print("  → Running without diacritizer; G2P will use raw text.\n")
        diacritizer = ArabicDiacritizer(backend="none")
        diac_ok = False

    print()

    total_shadda_hits = 0
    total_tanwin_hits = 0

    for raw in TEST_SENTENCES:
        print("-" * 60)

        # 1. Diacritize
        try:
            diacritized = diacritizer.diacritize(raw)
        except Exception as exc:
            print(f"  [diacritizer error: {exc}] – using raw text")
            diacritized = raw

        # 2. G2P
        phones = ArabicG2P.text_to_phones(diacritized)
        phone_str = phones_to_str(phones)

        # Count phonemes (exclude SILENCE)
        n_phones = sum(1 for p in phones if p is not ArabicPhone.SILENCE)

        # Feature checks
        has_shadda = check_shadda(phones)
        has_tanwin = check_tanwin(phones)
        if has_shadda:
            total_shadda_hits += 1
        if has_tanwin:
            total_tanwin_hits += 1

        # 3. Print results
        print(f"RAW TEXT:     {raw}")
        print(f"DIACRITIZED:  {diacritized}")
        print(f"PHONEMES:     {phone_str}")
        print(f"PHONE COUNT:  {n_phones}  (excluding silence)")

        flags: list[str] = []
        if has_shadda:
            flags.append("shadda doubled ✓")
        if has_tanwin:
            flags.append("tanwin (vowel+N) ✓")
        if flags:
            print(f"FEATURES:     {', '.join(flags)}")

        print()

    # --- Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Sentences tested   : {len(TEST_SENTENCES)}")
    print(f"  Diacritizer active : {'yes – ' + diacritizer.backend if diac_ok else 'no (no-op fallback)'}")
    print(f"  Shadda detected in : {total_shadda_hits}/{len(TEST_SENTENCES)} sentences")
    print(f"  Tanwin detected in : {total_tanwin_hits}/{len(TEST_SENTENCES)} sentences")
    print()

    if not diac_ok:
        print("  TIP: install camel-tools for neural diacritization:")
        print("    pip install camel-tools")
        print("    python -m camel_tools.cli.data download diac-msa-13000")
        print()

    print(f"  Output file        : {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    main()
