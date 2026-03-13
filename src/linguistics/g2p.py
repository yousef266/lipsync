from typing import List
from ..core.types import ArabicPhone


class ArabicG2P:
    """Arabic Grapheme-to-Phoneme converter.

    Handles both diacritized input (from ArabicDiacritizer) and plain text.
    Diacritics recognised:
      َ  fatha       → A
      ُ  damma       → U
      ِ  kasra       → I
      ً  tanwin fath → A N
      ٌ  tanwin damm → U N
      ٍ  tanwin kasr → I N
      ّ  shadda      → doubles the preceding consonant phone
      ْ  sukun       → (no vowel inserted)
      ـ  tatweel     → skipped
    """

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

    # Short-vowel diacritics → single phone
    _DIACRITIC_TO_PHONE = {
        "\u064E": ArabicPhone.A,   # fatha   َ
        "\u064F": ArabicPhone.U,   # damma   ُ
        "\u0650": ArabicPhone.I,   # kasra   ِ
    }

    # Tanwin diacritics → two phones (vowel + nasal N)
    _TANWIN_TO_PHONES = {
        "\u064B": [ArabicPhone.A, ArabicPhone.N],  # tanwin fath  ً
        "\u064C": [ArabicPhone.U, ArabicPhone.N],  # tanwin damm  ٌ
        "\u064D": [ArabicPhone.I, ArabicPhone.N],  # tanwin kasr  ٍ
    }

    # Characters that carry no phonetic value and are silently skipped
    _SKIP_CHARS = {
        "\u0651",  # shadda  ّ  – handled separately (doubles preceding phone)
        "\u0652",  # sukun   ْ  – explicit zero-vowel marker
        "\u0640",  # tatweel ـ  – aesthetic stretcher
        "\u0670",  # superscript alef ٰ
    }

    @classmethod
    def text_to_phones(cls, text: str) -> List[ArabicPhone]:
        """Convert Arabic text (ideally diacritized) to a phoneme sequence.

        Parameters
        ----------
        text : str
            Arabic word or sentence, with or without diacritics.

        Returns
        -------
        List[ArabicPhone]
        """
        phones: List[ArabicPhone] = []

        for char in text:
            if char in cls.LETTER_TO_PHONE:
                phones.append(cls.LETTER_TO_PHONE[char])

            elif char in cls._DIACRITIC_TO_PHONE:
                phones.append(cls._DIACRITIC_TO_PHONE[char])

            elif char in cls._TANWIN_TO_PHONES:
                phones.extend(cls._TANWIN_TO_PHONES[char])

            elif char == "\u0651":  # shadda – duplicate the last consonant
                if phones:
                    phones.append(phones[-1])

            elif char in cls._SKIP_CHARS:
                pass  # sukun / tatweel carry no lip-sync information

            elif char == " ":
                phones.append(ArabicPhone.SILENCE)

        return phones
