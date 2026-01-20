from typing import List
from ..core.types import ArabicPhone

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
