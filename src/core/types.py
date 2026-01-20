from enum import Enum

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
