from typing import Optional
from ..core.types import ArabicPhone, Shape, Emotion

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

    EMOTION_MODIFIERS = {
        Emotion.HAPPY: {Shape.D: 1.2, Shape.C: 1.1}, 
        Emotion.SAD: {Shape.D: 0.8, Shape.C: 0.9}, 
        Emotion.ANGRY: {Shape.B: 1.1, Shape.G: 1.2}, 
        Emotion.SURPRISED: {Shape.D: 1.3},
        Emotion.NEUTRAL: {},
    }

    @classmethod
    def get_shape(cls, phone: ArabicPhone, emotion: Emotion = Emotion.NEUTRAL) -> Shape:
        base_shape = cls.BASE_MAPPING.get(phone, Shape.B)
        return base_shape

    @classmethod
    def get_tween_shape(cls, shape1: Shape, shape2: Shape) -> Optional[Shape]:
        transitions = {
            (Shape.A, Shape.D): Shape.C,
            (Shape.D, Shape.A): Shape.C,
            (Shape.B, Shape.D): Shape.C,
            (Shape.F, Shape.D): Shape.E,
        }
        return transitions.get((shape1, shape2))
