"""
Arabic Neural Diacritizer
=========================
Wraps a neural diacritization backend and exposes a single method:

    diacritize(text: str) -> str

Backends (tried in order when backend="auto"):
  1. camel-tools  – CAMeL-Lab BERT model, best quality
                    pip install camel-tools
                    python -m camel_tools.cli.data download diac-msa-13000

  2. no-op        – returns text unchanged; phoneme quality is reduced.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from typing import Callable


class ArabicDiacritizer:
    """
    Neural Arabic diacritizer – adds harakat to undiacritized text.

    Parameters
    ----------
    backend : str
        "auto"  – try camel-tools → transformers → no-op  (default)
        "camel" – force camel-tools  (raises ImportError if not installed)
        "hf"    – force HuggingFace transformers pipeline
        "none"  – no-op (useful for testing or already-diacritized input)
    device : str | None
        Torch device string, e.g. "cuda" or "cpu".
        None = auto-detect (CUDA if available, else CPU).

    Example
    -------
    >>> diac = ArabicDiacritizer()
    >>> diac.diacritize("كتب محمد الرسالة")
    'كَتَبَ مُحَمَّدٌ الرِّسَالَةَ'
    """

    # ---------------------------------------------------------------------------
    # Class-level config – override before instantiation to use a different model
    # ---------------------------------------------------------------------------

    # HuggingFace model ID used when camel-tools is unavailable.
    # Must support the "text2text-generation" pipeline task.
    # Example alternatives (install separately):
    #   "abdoulsn/arabic-text-diacritization"
    HF_MODEL_ID: str = "CAMeL-Lab/camelbert-msa-diacritizer"

    # Maximum token length forwarded to the HF pipeline.
    HF_MAX_LENGTH: int = 512

    def __init__(
        self,
        backend: str = "auto",
        device: str | None = None,
    ) -> None:
        self._fn: Callable[[str], str] = lambda t: t
        self._backend_name = "none"

        if backend == "none":
            warnings.warn(
                "ArabicDiacritizer is running in no-op mode. "
                "Phoneme quality will be reduced for undiacritized text.",
                UserWarning,
                stacklevel=2,
            )
            return

        if backend in ("auto", "camel"):
            if self._try_load_camel():
                return
            if backend == "camel":
                raise ImportError(
                    "camel-tools is required but not installed.\n"
                    "  pip install camel-tools\n"
                    "  python -m camel_tools.cli.data download diac-msa-13000"
                )

        if backend in ("auto", "hf"):
            if self._try_load_hf(device):
                return
            if backend == "hf":
                raise RuntimeError(
                    f"Could not load HuggingFace model '{self.HF_MODEL_ID}'.\n"
                    "  pip install transformers\n"
                    f"  Verify the model ID: {self.HF_MODEL_ID}"
                )

        warnings.warn(
            "ArabicDiacritizer: no backend available. "
            "Install camel-tools (recommended) or transformers. "
            "Falling back to no-op – phoneme accuracy will be reduced.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def diacritize(self, text: str) -> str:
        """
        Add Arabic diacritics (harakat) to undiacritized text.

        Parameters
        ----------
        text : str
            Raw Arabic text, with or without diacritics.

        Returns
        -------
        str
            Fully diacritized Arabic text.
        """
        if not text or not text.strip():
            return text
        cleaned = self._normalize(text.strip())
        return self._fn(cleaned)

    @property
    def backend(self) -> str:
        """Name of the active backend (read-only)."""
        return self._backend_name

    def __repr__(self) -> str:
        return f"ArabicDiacritizer(backend={self._backend_name!r})"

    # ---------------------------------------------------------------------------
    # Backend loaders
    # ---------------------------------------------------------------------------

    # Morphology database used for diacritization.
    # "calima-msa-r13"  → Modern Standard Arabic (always bundled)
    # "calima-egy-r13"  → Egyptian Arabic dialect
    CAMEL_DB: str = "calima-msa-r13"

    def _try_load_camel(self) -> bool:
        """Load the camel-tools MLEDisambiguator diacritizer.

        camel-tools v1.x uses MLEDisambiguator (not MSADiacritizer).
        It picks the most-likely morphological analysis for each token
        and returns its 'diac' field, which is the fully-vowelled form.
        """
        try:
            from camel_tools.disambig.mle import MLEDisambiguator        # type: ignore
            from camel_tools.tokenizers.word import simple_word_tokenize  # type: ignore

            disambig = MLEDisambiguator.pretrained(self.CAMEL_DB)

            def _run(text: str) -> str:
                tokens = simple_word_tokenize(text)
                results = disambig.disambiguate(tokens)
                diacritized = [
                    r.analyses[0].analysis.get("diac", r.word)
                    for r in results
                ]
                return " ".join(diacritized)

            self._fn = _run
            self._backend_name = f"camel-tools ({self.CAMEL_DB})"
            return True

        except Exception:
            return False

    def _try_load_hf(self, device: str | None) -> bool:
        """Try to load a HuggingFace seq2seq diacritization pipeline."""
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore

            _device = device
            if _device is None:
                _device = "cuda" if torch.cuda.is_available() else "cpu"

            pipe = pipeline(
                "text2text-generation",
                model=self.HF_MODEL_ID,
                device=_device,
            )
            max_len = self.HF_MAX_LENGTH

            def _run(text: str) -> str:
                out = pipe(text, max_new_tokens=max_len)
                return out[0]["generated_text"]

            self._fn = _run
            self._backend_name = f"hf:{self.HF_MODEL_ID}"
            return True

        except Exception:
            return False

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Light normalization: remove non-Arabic punctuation noise."""
        # Strip tatweel (U+0640) – it's a purely aesthetic stretcher
        text = text.replace("\u0640", "")
        # Normalize to NFC
        return unicodedata.normalize("NFC", text)
