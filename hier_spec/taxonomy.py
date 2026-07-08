#!/usr/bin/env python3
"""Architecture-then-model taxonomy for the 24 known MLAAD-v5 training models.

The protocol CSVs carry only ``model_name`` (no architecture column), so the
Supra-to-Sub hierarchy is authored here. Grouping the 24 training models by
generator family gives **4 multi-model architectures** (XTTS, Tacotron, VITS,
Bark) and **9 singleton architectures** (|M_a| = 1) -> 13 architectures total.

``vixTTS`` is treated as its own singleton architecture (not folded into XTTS),
which is what makes the singleton count exactly 9.

:func:`derive_arch` maps *any* model name (including unseen dev/eval models) to
an architecture label for **ground-truth calibration and metrics only** -- it is
never used at inference, which relies solely on the trained heads and the OOD
gates.
"""
from __future__ import annotations

from typing import Dict, List

UNKNOWN_ARCH = "__unknown_arch__"

# --- explicit architecture of every known (train.csv) model -----------------
ARCH_OF_KNOWN: Dict[str, str] = {
    # Tacotron (5)
    "tts_models/en/ljspeech/tacotron2-DCA": "Tacotron",
    "tts_models/en/ljspeech/tacotron2-DDC": "Tacotron",
    "tts_models/en/ljspeech/tacotron2-DDC_ph": "Tacotron",
    "tts_models/fr/mai/tacotron2-DDC": "Tacotron",
    "tts_models/de/thorsten/tacotron2-DDC": "Tacotron",
    # VITS (6)
    "tts_models/en/ljspeech/vits--neon": "VITS",
    "tts_models/fr/css10/vits": "VITS",
    "tts_models/de/css10/vits-neon": "VITS",
    "tts_models/it/mai_male/vits": "VITS",
    "tts_models/it/mai_female/vits": "VITS",
    "tts_models/lt/cv/vits": "VITS",
    # XTTS (2)
    "tts_models/multilingual/multi-dataset/xtts_v1.1": "XTTS",
    "tts_models/multilingual/multi-dataset/xtts_v2": "XTTS",
    # Bark (2)
    "suno/bark": "Bark",
    "suno/bark-small": "Bark",
    # Singletons (9)
    "facebook/mms-tts-deu": "MMS",
    "griffin_lim": "griffin_lim",
    "Mars5": "Mars5",
    "MeloTTS": "MeloTTS",
    "Metavoice-1B": "Metavoice-1B",
    "tts_models/en/ljspeech/fast_pitch": "fast_pitch",
    "tts_models/en/ljspeech/speedy-speech": "speedy-speech",
    "tts_models/it/mai_female/glow-tts": "Glow-TTS",
    "vixTTS": "vixTTS",
}

#: Architectures that contain more than one known model (Stage-2 gate applies).
MULTI_MODEL_ARCHS = ("Bark", "Tacotron", "VITS", "XTTS")


def _sorted_unique(values) -> List[str]:
    return sorted(set(values))


def arch_label_map() -> Dict[str, int]:
    """Architecture name -> index ``0..12`` (sorted, deterministic)."""
    return {a: i for i, a in enumerate(_sorted_unique(ARCH_OF_KNOWN.values()))}


def known_archs() -> List[str]:
    return _sorted_unique(ARCH_OF_KNOWN.values())


def models_of_arch() -> Dict[str, List[str]]:
    """Architecture -> sorted list of its known member model names."""
    out: Dict[str, List[str]] = {}
    for model, arch in ARCH_OF_KNOWN.items():
        out.setdefault(arch, []).append(model)
    return {a: sorted(ms) for a, ms in out.items()}


def model_label_maps() -> Dict[str, Dict[str, int]]:
    """For each multi-model architecture: member model name -> local index."""
    m_of_a = models_of_arch()
    return {
        arch: {m: i for i, m in enumerate(m_of_a[arch])}
        for arch in MULTI_MODEL_ARCHS
    }


def singleton_model_of_arch() -> Dict[str, str]:
    """For each singleton architecture: the single known model name it maps to."""
    m_of_a = models_of_arch()
    return {a: ms[0] for a, ms in m_of_a.items() if a not in MULTI_MODEL_ARCHS}


def derive_arch(model_name: str) -> str:
    """Ground-truth architecture for any model name (unseen models included).

    Used only to build calibration/metrics targets, never at inference.
    Returns :data:`UNKNOWN_ARCH` for models whose family was not in training.
    """
    if model_name in ARCH_OF_KNOWN:
        return ARCH_OF_KNOWN[model_name]
    s = model_name.lower()
    # order matters: check specific families before generic substrings
    if "xtts" in s:
        return "XTTS"
    if "bark" in s:
        return "Bark"
    if "tacotron" in s:
        return "Tacotron"
    if "glow-tts" in s or "glow_tts" in s:
        return "Glow-TTS"
    if "vits" in s:
        return "VITS"
    if "mms-tts" in s or "mms_tts" in s:
        return "MMS"
    if "griffin" in s:
        return "griffin_lim"
    if "melo" in s:
        return "MeloTTS"
    if "mars5" in s:
        return "Mars5"
    if "metavoice" in s:
        return "Metavoice-1B"
    if "fast_pitch" in s or "fastpitch" in s:
        return "fast_pitch"
    if "speedy" in s:
        return "speedy-speech"
    return UNKNOWN_ARCH
