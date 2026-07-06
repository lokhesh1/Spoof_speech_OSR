#!/usr/bin/env python3
"""Load speech/emotion pre-trained models from Hugging Face.

Single entry point::

    from model_loader import load_model
    model = load_model("openai/whisper-base", device="auto")
    layers = model.forward_layers(waveform)  # {layer_idx: (T, D) tensor}

Layer 0 = encoder input embedding, 1..N = transformer block outputs.
Architecture (Whisper vs. wav2vec2/HuBERT/WavLM) is auto-detected from the
model config on Hugging Face.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("PyTorch is required.  pip install torch") from exc

logger = logging.getLogger("ser.model_loader")
TARGET_SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
_NOISY_LOGGERS = (
    "root", "httpx", "httpcore", "urllib3", "filelock",
    "huggingface_hub", "numba", "torchaudio", "matplotlib",
)


class _MuteThirdParty(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name.split(".", 1)[0] not in _NOISY_LOGGERS


def configure_logging(verbose: bool = False) -> None:
    """Install a filtered stderr handler; mutes HF/torchaudio download chatter."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.addFilter(_MuteThirdParty())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("ser").setLevel(logging.DEBUG if verbose else logging.INFO)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class BaseSERModel(ABC):
    """Common interface for all loaded models."""

    def __init__(self, hf_id: str, device: str = "cpu") -> None:
        self.hf_id = hf_id
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        """Run on a 1-D 16 kHz mono waveform.

        Returns ``{layer_idx: (T, D) float32 tensor}``.
        Layer 0 = input embedding, 1..N = transformer block outputs.
        """


# --------------------------------------------------------------------------- #
# wav2vec2 / HuBERT / WavLM and compatible architectures
# --------------------------------------------------------------------------- #
class TransformersSERModel(BaseSERModel):

    def __init__(self, hf_id: str, device: str = "cpu") -> None:
        super().__init__(hf_id, device)
        from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor

        try:
            self.processor = AutoFeatureExtractor.from_pretrained(hf_id)
        except Exception as exc:
            logger.warning(
                "No feature-extractor config in '%s' (%s); using default 16 kHz extractor.",
                hf_id, exc,
            )
            self.processor = Wav2Vec2FeatureExtractor(
                sampling_rate=TARGET_SAMPLE_RATE, do_normalize=True,
                return_attention_mask=True,
            )
        self.model = AutoModel.from_pretrained(hf_id, output_hidden_states=True)
        self.model.to(device).eval()
        logger.info("Loaded '%s' (wav2vec2/HuBERT/WavLM) on %s.", hf_id, device)

    @torch.no_grad()
    def forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        inputs = self.processor(
            waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # hidden_states: tuple of (1, T, D); index 0 = embeddings, 1..N = layers
        return {i: hs.squeeze(0).cpu() for i, hs in enumerate(outputs.hidden_states)}


# --------------------------------------------------------------------------- #
# Whisper encoder
# --------------------------------------------------------------------------- #
class WhisperSERModel(BaseSERModel):

    def __init__(self, hf_id: str, device: str = "cpu") -> None:
        super().__init__(hf_id, device)
        from transformers import AutoModel, WhisperFeatureExtractor

        self.processor = WhisperFeatureExtractor.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id)
        self.model.to(device).eval()
        self._encoder = self._locate_encoder()
        logger.info(
            "Loaded '%s' (Whisper encoder) on %s; %d transformer layers.",
            hf_id, device, len(self._encoder.layers),
        )

    def _locate_encoder(self):
        m = self.model
        for candidate in (
            getattr(m, "encoder", None),
            getattr(getattr(m, "whisper", None), "encoder", None),
        ):
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate
        for module in m.modules():
            if hasattr(module, "layers") and hasattr(module, "embed_positions"):
                return module
        raise RuntimeError(
            f"Cannot locate a Whisper encoder in '{self.hf_id}'. "
            f"Expected model.encoder with .layers and .embed_positions."
        )

    def _valid_frames(self, num_samples: int, total: int) -> int:
        hop = getattr(self.processor, "hop_length", 160) or 160
        return max(1, min(int(np.ceil(num_samples / hop / 2)), total))

    @torch.no_grad()
    def forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        feats = self.processor(
            waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        enc_out = self._encoder(
            input_features=feats.input_features.to(self.device),
            output_hidden_states=True,
        )
        total = enc_out.hidden_states[0].shape[1]
        valid = self._valid_frames(int(waveform.shape[-1]), total)
        return {i: hs.squeeze(0)[:valid].cpu() for i, hs in enumerate(enc_out.hidden_states)}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_model(hf_id: str, device: str = "auto") -> BaseSERModel:
    """Load a model from Hugging Face by repo ID.

    Architecture is auto-detected from the model config:
    - ``model_type == "whisper"``  -> :class:`WhisperSERModel`
    - anything else                -> :class:`TransformersSERModel`

    Parameters
    ----------
    hf_id:
        Hugging Face repo ID (e.g. ``"openai/whisper-base"``,
        ``"superb/hubert-large-superb-er"``).
    device:
        ``"auto"`` picks CUDA if available, else CPU.
        Or pass ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` explicitly.
    """
    device = resolve_device(device)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "").lower()
    logger.info("Detected model_type='%s' for '%s'.", model_type, hf_id)
    if model_type == "whisper":
        return WhisperSERModel(hf_id, device)
    return TransformersSERModel(hf_id, device)
