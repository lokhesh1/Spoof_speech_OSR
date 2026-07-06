#!/usr/bin/env python3
"""Shared pieces for the LFCC + per-class GMM + one-class-SVM baseline.

This is the *first* (non-embedding) source-tracing system: raw MLAAD audio is
turned into LFCCs, one 512-component GMM is fit per known class, and each clip
is described by the vector of per-GMM log-likelihoods. A single global
one-class SVM gates known vs. unknown; the known class is ``argmax`` of that
log-likelihood vector.

Feature front-end (fixed by spec)
---------------------------------
* 16 kHz mono
* 30 ms frame  -> ``win_length = 480`` samples
* 59 % overlap -> ``hop_length = 197`` samples  (hop = round(480 * 0.41))
* 20 LFCC coefficients, 70 linear filters, ``n_fft = 512``
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("gmm_baseline.common")

UNKNOWN_LABEL = -1


# --------------------------------------------------------------------------- #
# Feature configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LFCCConfig:
    sample_rate: int = 16_000
    frame_seconds: float = 0.03          # 30 ms window
    overlap: float = 0.59                # 59 % overlap
    n_fft: int = 512
    n_lfcc: int = 20                     # 20 cepstral coefficients (no deltas)
    n_filter: int = 70                   # 70 linear filters

    @property
    def win_length(self) -> int:
        return int(round(self.frame_seconds * self.sample_rate))       # 480

    @property
    def hop_length(self) -> int:
        return int(round(self.win_length * (1.0 - self.overlap)))      # 197

    def to_dict(self) -> dict:
        d = asdict(self)
        d["win_length"] = self.win_length
        d["hop_length"] = self.hop_length
        return d


def build_lfcc_transform(cfg: LFCCConfig):
    """Construct a ``torchaudio`` LFCC transform for ``cfg``."""
    from torchaudio.transforms import LFCC

    return LFCC(
        sample_rate=cfg.sample_rate,
        n_filter=cfg.n_filter,
        n_lfcc=cfg.n_lfcc,
        speckwargs={
            "n_fft": cfg.n_fft,
            "win_length": cfg.win_length,
            "hop_length": cfg.hop_length,
        },
    )


# --------------------------------------------------------------------------- #
# Audio + LFCC
# --------------------------------------------------------------------------- #
def load_audio(path: Path | str, sample_rate: int = 16_000):
    """Load an audio file as a 1-D mono float32 waveform tensor at ``sample_rate``.

    Uses ``soundfile`` for decoding (no TorchCodec dependency) and
    ``torchaudio.functional.resample`` only when the native rate differs.
    """
    import soundfile as sf
    import torch

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:                         # (n, channels) -> mono
        data = data.mean(axis=1)
    wav = torch.from_numpy(np.ascontiguousarray(data))
    if sr != sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, sample_rate).squeeze(0)
    return wav


def compute_lfcc(path: Path | str, transform, cfg: LFCCConfig) -> np.ndarray:
    """Return the LFCC sequence ``(T, n_lfcc)`` (float32) for one clip."""
    import torch

    wav = load_audio(path, cfg.sample_rate)
    with torch.no_grad():
        lfcc = transform(wav.unsqueeze(0))     # (1, n_lfcc, T)
    return lfcc.squeeze(0).transpose(0, 1).contiguous().numpy().astype(np.float32)


def lfcc_cached(
    rel: str,
    wav_path: Path | str,
    transform,
    cfg: LFCCConfig,
    cache_dir: Optional[Path],
) -> np.ndarray:
    """LFCC for a clip, reading/writing an ``.npy`` cache under ``cache_dir``.

    ``cache_dir`` mirrors the protocol ``rel`` layout. With no cache dir the
    LFCC is simply recomputed each call.
    """
    if cache_dir is None:
        return compute_lfcc(wav_path, transform, cfg)

    cache_path = Path(cache_dir) / Path(rel).with_suffix(".npy")
    if cache_path.is_file():
        return np.load(cache_path)
    feat = compute_lfcc(wav_path, transform, cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, feat)
    return feat


# --------------------------------------------------------------------------- #
# GMM scoring
# --------------------------------------------------------------------------- #
def score_vector(lfcc: np.ndarray, gmms: Sequence) -> np.ndarray:
    """K-dim vector: mean per-frame log-likelihood of ``lfcc`` under each GMM.

    ``sklearn``'s ``GaussianMixture.score`` already returns the mean per-sample
    (per-frame) log-likelihood, so this is length-invariant by construction.
    """
    lfcc = lfcc.astype(np.float64, copy=False)   # match GMMs fit in float64
    return np.fromiter((g.score(lfcc) for g in gmms), dtype=np.float64,
                       count=len(gmms))


# --------------------------------------------------------------------------- #
# Artifact IO
# --------------------------------------------------------------------------- #
def save_artifacts(path: Path | str, artifacts: dict) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)
    logger.info("Saved artifacts -> %s", path)


def load_artifacts(path: Path | str) -> dict:
    import joblib

    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Open-set metrics
# --------------------------------------------------------------------------- #
def _balanced(parts: Sequence[float]) -> float:
    """Mean over the groups that are defined (drop NaN for empty groups)."""
    vals = [v for v in parts if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def compute_metrics(
    y_true_label: Sequence[int],
    y_true_known: Sequence[bool],
    y_pred_label: Sequence[int],
    y_pred_known: Sequence[bool],
    y_pred_argmax: Sequence[int],
) -> Dict[str, float]:
    """Open-set source-tracing metrics.

    ``*_label`` uses class index ``0..K-1`` for known, ``-1`` for unknown.
    ``*_known`` is the binary known/unknown gate decision. ``y_pred_argmax`` is
    the raw ``argmax`` class before the gate (for gate-independent closed-set).

    Two "balanced" numbers average the known and unknown groups equally, so the
    eval imbalance (~13.5k known vs ~20.3k unknown on the model axis) does not
    dominate:

    * ``balanced_detection_acc`` = mean(known_recall, unknown_recall) -- balanced
      accuracy of the known-vs-unknown gate on the model axis.
    * ``balanced_open_set_acc``  = mean(known open-set acc, unknown recall) --
      balanced accuracy of the full task (known must be gated-known AND correct
      class; unknown must be gated unknown).
    """
    yt = np.asarray(y_true_label)
    tk = np.asarray(y_true_known, dtype=bool)
    yp = np.asarray(y_pred_label)
    pk = np.asarray(y_pred_known, dtype=bool)
    am = np.asarray(y_pred_argmax)
    n = len(yt)

    n_known = int(np.sum(tk))
    n_unknown = int(np.sum(~tk))

    # known-vs-unknown gate (unknown = positive class for P/R/F1)
    known_recall = float(np.mean(pk[tk])) if n_known else float("nan")
    unknown_recall = float(np.mean(~pk[~tk])) if n_unknown else float("nan")
    tp = int(np.sum(~pk & ~tk)); fp = int(np.sum(~pk & tk))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + n_unknown) if n_unknown else 0.0
    f1 = 2 * precision * rec / (precision + rec) if (precision + rec) else 0.0

    # per-group accuracy
    closed = float(np.mean(am[tk] == yt[tk])) if n_known else float("nan")   # gate-free
    acc_known_open = float(np.mean(yp[tk] == yt[tk])) if n_known else float("nan")

    return {
        "n": n,
        "n_known": n_known,
        "n_unknown": n_unknown,
        "detection_acc": float(np.mean(pk == tk)),
        "balanced_detection_acc": _balanced([known_recall, unknown_recall]),
        "balanced_open_set_acc": _balanced([acc_known_open, unknown_recall]),
        "known_recall": known_recall,
        "unknown_recall": unknown_recall,
        "unknown_precision": precision,
        "unknown_f1": f1,
        "closed_set_acc": closed,
        "known_open_set_acc": acc_known_open,
        "open_set_acc": float(np.mean(yp == yt)),
    }


def format_metrics(tag: str, m: Dict[str, float]) -> str:
    return (
        f"[{tag}] n={m['n']} (known={m['n_known']}, unknown={m['n_unknown']})\n"
        f"    balanced acc (model axis): open-set {m['balanced_open_set_acc']:.4f}"
        f"  |  detection {m['balanced_detection_acc']:.4f}\n"
        f"    open-set acc (micro)     : {m['open_set_acc']:.4f}\n"
        f"    detection acc (micro)    : {m['detection_acc']:.4f}"
        f"  (known recall {m['known_recall']:.3f}, unknown recall {m['unknown_recall']:.3f})\n"
        f"    unknown P/R/F1           : {m['unknown_precision']:.3f} / "
        f"{m['unknown_recall']:.3f} / {m['unknown_f1']:.3f}\n"
        f"    closed-set acc (known)   : {m['closed_set_acc']:.4f}"
        f"  (gate-free)  |  open-set on known {m['known_open_set_acc']:.4f}"
    )


# --------------------------------------------------------------------------- #
# protocol import helper
# --------------------------------------------------------------------------- #
def add_repo_to_path() -> None:
    """Make ``protocols_mlaad`` importable when running from ``gmm_baseline/``."""
    import sys

    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
