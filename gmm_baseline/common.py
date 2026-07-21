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


def top2_log_ratio(C: np.ndarray) -> float:
    """Log of the top-two likelihood ratio: ``top1 - top2`` (>= 0).

    Preferred as a *ranking* score (EER, ROC): monotonically equivalent to
    :func:`top2_likelihood_ratio` but cannot overflow to ``inf``.
    ``inf`` when there is only one class.
    """
    if C.shape[0] < 2:
        return float("inf")
    top1, top2 = np.partition(C, -2)[-2:][::-1]   # two largest, descending
    return float(top1 - top2)


def top2_likelihood_ratio(C: np.ndarray) -> float:
    """Likelihood ratio between the two best-scoring GMMs for one clip.

    ``C`` is a K-dim vector of mean per-frame *log*-likelihoods (from
    :func:`score_vector`). The ratio of the two largest likelihoods is
    ``exp(top1 - top2)`` -- equivalently the ratio of the top-two softmax
    probabilities, since the normaliser cancels. Length-invariant because the
    log-liks are per-frame means. ``inf`` when there is only one class.
    """
    return float(np.exp(top2_log_ratio(C)))


def _clean_gate_scores(y_true_known: Sequence[bool],
                       y_score_known: Sequence[float],
                       caller: str) -> Optional[tuple]:
    """Validate a (label, score) pair for the threshold-free gate metrics.

    Drops non-finite scores and returns ``None`` when the metric is undefined
    (i.e. one of the two groups is absent).
    """
    y = np.asarray(y_true_known, dtype=bool)
    s = np.asarray(y_score_known, dtype=np.float64)
    finite = np.isfinite(s)
    if not finite.all():
        logger.warning("%s: %d non-finite scores dropped", caller, int((~finite).sum()))
        y, s = y[finite], s[finite]
    if not y.any() or y.all():
        return None
    return y, s


def compute_eer(y_true_known: Sequence[bool],
                y_score_known: Sequence[float]) -> tuple[float, float]:
    """Equal error rate of the known/unknown gate, and the score at the EER.

    ``y_score_known`` is a continuous score where *higher = more known* (e.g.
    the top-2 log ratio, or ``OneClassSVM.decision_function``). Threshold-free:
    it characterises the score's ranking, not any particular operating point.
    With ``known`` as the positive class, ``fpr`` is the rate of unknown clips
    wrongly accepted (FAR) and ``1 - tpr`` the rate of known clips wrongly
    rejected (FRR); the EER is where the two cross.

    Returns ``(nan, nan)`` unless both groups are present.
    """
    from sklearn.metrics import roc_curve

    cleaned = _clean_gate_scores(y_true_known, y_score_known, "compute_eer")
    if cleaned is None:
        return float("nan"), float("nan")
    y, s = cleaned

    fpr, tpr, thr = roc_curve(y.astype(int), s, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def compute_auroc(y_true_known: Sequence[bool],
                  y_score_known: Sequence[float]) -> float:
    """Area under the ROC curve for the known/unknown gate (higher = better).

    Same score convention and threshold-freedom as :func:`compute_eer`, and
    read off the same ROC. Equals the probability that a randomly drawn known
    clip outranks a randomly drawn unknown one: 1.0 is perfect separation, 0.5
    is chance. Symmetric in the class convention -- scoring *unknown* as the
    positive class with a negated score gives the identical value.

    Returns ``nan`` unless both groups are present.
    """
    from sklearn.metrics import roc_auc_score

    cleaned = _clean_gate_scores(y_true_known, y_score_known, "compute_auroc")
    if cleaned is None:
        return float("nan")
    y, s = cleaned
    return float(roc_auc_score(y.astype(int), s))


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
    y_score_known: Optional[Sequence[float]] = None,
    n_classes: Optional[int] = None,
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

    Two macro-F1 numbers weight every class equally regardless of support:

    * ``macro_f1_open``   -- over the K+1 classes (knowns + ``unknown``); the
      full open-set task, gate included.
    * ``macro_f1_closed`` -- over the known classes on known clips using the raw
      ``argmax``; pure attribution, gate-free.

    ``y_score_known`` (optional) is a continuous gate score, *higher = more
    known*, used only for the threshold-free ``eer``/``eer_threshold``.
    ``n_classes`` pins the macro-F1 label set to ``0..K-1`` instead of deriving
    it from the classes observed in this split.
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
    rec = tp / n_unknown if n_unknown else 0.0   # = unknown_recall (fn = n_unknown - tp)
    f1 = 2 * precision * rec / (precision + rec) if (precision + rec) else 0.0

    # per-group accuracy
    closed = float(np.mean(am[tk] == yt[tk])) if n_known else float("nan")   # gate-free
    acc_known_open = float(np.mean(yp[tk] == yt[tk])) if n_known else float("nan")

    # macro-F1: every class counts equally, whatever its support
    from sklearn.metrics import f1_score

    if n_classes is not None:
        known_labels = list(range(n_classes))
    else:
        seen = np.concatenate([yt, yp, am])
        known_labels = sorted({int(v) for v in seen if v != UNKNOWN_LABEL})
    macro_f1_open = float(f1_score(yt, yp, labels=known_labels + [UNKNOWN_LABEL],
                                   average="macro", zero_division=0))
    if n_known:
        # restrict to classes actually present among the known ground truth
        closed_labels = sorted({int(v) for v in yt[tk]})
        macro_f1_closed = float(f1_score(yt[tk], am[tk], labels=closed_labels,
                                         average="macro", zero_division=0))
    else:
        macro_f1_closed = float("nan")

    if y_score_known is not None:
        eer, eer_thr = compute_eer(tk, y_score_known)
        auroc = compute_auroc(tk, y_score_known)
    else:
        eer = eer_thr = auroc = float("nan")

    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "auroc": auroc,
        "macro_f1_open": macro_f1_open,
        "macro_f1_closed": macro_f1_closed,
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
    eer = ("n/a" if np.isnan(m["eer"])
           else f"{m['eer']:.4f}  (@ score {m['eer_threshold']:.3f})")
    auroc = "n/a" if np.isnan(m["auroc"]) else f"{m['auroc']:.4f}"
    return (
        f"[{tag}] n={m['n']} (known={m['n_known']}, unknown={m['n_unknown']})\n"
        f"    balanced acc (model axis): open-set {m['balanced_open_set_acc']:.4f}"
        f"  |  detection {m['balanced_detection_acc']:.4f}\n"
        f"    macro-F1                 : open-set (K+1) {m['macro_f1_open']:.4f}"
        f"  |  closed-set (K, gate-free) {m['macro_f1_closed']:.4f}\n"
        f"    gate (threshold-free)    : EER {eer}  |  AUROC {auroc}\n"
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
