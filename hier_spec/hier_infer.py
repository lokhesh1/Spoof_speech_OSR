#!/usr/bin/env python3
"""Shared inference helpers for Hier-Spec calibration and evaluation.

Loads a trained checkpoint + Mahalanobis stats, embeds clips, and exposes the
per-level Mahalanobis machinery (Stage-1 architecture scores and, given a
chosen architecture, Stage-2 model scores over that architecture's members).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import taxonomy as tax
from common import HierSpecConfig, add_repo_to_path, mahalanobis
from data import HierSpecDataset, collate


class Bundle:
    """A loaded Hier-Spec model plus its statistics and label bookkeeping."""

    def __init__(self, model, stats: dict, meta: dict, device):
        self.model = model
        self.device = device
        self.meta = meta
        self.cfg = HierSpecConfig(**meta["config"])
        self.arch_label_map: Dict[str, int] = meta["arch_label_map"]
        self.idx_to_arch = {i: a for a, i in self.arch_label_map.items()}
        self.model_label_maps: Dict[str, Dict[str, int]] = meta["model_label_maps"]
        self.label_map: Dict[str, int] = meta["label_map"]           # 24 known models
        self.multi_model_archs: List[str] = meta["multi_model_archs"]

        self.arch_means = stats["arch_means"]
        self.arch_chol = stats["arch_chol"]
        self.model_means = stats["model_means"]
        self.model_chol = stats["model_chol"]

        # flat model index (0..23) of each multi-model arch's members, in the
        # same local order the model head uses
        m_of_a = tax.models_of_arch()
        self.arch_member_flat: Dict[str, List[int]] = {}
        self.arch_member_names: Dict[str, List[str]] = {}
        for arch in self.multi_model_archs:
            members = sorted(m_of_a[arch])                # matches model_label_maps
            self.arch_member_flat[arch] = [self.label_map[m] for m in members]
            self.arch_member_names[arch] = members
        self.singleton_model = tax.singleton_model_of_arch()

    # -- Mahalanobis scores -------------------------------------------------- #
    def stage1_scores(self, emb: np.ndarray) -> np.ndarray:
        """(N, 13) architecture Mahalanobis distances."""
        return mahalanobis(emb, self.arch_means, self.arch_chol)

    def stage2_scores(self, emb: np.ndarray, arch: str) -> np.ndarray:
        """(N, |members|) model Mahalanobis distances within ``arch``."""
        flat_idx = self.arch_member_flat[arch]
        return mahalanobis(emb, self.model_means[flat_idx], self.model_chol)


def load_bundle(artifacts_dir: Path, device, checkpoint: str = "best.pt") -> Bundle:
    import torch

    from model import HierSpecModel

    artifacts_dir = Path(artifacts_dir)
    ckpt = torch.load(artifacts_dir / checkpoint, map_location=device,
                      weights_only=False)
    meta = {k: ckpt[k] for k in ("config", "arch_label_map", "model_label_maps",
                                 "label_map", "multi_model_archs")}
    stats = dict(np.load(artifacts_dir / "stats.npz"))
    model = HierSpecModel(meta["arch_label_map"], meta["model_label_maps"],
                          tuple(meta["multi_model_archs"]),
                          feat_dim=meta["config"]["feat_dim"],
                          s=meta["config"]["arcface_s"],
                          m=meta["config"]["arcface_m"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return Bundle(model, stats, meta, device)


def embed_clips(bundle: Bundle, clips, *, batch_size: int, num_workers: int
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[dict]]:
    """Return ``(emb, arch_pred, model_pred_local, metas)``.

    ``emb`` is ``(N, 160)`` float64, ``arch_pred`` the arch-head argmax
    (``N,``), ``model_pred_local`` maps each multi-model arch -> its model-head
    argmax over all N clips, and ``metas`` the per-clip protocol metadata.
    """
    import torch
    from torch.utils.data import DataLoader

    ds = HierSpecDataset(clips, max_frames=bundle.cfg.max_frames, train=False,
                         arch_label_map=bundle.arch_label_map,
                         model_label_maps=bundle.model_label_maps,
                         return_meta=True, seed=bundle.cfg.seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)

    embs: List[np.ndarray] = []
    arch_preds: List[np.ndarray] = []
    model_preds: Dict[str, List[np.ndarray]] = {a: [] for a in bundle.multi_model_archs}
    metas: List[dict] = []
    with torch.no_grad():
        for feats, _arch_lab, _model_local, _flat, meta in loader:
            arch_logits, model_logits, x = bundle.model.predict_logits(
                feats.to(bundle.device, non_blocking=True))
            embs.append(x.cpu().numpy())
            arch_preds.append(arch_logits.argmax(1).cpu().numpy())
            for arch in bundle.multi_model_archs:
                model_preds[arch].append(model_logits[arch].argmax(1).cpu().numpy())
            metas.extend(meta)
    emb = np.concatenate(embs).astype(np.float64)
    arch_pred = np.concatenate(arch_preds)
    model_pred_local = {a: np.concatenate(v) for a, v in model_preds.items()}
    return emb, arch_pred, model_pred_local, metas


def load_split_clips(protocol_dir: str, split: str, feat_root: str, layer: int,
                     label_map, subsplit=None):
    add_repo_to_path()
    from protocols_mlaad import load_split

    return load_split(protocol_dir, split, subsplit=subsplit,
                      feat_root=feat_root, layer=layer, label_map=label_map)
