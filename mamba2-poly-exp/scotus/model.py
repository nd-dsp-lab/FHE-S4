"""Mamba-2 sequence classifier with swappable operators, mirroring Hyunjung's setup.

    feature = final-RMSNorm output at the last non-padding position
    logits  = Linear(d_model -> 13)(feature)

The backbone runs through allpoly.forward.allpoly_hidden, so its non-polynomial
operators come from an ops object: ExactOps (the real model) or PolyOps (Brandon's
replacements). Swapping ops does not touch any weight.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from allpoly.forward import ExactOps, allpoly_hidden


class Classifier(nn.Module):
    def __init__(self, lm, n_classes: int, ops=None, chunk_size: int = 128, seed: int = 0):
        super().__init__()
        self.lm = lm
        d = lm.backbone.embedding.weight.shape[1]
        self.head = nn.Linear(d, n_classes)
        g = torch.Generator().manual_seed(seed)        # her PDF: N(0, 0.02^2), zero bias, seed 0
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(self.head.weight.shape, generator=g) * 0.02)
            self.head.bias.zero_()
        self.head.to(lm.backbone.embedding.weight.device)
        self.ops = ops if ops is not None else ExactOps()
        self.chunk_size = chunk_size

    def forward(self, ids, last_idx):
        h = allpoly_hidden(self.lm, ids, self.ops, chunk_size=self.chunk_size)   # (B, L, D)
        feat = h[torch.arange(h.shape[0], device=h.device), last_idx]
        return self.head(feat.float())


def subset_layers(lm, keep):
    """A copy of `lm` keeping only layers `keep` (her student: teacher layers 0,2,...,22).

    Renumbers each mixer's layer_idx to 0..n-1: operators are looked up by loop
    index while the mixer reports layer_idx to the collectors, and they must agree.
    """
    lm = copy.deepcopy(lm)
    layers = [lm.backbone.layers[i] for i in keep]
    lm.backbone.layers = nn.ModuleList(layers)
    for new_i, blk in enumerate(lm.backbone.layers):
        if hasattr(blk.mixer, "layer_idx"):
            blk.mixer.layer_idx = new_i
        if hasattr(blk, "layer_idx"):
            blk.layer_idx = new_i
    return lm
