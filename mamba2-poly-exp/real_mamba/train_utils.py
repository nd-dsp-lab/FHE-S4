"""Shared training machinery for Parts 9 and 10.

Kept in one place so that the fine-tuning and distillation scripts cannot drift
apart in how they select parameters, log, or measure memory.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from baby_mamba.polynomial import PerHeadPolyExp, PolyExp
from real_mamba.model import iter_mixers


# =============================================================================
# reproducibility
# =============================================================================

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Deterministic where it is free. We do NOT set
        # torch.use_deterministic_algorithms(True) because it makes some reduction
        # kernels 5-10x slower for no scientific gain here; the seed already fixes
        # data order and initialisation.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Part 9's three progressive modes
# =============================================================================
# The rule from the brief: do not full-fine-tune unless the lightweight versions
# clearly fail. So each mode unfreezes strictly more than the last, and the
# report says exactly how many parameters were touched.
#
#   MODE A  polynomial coefficients FROZEN.
#           Trainable: the parameters immediately around the transition --
#           dt_bias and A_log (the two things that produce z), plus D.
#           ~1.7k parameters. Interpretation: "can the model re-tune WHERE it
#           evaluates the polynomial, without changing the polynomial?"
#
#   MODE B  polynomial coefficients TRAINABLE, plus everything in MODE A.
#           Interpretation: "can the polynomial move to suit the model?"
#           Note this can leave the Chebyshev/minimax guarantee behind -- the
#           coefficients become whatever the loss wants, so we re-measure
#           max|P-exp| and frac(a<0) afterwards instead of assuming.
#
#   MODE C  MODE B plus LoRA on the projections adjacent to the transition
#           (in_proj, out_proj). Still a small fraction of the 129M parameters.
#
# `full` exists only as an escape hatch for the write-up and is not the default.

TRANSITION_LOCAL_PARAMS = ("dt_bias", "A_log", "D")


def _poly_modules(model):
    holder = getattr(model, "fhe_transitions", None)
    if holder is None:
        return []
    return [m for m in holder.modules() if isinstance(m, (PolyExp, PerHeadPolyExp))]


class LoRALinear(nn.Module):
    """y = W x + (B A x) * (alpha / r), with W frozen.

    Deliberately minimal: no dropout, no bias adaptation, no merging. Wraps an
    existing nn.Linear in place so the pretrained weight object is reused, not
    copied.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero, so the wrapped module is EXACTLY the pretrained one at
        # step 0. Any change in loss is then attributable to training, not to the
        # wrapping itself.
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model, r: int = 8, alpha: float = 16.0,
               targets=("in_proj", "out_proj")) -> int:
    """Wrap the named linear layers of every Mamba-2 mixer with LoRA. Returns count."""
    n = 0
    for _, mixer in iter_mixers(model):
        for name in targets:
            mod = getattr(mixer, name, None)
            if isinstance(mod, nn.Linear):
                setattr(mixer, name, LoRALinear(mod, r=r, alpha=alpha))
                n += 1
    return n


def select_trainable(model, mode: str, lora_r: int = 8, lora_alpha: float = 16.0,
                     verbose: bool = True, skip_poly: bool = False) -> dict:
    """Freeze everything, then unfreeze exactly what `mode` allows.

    skip_poly: used by the exact-exp control run, which has no polynomial
    coefficients to train but should otherwise get the same trainable set.
    """
    if mode not in ("A", "B", "C", "full"):
        raise ValueError("--mode must be A, B, C or full")

    for p in model.parameters():
        p.requires_grad_(False)

    groups: dict[str, list] = {"transition_local": [], "poly_coeffs": [], "lora": [], "all": []}

    if mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
            groups["all"].append(p)
    else:
        # the parameters that PRODUCE z, in every layer
        for _, mixer in iter_mixers(model):
            for name in TRANSITION_LOCAL_PARAMS:
                p = getattr(mixer, name, None)
                if isinstance(p, nn.Parameter):
                    p.requires_grad_(True)
                    groups["transition_local"].append(p)
        if mode in ("B", "C") and not skip_poly:
            for pm in _poly_modules(model):
                if not isinstance(pm.coeffs, nn.Parameter):
                    raise RuntimeError(
                        f"--mode {mode} needs trainable polynomial coefficients. "
                        "Pass --trainable-poly so the transition is built with them."
                    )
                pm.coeffs.requires_grad_(True)
                groups["poly_coeffs"].append(pm.coeffs)
            if not groups["poly_coeffs"] and not skip_poly:
                raise RuntimeError(
                    f"--mode {mode} found no polynomial coefficients. Are you running "
                    "with --transition exact?"
                )
        if mode == "C":
            n = apply_lora(model, r=lora_r, alpha=lora_alpha)
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)
                    groups["lora"].append(p)
            if verbose:
                print(f"[train] wrapped {n} linear layers with LoRA (r={lora_r}, alpha={lora_alpha})")

    counts = {k: sum(p.numel() for p in v) for k, v in groups.items() if v}
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {"mode": mode, "trainable_groups": counts,
            "n_trainable": total_trainable, "n_total": total,
            "trainable_fraction": total_trainable / max(total, 1)}
    if verbose:
        print(f"[train] MODE {mode}: {total_trainable:,} trainable of {total:,} "
              f"({100 * info['trainable_fraction']:.4f}%)")
        for k, v in counts.items():
            print(f"          {k:18s} {v:,}")
    return info


# =============================================================================
# data batching by token budget
# =============================================================================

def token_budget_batches(blocks: torch.Tensor, token_budget: int, micro_batch: int,
                         seed: int = 0, shuffle: bool = True):
    """Yield micro-batches, cycling the data until `token_budget` tokens are seen.

    Reports the number of epochs implied so that a large budget on a small corpus
    is visible rather than silently repeating.
    """
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(blocks.shape[0], generator=g) if shuffle \
        else torch.arange(blocks.shape[0])
    seq_len = blocks.shape[1]
    seen, i = 0, 0
    while seen < token_budget:
        idx = []
        for _ in range(micro_batch):
            if i >= len(order):
                order = torch.randperm(blocks.shape[0], generator=g) if shuffle \
                    else torch.arange(blocks.shape[0])
                i = 0
            idx.append(int(order[i]))
            i += 1
        batch = blocks[idx]
        seen += batch.numel()
        yield batch, seen


# =============================================================================
# logging
# =============================================================================

@dataclass
class RunLogger:
    outdir: Path
    fields: list = field(default_factory=lambda: [
        "step", "tokens_seen", "loss", "lm_loss", "kd_loss", "transition_loss",
        "lr", "grad_norm", "seconds", "peak_gpu_gb"])

    def __post_init__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.path = self.outdir / "training_log.csv"
        self.f = self.path.open("w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fields)
        self.w.writeheader()
        self.t0 = time.time()

    def log(self, **kw):
        row = {k: kw.get(k, "") for k in self.fields}
        row["seconds"] = round(time.time() - self.t0, 3)
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def poly_health(model) -> dict:
    """After training in MODE B/C the coefficients are no longer a Chebyshev fit.
    Re-measure what they became, rather than quoting the fit-time numbers."""
    from baby_mamba.polynomial import approximation_report
    out = {}
    for i, pm in enumerate(_poly_modules(model)):
        d = pm.to_dict()
        if isinstance(pm, PolyExp) and pm.interval:
            d["error_report_after_training"] = approximation_report(
                pm.coeff_list(), *pm.interval)
        out[f"{pm.name}"] = d
    return out


# =============================================================================
# post-training transition health (feeds results_summary.csv)
# =============================================================================

@torch.no_grad()
def probe_transition_stats(model, handle, blocks, device="cpu", n_blocks=4) -> dict:
    """Re-run a few blocks and record what `a` actually looks like now.

    Part 9/10 change the parameters that produce `z` (and in MODE B the
    polynomial itself), so the fit-time diagnostics are stale afterwards. This
    measures the real thing.
    """
    import math as _math

    state = {"n": 0, "amin": _math.inf, "amax": -_math.inf,
             "lt0": 0, "gt1": 0, "nan": 0, "inf": 0, "zmin": _math.inf}

    def collector(layer_idx, z, a):
        af = a.detach().float()
        state["n"] += af.numel()
        state["nan"] += int(torch.isnan(af).sum())
        state["inf"] += int(torch.isinf(af).sum())
        fin = af[torch.isfinite(af)]
        if fin.numel():
            state["amin"] = min(state["amin"], float(fin.min()))
            state["amax"] = max(state["amax"], float(fin.max()))
            state["lt0"] += int((fin < 0).sum())
            state["gt1"] += int((fin > 1).sum())
        state["zmin"] = min(state["zmin"], float(z.detach().float().min()))

    previous = [m.__dict__.get("forward") for m in handle.mixers]
    from real_mamba.patch import mamba2_reference_forward
    for mixer in handle.mixers:
        def bound(u, _m=mixer, inference_params=None, **kw):
            return mamba2_reference_forward(_m, u, _m._transition, collector=collector, **kw)
        mixer.forward = bound
    try:
        model.eval()
        model(blocks[:n_blocks].to(device))
    finally:
        for mixer, prev in zip(handle.mixers, previous):
            if prev is None:
                mixer.__dict__.pop("forward", None)
                # reinstall the patched forward without the collector
                def bound2(u, _m=mixer, inference_params=None, **kw):
                    return mamba2_reference_forward(_m, u, _m._transition, **kw)
                mixer.forward = bound2
            else:
                mixer.forward = prev
    n = max(state["n"], 1)
    return {"transition_min": state["amin"], "transition_max": state["amax"],
            "frac_transition_lt_0": state["lt0"] / n, "frac_transition_gt_1": state["gt1"] / n,
            "frac_transition_nan": state["nan"] / n, "z_min_observed": state["zmin"],
            "n_transition_values": state["n"]}
