#!/usr/bin/env python
"""PHASE 1 -- what do the three RMSNorm instances actually see?

    python norm/collect_norm_stats.py --lengths 512 2048 --blocks 24

Records the distribution of the mean-square argument `v` at every norm
instance, per layer, at each sequence length:

    median, p1, p99, min, max, and the ratio p99/p1

This serves two purposes at once:
  * the per-layer median of sqrt(v) INITIALISES the learned constant in Path A
  * the per-layer range is the spec Path B's prescale has to cover

and it answers the question Path A lives or dies on: **is the distribution
wider at L=2048 than at L=512?** If the argument drifts with sequence length, a
single static constant cannot serve both, and that is a real limitation of the
whole approach rather than a tuning problem.

THE THREE SITES (the brief said 3 per block; the model has 2 per block + 1)
    norm_pre     the residual-stream RMSNorm before each mixer      x24
    norm_gated   the gated RMSNorm inside each mixer, before out_proj  x24
    norm_f       the final RMSNorm before the LM head               x1
                                                                  --- 49 total
There is no second block norm because mamba2-130m has d_intermediate=0, so the
MLP branch is an Identity and carries no norm of its own.

Two different capture mechanisms are needed, and it matters:
  * `norm_pre` and `norm_f` are invoked as MODULES, so a forward pre-hook sees
    their input directly.
  * `norm_gated` is invoked as a FUNCTION from the patched mixer forward
    (`rms_norm_gated_ref(...)`), so a module hook never fires. It is captured
    through the mixer's existing gate_collector instead. Its argument is also
    NOT the raw input: with norm_before_gate=False the norm is applied to
    `x * silu(z)`, so the mean-square is taken after gating.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baby_mamba.transition import ExactExp
from real_mamba.data import get_blocks
from real_mamba.model import DEFAULT_MODEL, iter_mixers, load_model, recommended_dtype
from real_mamba.patch import mamba2_reference_forward, patch_transition

QUANTILES = (0.0, 0.01, 0.5, 0.99, 1.0)


class Reservoir:
    """Fixed-memory sample of one scalar stream, for quantiles."""

    def __init__(self, cap=200_000, seed=0):
        self.cap, self.n, self.filled = cap, 0, 0
        self.buf = np.empty(cap, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.vmin, self.vmax = np.inf, -np.inf

    def update(self, t):
        a = t.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return
        self.vmin = min(self.vmin, float(a.min()))
        self.vmax = max(self.vmax, float(a.max()))
        k = a.size
        if self.filled < self.cap:
            take = min(self.cap - self.filled, k)
            self.buf[self.filled:self.filled + take] = a[:take]
            self.filled += take
            a, k = a[take:], k - take
            if k == 0:
                self.n += take
                return
        idx = self.rng.integers(0, self.n + np.arange(1, k + 1))
        keep = idx < self.cap
        if keep.any():
            self.buf[idx[keep]] = a[keep]
        self.n += k

    def summary(self):
        if self.filled == 0:
            return {"count": 0}
        s = self.buf[:self.filled]
        q = np.quantile(s, QUANTILES)
        p1, p99 = float(q[1]), float(q[3])
        return {"count": int(self.n), "min": self.vmin, "max": self.vmax,
                "p1": p1, "median": float(q[2]), "p99": p99,
                "p99_over_p1": (p99 / p1) if p1 > 0 else float("inf"),
                "max_over_min": (self.vmax / self.vmin) if self.vmin > 0 else float("inf"),
                "sqrt_median": float(np.sqrt(q[2]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=("auto", "float32", "bfloat16", "float16"))
    ap.add_argument("--data", default="wikitext2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--blocks", type=int, default=24, help="sequences per length")
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("norm/norm_stats.json"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    eps = cfg.norm_epsilon

    # locate the module-invoked norms by their qualified name
    pre_norms, final_norm = {}, None
    for name, mod in model.named_modules():
        if not hasattr(mod, "weight") or not hasattr(mod, "eps"):
            continue
        if name.endswith("norm_f"):
            final_norm = (name, mod)
        elif name.endswith(".norm") and ".mixer" not in name and ".layers." in name:
            pre_norms[int(name.split(".layers.")[1].split(".")[0])] = (name, mod)
    print(f"\n[phase1] found {len(pre_norms)} residual pre-norms, "
          f"{'1' if final_norm else '0'} final norm, "
          f"{sum(1 for _ in iter_mixers(model))} gated norms "
          f"-> {len(pre_norms) + (1 if final_norm else 0) + sum(1 for _ in iter_mixers(model))}"
          f" instances total")

    out = {"config": {k: (str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
           "backend": backend, "eps": eps, "by_length": {}}

    for L in args.lengths:
        blocks, _, info = get_blocks(args.data, args.split, L, verbose=False)
        blocks = blocks[: args.blocks]
        R = {"norm_pre": {}, "norm_gated": {}, "norm_f": {}}

        def res(site, layer):
            d = R[site]
            if layer not in d:
                d[layer] = Reservoir(seed=args.seed + hash((site, layer)) % 1000)
            return d[layer]

        # --- module pre-hooks for norm_pre and norm_f ----------------------
        handles = []
        for layer, (name, mod) in pre_norms.items():
            def hook(m, a, _l=layer):
                x = a[0].detach().float()
                res("norm_pre", _l).update(x.square().mean(dim=-1) + eps)
            handles.append(mod.register_forward_pre_hook(hook))
        if final_norm:
            def hook_f(m, a):
                x = a[0].detach().float()
                res("norm_f", 0).update(x.square().mean(dim=-1) + eps)
            handles.append(final_norm[1].register_forward_pre_hook(hook_f))

        # --- gate_collector for norm_gated (a FUNCTION, not a module) ------
        def gate_collector(layer_idx, name, tensor):
            if name == "rmsnorm_meansq":
                res("norm_gated", layer_idx).update(tensor)

        ph = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)
        for mixer in ph.mixers:
            def bound(u, _m=mixer, inference_params=None, **kw):
                return mamba2_reference_forward(_m, u, _m._transition,
                                                 chunk_size=args.chunk_size,
                                                 gate_collector=gate_collector, **kw)
            mixer.forward = bound

        t0 = time.time()
        try:
            with torch.no_grad():
                for i in range(blocks.shape[0]):
                    model(blocks[i:i + 1].to(args.device))
                    print(f"\r  L={L:>5} {i + 1}/{blocks.shape[0]} "
                          f"({time.time() - t0:.0f}s)", end="", flush=True)
            print()
        finally:
            for h in handles:
                h.remove()
            ph.restore()

        out["by_length"][str(L)] = {
            site: {str(l): r.summary() for l, r in sorted(d.items())}
            for site, d in R.items()
        }

    args.out.write_text(json.dumps(out, indent=2) + "\n")

    # ------------------- report ------------------------------------------
    print()
    print("=" * 104)
    print("PHASE 1 -- the mean-square argument v at each RMSNorm instance")
    print("=" * 104)
    for L, bl in out["by_length"].items():
        print(f"\n--- L={L} ---")
        hdr = (f"{'site':>11} {'layers':>7} {'min v':>11} {'median v':>11} "
               f"{'max v':>11} {'median p99/p1':>14} {'worst p99/p1':>13} {'max/min':>11}")
        print(hdr); print("-" * len(hdr))
        for site, d in bl.items():
            if not d:
                continue
            meds = [v["median"] for v in d.values() if v.get("count")]
            r99 = [v["p99_over_p1"] for v in d.values() if v.get("count")]
            mn = min(v["min"] for v in d.values() if v.get("count"))
            mx = max(v["max"] for v in d.values() if v.get("count"))
            print(f"{site:>11} {len(d):>7} {mn:11.4g} {np.median(meds):11.4g} "
                  f"{mx:11.4g} {np.median(r99):14.2f} {max(r99):13.2f} "
                  f"{mx / mn if mn > 0 else float('inf'):11.4g}")

    # the drift question Path A lives or dies on
    if len(out["by_length"]) > 1:
        Ls = sorted(out["by_length"], key=int)
        lo, hi = Ls[0], Ls[-1]
        print()
        print("=" * 104)
        print(f"DRIFT: does the argument move between L={lo} and L={hi}?")
        print("=" * 104)
        print("A single learned CONSTANT divisor can only serve both lengths if these")
        print("agree. A large shift means per-length constants, which is a real")
        print("limitation of Path A rather than a tuning problem.")
        print(f"\n{'site':>11} {'layer':>6} {f'median v @{lo}':>16} {f'median v @{hi}':>16} "
              f"{'ratio':>9} {'p99/p1 shift':>13}")
        for site in ("norm_pre", "norm_gated", "norm_f"):
            a, b = out["by_length"][lo].get(site, {}), out["by_length"][hi].get(site, {})
            common = sorted(set(a) & set(b), key=int)
            if not common:
                continue
            ratios = []
            for l in common:
                if not (a[l].get("count") and b[l].get("count")):
                    continue
                ratios.append(b[l]["median"] / max(a[l]["median"], 1e-30))
            if not ratios:
                continue
            worst = max(common, key=lambda l: abs(np.log(
                max(b[l]["median"], 1e-30) / max(a[l]["median"], 1e-30))))
            print(f"{site:>11} {'median':>6} {'':>16} {'':>16} "
                  f"{np.median(ratios):9.3f} {'':>13}")
            print(f"{'':>11} {worst:>6} {a[worst]['median']:16.4g} "
                  f"{b[worst]['median']:16.4g} "
                  f"{b[worst]['median'] / max(a[worst]['median'], 1e-30):9.3f} "
                  f"{b[worst]['p99_over_p1'] / max(a[worst]['p99_over_p1'], 1e-30):13.3f}"
                  f"   <- worst layer")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
