#!/usr/bin/env python
"""Part 6 -- measure the REAL z = A*delta distribution in pretrained Mamba2-130M.

    python collect_transition_stats.py --blocks 32 --seq-len 1024

This is the most important measurement in the project and it has to happen
BEFORE we choose a polynomial. Part 4 showed that a polynomial is accurate
inside its fit interval and *divergent* outside it (degree 3 produced a = -45.8
when z reached -25). So the question "is degree 4 enough?" cannot be answered
without knowing where z actually lives.

The model is UNTOUCHED here: transition = exact exp. We only observe.

Memory: we keep streaming moments (Welford) plus a fixed-size reservoir sample
per layer for the percentiles. Nothing large is retained.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from baby_mamba.transition import ExactExp
from real_mamba.data import get_blocks
from real_mamba.model import DEFAULT_MODEL, iter_mixers, load_model, recommended_dtype
from real_mamba.patch import patched

PERCENTILES = (0.01, 0.1, 1.0, 50.0, 99.0, 99.9, 99.99)
Z_THRESHOLDS = (-2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -16.0, -20.0)


class HeadStats:
    """Per-HEAD statistics of z, kept exactly for min/max and by reservoir for
    percentiles.

    WHY PER HEAD MATTERS SO MUCH HERE: `A` is one scalar per head, and in the
    pretrained checkpoint those scalars span FIVE ORDERS OF MAGNITUDE
    (|A| from 4e-4 to 3.6e4). A single global interval is therefore hopeless:
    it must be wide enough for the most extreme head, and a degree-4 polynomial
    on a wide interval is useless for every other head. Since `A` is a *weight*,
    it is plaintext under FHE -- so per-head coefficients cost nothing extra, and
    each head can have its own interval. Hence these statistics.
    """

    def __init__(self, nheads: int, reservoir_per_head: int = 8192, seed: int = 0):
        self.nheads = nheads
        self.n = 0
        self.vmin = torch.full((nheads,), math.inf)
        self.vmax = torch.full((nheads,), -math.inf)
        self.sum = torch.zeros(nheads, dtype=torch.float64)
        self.sumsq = torch.zeros(nheads, dtype=torch.float64)
        self.cap = reservoir_per_head
        self.res = torch.zeros(nheads, self.cap, dtype=torch.float32)
        self.filled = 0
        self.g = torch.Generator().manual_seed(seed)

    def update(self, x: torch.Tensor):
        """x: (batch, seqlen, nheads) -> flattened to (k, nheads)."""
        v = x.detach().float().reshape(-1, self.nheads).cpu()
        self.vmin = torch.minimum(self.vmin, v.amin(dim=0))
        self.vmax = torch.maximum(self.vmax, v.amax(dim=0))
        self.sum += v.double().sum(dim=0)
        self.sumsq += (v.double() ** 2).sum(dim=0)
        k = v.shape[0]

        if self.filled < self.cap:
            take = min(self.cap - self.filled, k)
            self.res[:, self.filled:self.filled + take] = v[:take].T
            self.filled += take
            v, k = v[take:], k - take
            if k == 0:
                self.n += take
                return
        # vectorised reservoir replacement, same rule for every head
        seen = self.n
        for start in range(0, k, 4096):
            chunk = v[start:start + 4096]
            m = chunk.shape[0]
            draws = (torch.rand(m, self.nheads, generator=self.g)
                     * (seen + start + 1 + torch.arange(m)[:, None])).long()
            keep = draws < self.cap
            if keep.any():
                rows = torch.nonzero(keep, as_tuple=False)
                self.res[rows[:, 1], draws[keep]] = chunk[rows[:, 0], rows[:, 1]]
        self.n += k

    def to_dict(self) -> dict:
        n_per_head = max(self.n, 1)
        mean = (self.sum / n_per_head)
        # ALLOW-CLAMP: a variance computed as E[x^2]-E[x]^2 can come out at -1e-9
        # from float rounding. This guards a sqrt, it is not hiding instability.
        var = (self.sumsq / n_per_head - mean ** 2).clamp_min(0)
        sample = self.res[:, :self.filled]
        qs = torch.quantile(sample.double(),
                            torch.tensor([0.0001, 0.001, 0.01, 0.5, 0.99, 0.999], dtype=torch.float64),
                            dim=1) if self.filled else None
        out = {"count_per_head": int(self.n),
               "min": [float(v) for v in self.vmin],
               "max": [float(v) for v in self.vmax],
               "mean": [float(v) for v in mean],
               "std": [float(v) for v in var.sqrt()]}
        if qs is not None:
            for name, row in zip(("p0.01", "p0.1", "p1", "p50", "p99", "p99.9"), qs):
                out[name] = [float(v) for v in row]
        return out


class StreamStats:
    """min/max/mean/std by streaming, percentiles by reservoir sampling.

    Welford's algorithm for the moments (numerically stable over 10^8 samples),
    and a uniform reservoir so the percentiles are unbiased without storing
    everything.
    """

    def __init__(self, reservoir_size: int = 200_000, thresholds=(), seed: int = 0):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.vmin = math.inf
        self.vmax = -math.inf
        self.thresholds = tuple(thresholds)
        self.below = {t: 0 for t in self.thresholds}
        self.reservoir = np.empty(reservoir_size, dtype=np.float64)
        self.filled = 0
        self.rng = np.random.default_rng(seed)
        self.n_nan = 0
        self.n_inf = 0

    def update(self, x: torch.Tensor):
        v = x.detach().float().reshape(-1)
        self.n_nan += int(torch.isnan(v).sum())
        self.n_inf += int(torch.isinf(v).sum())
        v = v[torch.isfinite(v)]
        if v.numel() == 0:
            return
        a = v.cpu().numpy().astype(np.float64)
        k = a.size

        # Welford, batched form
        self.vmin = min(self.vmin, float(a.min()))
        self.vmax = max(self.vmax, float(a.max()))
        bmean = float(a.mean())
        bm2 = float(((a - bmean) ** 2).sum())
        delta = bmean - self.mean
        tot = self.n + k
        self.mean += delta * k / tot
        self.m2 += bm2 + delta ** 2 * self.n * k / tot
        self.n = tot

        for t in self.thresholds:
            self.below[t] += int((a < t).sum())

        # reservoir: fill, then replace with probability size/n
        cap = self.reservoir.size
        if self.filled < cap:
            take = min(cap - self.filled, k)
            self.reservoir[self.filled:self.filled + take] = a[:take]
            self.filled += take
            a, k = a[take:], k - take
            if k == 0:
                return
        seen_before = self.n - k
        idx = self.rng.integers(0, seen_before + np.arange(1, k + 1))
        keep = idx < cap
        if keep.any():
            self.reservoir[idx[keep]] = a[keep]

    def to_dict(self) -> dict:
        if self.n == 0:
            return {"count": 0}
        sample = self.reservoir[:self.filled]
        qs = np.percentile(sample, PERCENTILES) if sample.size else [float("nan")] * len(PERCENTILES)
        d = {
            "count": int(self.n),
            "min": self.vmin,
            "max": self.vmax,
            "mean": self.mean,
            "std": math.sqrt(self.m2 / max(self.n - 1, 1)),
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "reservoir_size": int(self.filled),
        }
        for p, q in zip(PERCENTILES, qs):
            d[f"p{p:g}"] = float(q)
        for t in self.thresholds:
            d[f"frac_lt_{t:g}"] = self.below[t] / self.n
        return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=("auto", "float32", "bfloat16", "float16"),
                    help="float32 is right for the pure-PyTorch `local` backend on any GPU. "
                         "Use 'auto' with --backend official (see recommended_dtype).")
    ap.add_argument("--data", default="wikitext2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=32, help="number of sequences to run")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--reservoir", type=int, default=200_000)
    ap.add_argument("--head-reservoir", type=int, default=8192,
                    help="reservoir size per head for the per-head percentiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("runs/part6_transition_stats"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    dtype = recommended_dtype(args.device) if args.dtype == 'auto' else getattr(torch, args.dtype)

    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    blocks, tok, info = get_blocks(args.data, args.split, args.seq_len)
    blocks = blocks[: args.blocks]

    n_layers = sum(1 for _ in iter_mixers(model))
    per_layer = {
        i: {"z": StreamStats(args.reservoir, Z_THRESHOLDS, args.seed + i),
            "a": StreamStats(args.reservoir, (0.0,), args.seed + 100 + i),
            "delta": StreamStats(args.reservoir, (), args.seed + 200 + i)}
        for i in range(n_layers)
    }
    head_stats = {i: HeadStats(next(iter_mixers(model))[1].nheads,
                               args.head_reservoir, args.seed + 300 + i)
                  for i in range(n_layers)}
    glob = {"z": StreamStats(args.reservoir * 2, Z_THRESHOLDS, args.seed + 7),
            "a": StreamStats(args.reservoir * 2, (0.0,), args.seed + 8),
            "delta": StreamStats(args.reservoir * 2, (), args.seed + 9)}

    # A is a parameter, so record it directly -- no sampling needed.
    A_per_layer = {}
    for idx, mixer in iter_mixers(model):
        A = (-torch.exp(mixer.A_log.float())).detach().cpu()
        A_per_layer[idx] = {
            "shape": list(A.shape), "min": float(A.min()), "max": float(A.max()),
            "mean": float(A.mean()), "std": float(A.std()), "values": [float(v) for v in A],
        }

    def collector(layer_idx, z, a):
        # delta = z / A is not recoverable per-element without A, so recompute it
        # from z using the layer's A (broadcast over heads).
        A = (-torch.exp(dict(iter_mixers(model))[layer_idx].A_log.float()))
        delta = z / A
        for key, val in (("z", z), ("a", a), ("delta", delta)):
            per_layer[layer_idx][key].update(val)
            glob[key].update(val)
        head_stats[layer_idx].update(z)

    print(f"\n[part6] running {len(blocks)} x {args.seq_len} tokens through {n_layers} layers "
          f"with the UNTOUCHED exp transition")
    t0 = time.time()
    with torch.no_grad(), patched(model, ExactExp(), chunk_size=args.chunk_size,
                                  collector=collector):
        for i in range(0, len(blocks), args.batch_size):
            model(blocks[i:i + args.batch_size].to(args.device))
            print(f"\r  {min(i + args.batch_size, len(blocks))}/{len(blocks)} blocks "
                  f"({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    out = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "data_info": info,
        "backend": backend,
        "n_layers": n_layers,
        "n_heads_per_layer": next(iter_mixers(model))[1].nheads,
        "A_per_layer": A_per_layer,
        "global": {k: v.to_dict() for k, v in glob.items()},
        "per_layer": {str(i): {k: v.to_dict() for k, v in d.items()}
                      for i, d in per_layer.items()},
        "per_head_z": {str(i): v.to_dict() for i, v in head_stats.items()},
    }
    (args.outdir / "transition_stats.json").write_text(json.dumps(out, indent=2) + "\n")

    # ---- the report ---------------------------------------------------------
    gz = out["global"]["z"]
    ga = out["global"]["a"]
    gd = out["global"]["delta"]
    print()
    print("=" * 78)
    print(f"GLOBAL, all {n_layers} layers, {gz['count']:,} samples of z")
    print("=" * 78)
    for label, d in (("delta = softplus(.)", gd), ("z = A*delta", gz), ("a = exp(z)", ga)):
        print(f"\n{label}")
        print(f"  min {d['min']:12.6g}   max {d['max']:12.6g}   "
              f"mean {d['mean']:12.6g}   std {d['std']:12.6g}")
        print("  percentiles: " + "  ".join(f"p{p:g}={d[f'p{p:g}']:.6g}" for p in PERCENTILES))
    print("\nA (a parameter, one scalar per head, 24 per layer)")
    amin = min(v["min"] for v in A_per_layer.values())
    amax = max(v["max"] for v in A_per_layer.values())
    n_heads = out["n_heads_per_layer"]
    print(f"  min {amin:.6g}   max {amax:.6g}   over all {n_layers * n_heads} heads "
          f"({n_layers} layers x {n_heads} heads)")

    print()
    print("=" * 78)
    print("HOW FAR NEGATIVE DOES z GO?  (this picks the polynomial interval)")
    print("=" * 78)
    for t in Z_THRESHOLDS:
        f = gz[f"frac_lt_{t:g}"]
        n = int(round(f * gz["count"]))
        print(f"  fraction(z < {t:6.1f}) = {f:.8f}   ({n:,} of {gz['count']:,})")

    print()
    print("=" * 78)
    print("PER LAYER (z)")
    print("=" * 78)
    print(f"{'layer':>5} {'min':>10} {'p0.01':>10} {'p1':>9} {'p50':>9} {'p99':>9} "
          f"{'max':>9} {'A min':>9} {'f(z<-8)':>10}")
    for i in range(n_layers):
        d = out["per_layer"][str(i)]["z"]
        print(f"{i:>5} {d['min']:10.4f} {d['p0.01']:10.4f} {d['p1']:9.4f} {d['p50']:9.4f} "
              f"{d['p99']:9.5f} {d['max']:9.5f} {A_per_layer[i]['min']:9.4f} "
              f"{d['frac_lt_-8']:10.2e}")

    # ---- the actionable conclusion -----------------------------------------
    print()
    print("=" * 78)
    print("WHAT INTERVAL SHOULD WE FIT ON?")
    print("=" * 78)
    covered = {}
    for t in Z_THRESHOLDS:
        covered[t] = 1.0 - gz[f"frac_lt_{t:g}"]
    print("  interval      fraction of z inside      z outside is where P(z) diverges")
    for t in Z_THRESHOLDS:
        bar = "#" * int(covered[t] * 40)
        print(f"  [{t:6.1f}, 0]   {covered[t]:.8f}   {bar}")
    print()
    print(f"  observed z min over this sample: {gz['min']:.4f}")
    print(f"  recommended --xmin (covers everything seen): {math.floor(gz['min'] * 1.1)}")
    print()
    print("  CAUTION: 'fraction inside' being 0.9999 is NOT reassuring. One token")
    print("  in 10,000 landing outside the interval is ~100 tokens per 1M, and a")
    print("  single a = -45 multiplies a whole head state by -45. Prefer an")
    print("  interval that covers the observed min with margin, and then check")
    print("  Part 8 for what actually happens.")
    print()
    print(f"wrote {args.outdir / 'transition_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
