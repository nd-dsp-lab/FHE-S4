#!/usr/bin/env python
"""PHASE 0b -- the PAIRED noise floor, which is the correct test for an operator swap.

    python eval/paired_significance.py --gate softplus --lengths 512

WHY THE UNPAIRED FLOOR IS THE WRONG YARDSTICK
---------------------------------------------
eval/noise_floor.py measures how much perplexity varies BETWEEN disjoint shards
of the same corpus with the model untouched: 2 sigma = 4.92 at L=512. That is a
real and useful number -- it says a single perplexity figure quoted to four
decimal places is meaningless as an absolute.

But it is not the error bar for the comparisons this project actually makes.
Those are PAIRED: identical tokens, identical weights, one operator swapped. The
shard-to-shard difficulty that produces the +-2.46 spread affects the exact and
the swapped model IDENTICALLY, so it cancels in the difference. Judging a paired
swap against an unpaired spread would declare almost any operator change
"not significant", which is permissive in our own favour and statistically
indefensible.

So this measures the paired statistic: per shard, evaluate both models on the
SAME tokens and take the difference. Report mean, std and 2 sigma OF THE
DIFFERENCE. That is the number a swap has to clear.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baby_mamba.transition import ExactExp
from real_mamba.data import get_tokenizer, load_text
from real_mamba.gates import build_per_channel_gates, build_squared_softplus
from real_mamba.model import DEFAULT_MODEL, load_model, recommended_dtype
from real_mamba.patch import mamba2_reference_forward, patch_transition
from real_mamba.transitions import build_per_head_transitions

from eval.noise_floor import make_shards, shard_nll

GATE_ARG = {"silu_conv": "silu_conv_gate", "silu_norm": "silu_norm_gate",
            "softplus": "softplus_gate"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=("auto","float32","bfloat16","float16"))
    ap.add_argument("--gate", nargs="+", default=["exp", "softplus"],
                    choices=["exp", "softplus", "silu_norm", "silu_conv"])
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--silu-margin", type=float, default=0.05)
    ap.add_argument("--gate-stats", type=Path, default=Path("runs/gate_stats/gate_stats.json"))
    ap.add_argument("--exp-stats", type=Path,
                    default=Path("runs/part6_transition_stats/transition_stats.json"))
    ap.add_argument("--data", default="wikitext2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512])
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eval/paired_significance.json"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    tok = get_tokenizer()
    ids = tok(load_text(args.data, args.split), return_tensors=None)["input_ids"]

    G = {}
    if any(g.startswith("silu") for g in args.gate):
        for k in ("silu_conv_in", "silu_norm_in"):
            G[k] = build_per_channel_gates(args.gate_stats, k, degree=args.degree,
                                           margin=args.silu_margin)
    if "softplus" in args.gate:
        G["softplus_in"] = build_squared_softplus(args.gate_stats,
                                                  q_degree=max(args.degree // 2, 1))
    exp_gate = (build_per_head_transitions(args.exp_stats, args.degree, pin_zero=True)
                if "exp" in args.gate else None)

    # The gate builders construct on CPU; the model is already on args.device.
    # Their buffers are registered, so nothing complains until the first forward
    # dies with "found at least two devices, cuda:0 and cpu" -- which is exactly
    # how the first GPU run of this script ended, after the exact baseline had
    # already spent 37s on all 16 shards. The forwards now also carry the device
    # on their casts, but doing it there would copy the coefficients host->device
    # on every call, so move them once, here.
    def _to_device(obj):
        # the builders return {layer_idx: Module}; be tolerant of a bare module
        # or a sequence too, since three different builders feed this
        if isinstance(obj, dict):
            for _m in obj.values():
                _to_device(_m)
        elif isinstance(obj, (list, tuple)):
            for _m in obj:
                _to_device(_m)
        elif hasattr(obj, "to"):
            obj.to(args.device)

    for _g in list(G.values()) + ([exp_gate] if exp_gate is not None else []):
        _to_device(_g)

    ph = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)

    def install(gate_name):
        kwmap = {}
        if gate_name == "softplus":
            kwmap["softplus_gate"] = G["softplus_in"]
        elif gate_name == "silu_norm":
            kwmap["silu_norm_gate"] = G["silu_norm_in"]
        elif gate_name == "silu_conv":
            kwmap["silu_conv_gate"] = G["silu_conv_in"]
        for mixer in ph.mixers:
            li = mixer.layer_idx
            kw = {a: g[li] for a, g in kwmap.items()}
            mixer._transition = exp_gate[li] if (gate_name == "exp" and exp_gate) else ExactExp()

            def bound(u, _m=mixer, _kw=kw, inference_params=None, **e):
                return mamba2_reference_forward(_m, u, _m._transition,
                                                 chunk_size=args.chunk_size, **_kw, **e)
            mixer.forward = bound

    out = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "backend": backend, "results": {}}

    for L in args.lengths:
        shards = make_shards(ids, args.shards, L)
        print(f"\n=== L={L}: {len(shards)} shards x {shards[0].shape[0]} blocks ===")
        install("exact_baseline")     # all ExactExp, no gates
        t0 = time.time()
        base = []
        for k, b in enumerate(shards):
            n, t = shard_nll(model, b, args.device, cfg.vocab_size)
            base.append(n / t)
            print(f"\r  exact  shard {k+1}/{len(shards)} ({time.time()-t0:.0f}s)", end="", flush=True)
        print()
        res = {}
        for gname in args.gate:
            install(gname)
            gl = []
            for k, b in enumerate(shards):
                n, t = shard_nll(model, b, args.device, cfg.vocab_size)
                gl.append(n / t)
                print(f"\r  {gname:9s} shard {k+1}/{len(shards)} ({time.time()-t0:.0f}s)",
                      end="", flush=True)
            print()
            # paired difference in PERPLEXITY, per shard
            dppl = [math.exp(g) - math.exp(b) for g, b in zip(gl, base)]
            dnll = [g - b for g, b in zip(gl, base)]
            sd = statistics.stdev(dppl) if len(dppl) > 1 else float("nan")
            mean = statistics.fmean(dppl)
            res[gname] = {
                "per_shard_delta_ppl": dppl, "per_shard_delta_nll": dnll,
                "mean_delta_ppl": mean, "std_delta_ppl": sd,
                "two_sigma_delta": 2 * sd,
                "sem_delta": sd / math.sqrt(len(dppl)) if len(dppl) > 1 else float("nan"),
                "significant_paired": abs(mean) > 2 * (sd / math.sqrt(len(dppl)))
                                       if len(dppl) > 1 else None,
                "n_shards": len(dppl),
                "exact_mean_ppl": statistics.fmean([math.exp(b) for b in base]),
            }
        out["results"][str(L)] = res
        install("exact_baseline")

    ph.restore()
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print()
    print("=" * 100)
    print("PAIRED significance -- same tokens, same weights, one operator swapped")
    print("=" * 100)
    hdr = (f"{'L':>6} {'gate':>10} {'mean d ppl':>12} {'std of d':>10} "
           f"{'SEM of d':>10} {'2*SEM':>9} {'verdict':>32}")
    print(hdr); print("-" * len(hdr))
    for L, res in out["results"].items():
        for g, r in res.items():
            v = ("SIGNIFICANT (paired)" if r["significant_paired"]
                 else "not distinguishable (paired)")
            print(f"{L:>6} {g:>10} {r['mean_delta_ppl']:+12.4f} {r['std_delta_ppl']:10.4f} "
                  f"{r['sem_delta']:10.4f} {2*r['sem_delta']:9.4f} {v:>32}")
    print("-" * len(hdr))
    print("Compare against the UNPAIRED floor (2 sigma = 4.92 at L=512): the paired")
    print("std of the difference is the correct error bar, and it is far smaller,")
    print("because shard difficulty cancels between the two models.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
