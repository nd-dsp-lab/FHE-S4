#!/usr/bin/env python
"""Measure the inputs of every REMAINING non-polynomial gate in Mamba-2.

    python collect_gate_stats.py --blocks 16

WHY THIS COMES FIRST
--------------------
The exp gate went from "infinite perplexity" to "indistinguishable from exact"
purely because we measured its input range per head before fitting, instead of
guessing [-8, 0]. Fitting on a guessed interval was not slightly worse -- it was
a total failure.

So before writing a single polynomial for softplus, SiLU or RMSNorm, this script
records what their inputs actually are, per layer and per channel/head:

    dt_raw          the raw dt projection            -> for the FUSED gate
    softplus_in     dt_raw + dt_bias                 -> softplus's input
    silu_conv_in    post-conv1d pre-activation       -> SiLU #1 (1792 channels)
    silu_norm_in    the gate branch z                -> SiLU #2 (1536 channels)
    rmsnorm_meansq  mean(x^2)+eps inside RMSNorm     -> what 1/sqrt must cover

The model is UNTOUCHED: exact softplus, exact SiLU, exact RMSNorm, exact exp.
We only observe.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from baby_mamba.transition import ExactExp
from collect_transition_stats import PERCENTILES, HeadStats, StreamStats
from real_mamba.data import get_blocks
from real_mamba.model import DEFAULT_MODEL, iter_mixers, load_model, recommended_dtype
from real_mamba.patch import patch_transition

# name -> (is it per-channel?, reservoir per channel)
GATES = {
    "dt_raw":         (True, 2048),
    "softplus_in":    (True, 2048),
    "silu_conv_in":   (True, 512),
    "silu_norm_in":   (True, 512),
    "rmsnorm_meansq": (False, 0),
}


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
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("runs/gate_stats"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    blocks, _, info = get_blocks(args.data, args.split, args.seq_len)
    blocks = blocks[: args.blocks]
    n_layers = sum(1 for _ in iter_mixers(model))

    glob = {g: StreamStats(200_000, (), args.seed + i)
            for i, g in enumerate(GATES)}
    per_channel: dict[str, dict[int, HeadStats]] = {g: {} for g, (pc, _) in GATES.items() if pc}

    def gate_collector(layer_idx, name, tensor):
        if name not in GATES:
            return
        glob[name].update(tensor)
        per_ch, res = GATES[name]
        if per_ch:
            d = per_channel[name]
            if layer_idx not in d:
                d[layer_idx] = HeadStats(tensor.shape[-1], res, args.seed + layer_idx)
            d[layer_idx].update(tensor)

    print(f"\n[gates] {len(blocks)} x {args.seq_len} tokens through {n_layers} layers, "
          f"model UNTOUCHED (exact softplus / SiLU / RMSNorm / exp)")
    t0 = time.time()
    handle = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)
    for m in handle.mixers:
        m._gate_collector = gate_collector
    # re-bind so the collector reaches the reference forward
    from real_mamba.patch import mamba2_reference_forward
    for mixer in handle.mixers:
        def bound(u, _m=mixer, inference_params=None, **kw):
            return mamba2_reference_forward(_m, u, _m._transition,
                                            chunk_size=args.chunk_size,
                                            gate_collector=_m._gate_collector, **kw)
        mixer.forward = bound
    try:
        with torch.no_grad():
            for i in range(0, len(blocks), args.batch_size):
                model(blocks[i:i + args.batch_size].to(args.device))
                print(f"\r  {min(i + args.batch_size, len(blocks))}/{len(blocks)} "
                      f"({time.time() - t0:.0f}s)", end="", flush=True)
        print()
    finally:
        handle.restore()

    out = {"config": {k: (str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
           "backend": backend, "data_info": info, "n_layers": n_layers,
           "global": {g: v.to_dict() for g, v in glob.items()},
           "per_channel": {g: {str(l): st.to_dict() for l, st in d.items()}
                           for g, d in per_channel.items()}}
    (args.outdir / "gate_stats.json").write_text(json.dumps(out, indent=2) + "\n")

    # ---------------- report -------------------------------------------------
    print()
    print("=" * 96)
    print("WHAT RANGE WOULD EACH REPLACEMENT HAVE TO COVER?")
    print("=" * 96)
    hdr = (f"{'gate input':>16} {'min':>12} {'p0.1':>11} {'p50':>11} {'p99.9':>11} "
           f"{'max':>12} {'spread':>10}")
    print(hdr); print("-" * len(hdr))
    for g in GATES:
        d = out["global"][g]
        if not d.get("count"):
            print(f"{g:>16}   (never observed)"); continue
        spread = (d["max"] / d["min"]) if d["min"] > 0 else float("nan")
        print(f"{g:>16} {d['min']:12.4g} {d['p0.1']:11.4g} {d['p50']:11.4g} "
              f"{d['p99.9']:11.4g} {d['max']:12.4g} "
              + (f"{spread:10.3g}x" if math.isfinite(spread) else f"{'':>10}"))
    print("-" * len(hdr))

    # per-channel narrowing: the lever that made the exp gate work
    print()
    print("HOW MUCH DOES GOING PER-CHANNEL NARROW THE INTERVAL?")
    print(f"{'gate input':>16} {'global width':>13} {'median per-ch':>15} {'narrowing':>11}")
    for g, d in per_channel.items():
        if not d:
            continue
        gd = out["global"][g]
        gw = gd["max"] - gd["min"]
        widths = []
        for st in d.values():
            sd = st.to_dict()
            widths += [hi - lo for lo, hi in zip(sd["min"], sd["max"])]
        if not widths:
            continue
        med = sorted(widths)[len(widths) // 2]
        print(f"{g:>16} {gw:13.4g} {med:15.4g} {gw / max(med, 1e-30):10.1f}x")
    print()
    print("A large narrowing factor means the per-channel trick that rescued the")
    print("exp gate should work here too. A factor near 1 means it will not, and")
    print("that gate needs either a higher degree or a different approach.")
    print()
    print(f"wrote {args.outdir / 'gate_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
