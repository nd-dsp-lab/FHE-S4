#!/usr/bin/env python
"""Replace the remaining non-polynomial gates one at a time, and measure.

    python eval_gates.py --degree 4 --blocks 40

Each gate is swapped in ALONE first, so its cost is attributable, then
cumulatively. Reports perplexity and the per-layer ct-ct depth, because depth
accumulates across the 24 blocks and that is the currency that matters.

Not covered here: RMSNorm's 1/sqrt. Measurement says its argument spans
6.1e+06x (0.0019 to 11,260 on mamba2-130m), so a single low-degree polynomial
cannot cover it and it needs Newton/Goldschmidt with range reduction. Separate job.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from baby_mamba.transition import ExactExp
from real_mamba.data import get_blocks
from real_mamba.eval_lm import evaluate, peak_memory_gb, reset_peak_memory
from real_mamba.gates import build_per_channel_gates, build_squared_softplus
from real_mamba.model import DEFAULT_MODEL, iter_mixers, load_model, recommended_dtype
from real_mamba.patch import mamba2_reference_forward, patch_transition
from real_mamba.transitions import build_per_head_transitions

GATE_ARG = {"silu_conv_in": "silu_conv_gate",
            "silu_norm_in": "silu_norm_gate",
            "softplus_in": "softplus_gate"}


class GateProbe:
    """Tracks the invariant each replacement can break."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_dt = 0
        self.n_dt_neg = 0
        self.dt_min = math.inf

    def note_delta(self, dt):
        d = dt.detach().float()
        self.n_dt += d.numel()
        self.n_dt_neg += int((d < 0).sum())
        self.dt_min = min(self.dt_min, float(d.min()))

    def to_dict(self):
        return {"delta_min": self.dt_min,
                "frac_delta_lt_0": self.n_dt_neg / max(self.n_dt, 1)}


def install(model, handle, transition, gates: dict, chunk_size, probe):
    """Bind the reference forward with a given set of per-layer gate modules."""
    for mixer in handle.mixers:
        li = mixer.layer_idx
        kw = {arg: gset[li] for arg, gset in gates.items() if li in gset}

        def bound(u, _m=mixer, _kw=kw, inference_params=None, **extra):
            out = mamba2_reference_forward(_m, u, _m._transition,
                                           chunk_size=chunk_size, **_kw, **extra)
            return out
        mixer._transition = transition[li] if isinstance(transition, dict) else transition
        mixer.forward = bound


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
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.15,
                    help="interval widening for softplus")
    ap.add_argument("--silu-margin", type=float, default=0.6,
                    help="interval widening for the SiLUs. Measured: on only 6 blocks "
                         "the observed (max - p99.9) tail already exceeds a 0.15 margin "
                         "for ~77%% of channels, and outside its interval a polynomial "
                         "diverges rather than degrading.")
    ap.add_argument("--softplus-form", default="squared",
                    choices=("squared", "direct"),
                    help="squared: Delta = Q(x)^2, non-negative BY CONSTRUCTION. "
                         "direct: a plain polynomial fit -- which gives inf perplexity, "
                         "because 415/576 heads then emit a negative Delta.")
    ap.add_argument("--gate-stats", type=Path,
                    default=Path("runs/gate_stats/gate_stats.json"))
    ap.add_argument("--exp-stats", type=Path,
                    default=Path("runs/part6_transition_stats/transition_stats.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("runs/gates_eval"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    blocks, _, info = get_blocks(args.data, args.split, args.seq_len)
    blocks = blocks[: args.blocks]

    print(f"\nfitting gates at degree {args.degree} from measured per-channel ranges")
    G = {k: build_per_channel_gates(args.gate_stats, k, degree=args.degree,
                                    margin=args.silu_margin)
         for k in ("silu_conv_in", "silu_norm_in")}
    if args.softplus_form == "squared":
        G["softplus_in"] = build_squared_softplus(args.gate_stats,
                                                  q_degree=max(args.degree // 2, 1),
                                                  margin=args.margin)
    else:
        G["softplus_in"] = build_per_channel_gates(args.gate_stats, "softplus_in",
                                                   degree=args.degree,
                                                   margin=args.margin)
    exp_gate = build_per_head_transitions(args.exp_stats, args.degree, pin_zero=True)
    dep = {k: v[0].ct_ct_depth for k, v in G.items()}
    dep["exp"] = exp_gate[0].ct_ct_depth
    for k, v in dep.items():
        print(f"  {k:15s} depth {v}")

    probe = GateProbe()
    handle = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)
    rows = []

    # (label, exp gate, which extra gates, per-layer depth on the critical path)
    # critical path before the scan = max( silu_conv , softplus -> exp )
    def crit(has_sp, has_exp, has_sc, has_sn):
        pre = max(dep["silu_conv_in"] if has_sc else 0,
                  (dep["softplus_in"] if has_sp else 0) + (dep["exp"] if has_exp else 0))
        return pre + (dep["silu_norm_in"] if has_sn else 0)

    configs = [
        ("baseline: everything exact",        False, False, False, False),
        ("exp only (the previous result)",    False, True,  False, False),
        ("SiLU after conv1d only",            False, False, True,  False),
        ("SiLU in gated norm only",           False, False, False, True),
        ("softplus only",                     True,  False, False, False),
        ("exp + SiLU(conv)",                  False, True,  True,  False),
        ("exp + softplus",                    True,  True,  False, False),
        ("ALL FOUR",                          True,  True,  True,  True),
    ]

    for label, sp, ex, sc, sn in configs:
        gates = {}
        if sp: gates[GATE_ARG["softplus_in"]] = G["softplus_in"]
        if sc: gates[GATE_ARG["silu_conv_in"]] = G["silu_conv_in"]
        if sn: gates[GATE_ARG["silu_norm_in"]] = G["silu_norm_in"]
        install(model, handle, exp_gate if ex else ExactExp(), gates,
                args.chunk_size, probe)
        reset_peak_memory(args.device)
        r = evaluate(model, blocks, device=args.device, batch_size=args.batch_size,
                     vocab_size=cfg.vocab_size)
        d = crit(sp, ex, sc, sn)
        rows.append({"config": label, "softplus": sp, "exp": ex,
                     "silu_conv": sc, "silu_norm": sn,
                     "degree": args.degree,
                     "per_layer_ct_ct_depth": d,
                     "network_depth_24_layers": d * 24,
                     "loss": r["loss"], "perplexity": r["perplexity"],
                     "peak_gpu_gb": peak_memory_gb(args.device)})
        print(f"  {label:34s} ppl {r['perplexity']:10.4f}   per-layer depth {d}")

    handle.restore()
    base = rows[0]["perplexity"]
    for r in rows:
        r["delta_ppl"] = r["perplexity"] - base

    print()
    print("=" * 104)
    print(f"REPLACING THE REMAINING GATES -- degree {args.degree}, per-channel intervals, "
          f"no training")
    print(f"{args.model} ({backend}) | {info['data']}/{info['split']} | "
          f"{len(blocks)} x {args.seq_len} tokens")
    print("=" * 104)
    hdr = (f"{'configuration':>34} {'ppl':>11} {'delta ppl':>11} "
           f"{'depth/layer':>12} {'x24 layers':>11}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['config']:>34} {r['perplexity']:11.4f} {r['delta_ppl']:+11.4f} "
              f"{r['per_layer_ct_ct_depth']:12d} {r['network_depth_24_layers']:11d}")
    print("-" * len(hdr))
    print("depth/layer counts the CRITICAL PATH: max(SiLU-conv, softplus->exp) + SiLU-norm.")
    print("softplus and exp compose in series, which is why fusing them into one")
    print("per-head polynomial (real_mamba/gates.py:FusedDtGate) is worth testing.")
    print("RMSNorm's 1/sqrt is NOT included -- its argument spans 6.1e+06x.")

    cols = list(rows[0].keys())
    with (args.outdir / "gates_eval.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    (args.outdir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"backend": backend, "gate_depths": dep}, indent=2) + "\n")
    (args.outdir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.outdir}/gates_eval.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
