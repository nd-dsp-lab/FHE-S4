#!/usr/bin/env python
"""Part 7 -- drop in the polynomial, train NOTHING, measure the damage.

    python eval_poly_exp.py --transition poly4
    python eval_poly_exp.py --transition poly4 --interval-mode per-head
    python eval_poly_exp.py --sweep                       # the whole table at once

Everything except `z -> a` is the pretrained model, unchanged. This is the
honest "how bad is it before we do anything clever" number, and it is the
baseline every later part is compared against.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from baby_mamba.transition import ExactExp
from real_mamba.data import get_blocks
from real_mamba.eval_lm import evaluate, peak_memory_gb, reset_peak_memory
from real_mamba.model import DEFAULT_MODEL, iter_mixers, load_model, recommended_dtype
from real_mamba.patch import patch_transition, set_transition
from real_mamba.transitions import (
    add_interval_mode_args,
    add_transition_args,
    depth_estimate_any,
    describe_any,
    transition_from_args,
)


class TransitionProbe:
    """Records the diagnostics Part 8 cares about while we evaluate."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.a_min = math.inf
        self.a_max = -math.inf
        self.n_lt0 = 0
        self.n_gt1 = 0
        self.n_nan = 0
        self.n_inf = 0
        self.z_min = math.inf

    def __call__(self, layer_idx, z, a):
        af = a.detach().float()
        self.n += af.numel()
        self.n_nan += int(torch.isnan(af).sum())
        self.n_inf += int(torch.isinf(af).sum())
        finite = af[torch.isfinite(af)]
        if finite.numel():
            self.a_min = min(self.a_min, float(finite.min()))
            self.a_max = max(self.a_max, float(finite.max()))
            self.n_lt0 += int((finite < 0).sum())
            self.n_gt1 += int((finite > 1).sum())
        self.z_min = min(self.z_min, float(z.detach().float().min()))

    def to_dict(self) -> dict:
        n = max(self.n, 1)
        return {"transition_min": self.a_min, "transition_max": self.a_max,
                "frac_transition_lt_0": self.n_lt0 / n,
                "frac_transition_gt_1": self.n_gt1 / n,
                "frac_transition_nan": self.n_nan / n,
                "frac_transition_inf": self.n_inf / n,
                "z_min_observed": self.z_min,
                "n_transition_values": self.n}


SWEEP = [
    ("exact", None, "global", False),
    ("poly2", 2, "global", False),
    ("poly3", 3, "global", False),
    ("poly4", 4, "global", False),
    ("poly2", 2, "global", True),
    ("poly3", 3, "global", True),
    ("poly4", 4, "global", True),
    ("poly2", 2, "per-head", True),
    ("poly3", 3, "per-head", True),
    ("poly4", 4, "per-head", True),
]


def build_parser():
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
    ap.add_argument("--blocks", type=int, default=None, help="limit the number of eval blocks")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="evaluate exact + poly2/3/4 in all interval modes, one table")
    ap.add_argument("--outdir", type=Path, default=Path("runs/part7_zero_training"))
    add_transition_args(ap)
    add_interval_mode_args(ap)
    return ap


def one_run(model, blocks, args, transition, probe, vocab_size, handle=None):
    probe.reset()
    reset_peak_memory(args.device)
    if handle is None:
        handle = patch_transition(model, transition, chunk_size=args.chunk_size,
                                  collector=probe)
        owned = True
    else:
        set_transition(handle, transition)
        owned = False
    try:
        t0 = time.time()
        res = evaluate(model, blocks, device=args.device, batch_size=args.batch_size,
                       vocab_size=vocab_size, progress=True)
        res["wall_seconds"] = time.time() - t0
        res["peak_gpu_memory_gb"] = peak_memory_gb(args.device)
        res.update(probe.to_dict())
        return res, handle
    finally:
        if owned:
            pass          # caller decides when to restore


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    model, cfg, backend = load_model(args.model, args.backend, args.device,
                                     recommended_dtype(args.device) if args.dtype == 'auto' else getattr(torch, args.dtype))
    blocks, tok, info = get_blocks(args.data, args.split, args.seq_len)
    if args.blocks:
        blocks = blocks[: args.blocks]
    n_layers = sum(1 for _ in iter_mixers(model))

    probe = TransitionProbe()
    rows = []
    handle = None

    combos = SWEEP if args.sweep else [(args.transition,
                                        None,
                                        args.interval_mode,
                                        args.pin_zero)]
    print()
    for spec, _deg, mode, pin in combos:
        sub = argparse.Namespace(**vars(args))
        sub.transition, sub.interval_mode, sub.pin_zero = spec, mode, pin
        try:
            transition = transition_from_args(sub)
        except SystemExit as e:
            print(f"[skip] {spec}/{mode}: {e}")
            continue
        desc = describe_any(transition)
        label = f"{spec}" + ("" if mode == "global" else "/per-head") + ("/pin0" if pin else "")
        print(f"=== {label} "
              + (f"(interval [{sub.xmin:g},{sub.xmax:g}])" if mode == "global" and spec != "exact"
                 else "(per-head intervals)" if mode != "global" else "")
              + " ===")
        res, handle = one_run(model, blocks, args, transition, probe, cfg.vocab_size, handle)
        rows.append({
            "transition": spec,
            "degree": desc.get("degree") or "",
            "interval_mode": mode if spec != "exact" else "",
            "interval": (f"[{sub.xmin:g},{sub.xmax:g}]" if mode == "global" and spec != "exact"
                         else ("per-head" if spec != "exact" else "")),
            "pin_zero": pin if spec != "exact" else "",
            "fit_method": args.fit_method if spec != "exact" else "",
            "fhe_depth_estimate": depth_estimate_any(transition),
            "validation_loss": res["loss"],
            "perplexity": res["perplexity"],
            "transition_min": res["transition_min"],
            "transition_max": res["transition_max"],
            "frac_transition_lt_0": res["frac_transition_lt_0"],
            "frac_transition_gt_1": res["frac_transition_gt_1"],
            "frac_transition_nan": res["frac_transition_nan"],
            "z_min_observed": res["z_min_observed"],
            "n_degenerate_heads": desc.get("n_degenerate_heads_total",
                                           desc.get("n_degenerate_heads", "")),
            "wall_seconds": res["wall_seconds"],
            "peak_gpu_memory_gb": res["peak_gpu_memory_gb"],
            "_desc": desc,
        })
        print(f"    loss {res['loss']:.4f}   ppl {res['perplexity']:.4f}\n")

    if handle is not None:
        handle.restore()

    # ---- delta vs exact ----------------------------------------------------
    base = next((r for r in rows if r["transition"] == "exact"), None)
    for r in rows:
        if base is None:
            r["delta_perplexity"] = ""
            r["delta_loss"] = ""
        else:
            r["delta_perplexity"] = r["perplexity"] - base["perplexity"]
            r["delta_loss"] = r["validation_loss"] - base["validation_loss"]

    # ---- the table ---------------------------------------------------------
    print("=" * 118)
    print("PART 7 -- POLYNOMIAL TRANSITION, ZERO TRAINING")
    print(f"model {args.model} ({backend} backend, {args.dtype}) | "
          f"{info['data']}/{info['split']} | {len(blocks)} x {args.seq_len} tokens | "
          f"{n_layers} layers")
    print("=" * 118)
    hdr = (f"{'transition':>12} {'deg':>4} {'interval':>10} {'pin0':>5} {'FHE depth':>10} "
           f"{'val loss':>9} {'ppl':>11} {'delta ppl':>11} {'min a':>10} {'frac a<0':>9} {'frac a>1':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        dp = (f"{r['delta_perplexity']:+11.4f}" if isinstance(r["delta_perplexity"], float)
              else f"{'':>11}")
        print(f"{r['transition']:>12} {str(r['degree']):>4} {str(r['interval']):>10} "
              f"{str(r['pin_zero']):>5} {r['fhe_depth_estimate']:>10} "
              f"{r['validation_loss']:9.4f} {r['perplexity']:11.4f} {dp} "
              f"{r['transition_min']:10.4f} {r['frac_transition_lt_0']:9.5f} "
              f"{r['frac_transition_gt_1']:9.5f}")
    print("-" * len(hdr))
    print("'min a' below 0 or 'frac a>1' above 0 means the decay left (0,1] -- something")
    print("exp can never do. Part 8 measures what that does to the state norms.")

    # ---- save --------------------------------------------------------------
    cols = [k for k in rows[0] if not k.startswith("_")]
    with (args.outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in cols} for r in rows])
    (args.outdir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"backend": backend, "data_info": info}, indent=2) + "\n")
    (args.outdir / "metrics.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "_desc"} for r in rows], indent=2) + "\n")
    (args.outdir / "polynomial_coefficients.json").write_text(json.dumps(
        {r["transition"] + "/" + str(r["interval_mode"]) + ("/pin0" if r["pin_zero"] else ""):
         r["_desc"] for r in rows}, indent=2) + "\n")
    print(f"\nwrote {args.outdir}/results.csv (+ config.json, metrics.json, "
          f"polynomial_coefficients.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
