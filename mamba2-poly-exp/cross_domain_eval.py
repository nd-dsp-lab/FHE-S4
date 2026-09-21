#!/usr/bin/env python
"""E1 -- do per-head intervals fitted on ONE corpus survive OTHERS?

    python cross_domain_eval.py --fit-data wikitext103 --eval-data wikitext2 pile lambada

THE RISK THIS EXISTS TO TEST
----------------------------
Part 6 measures the distribution of z on some corpus, Part 7 fits one polynomial
per head to the interval that measurement implies, and Part 7 then evaluates on
the same corpus. If the intervals are really a property of the MODEL, that is
fine. If they are partly a property of the DATA, then the headline result
("a depth-1 quadratic costs +0.035 perplexity") is an artefact of measuring and
evaluating in the same place, and it would collapse on anything else.

`z = A * delta`. `A` is a weight, so it cannot move with the data. `delta` is
`softplus(proj(u) + dt_bias)`, so it can. This script measures how much that
matters.

For every held-out corpus it reports, per degree:
  * perplexity, and delta vs the exact model ON THAT SAME CORPUS
  * the fraction of z values that land OUTSIDE the interval their head was
    fitted on -- the direct measure of interval staleness
  * frac(a<0), frac(a>1), min a, and any NaN

Nothing is fitted on the evaluation corpora. The intervals come from --fit-data
only, and are written out so the whole thing is reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from baby_mamba.polynomial import PerHeadPolyExp
from baby_mamba.transition import ExactExp
from real_mamba.data import DATASETS, get_blocks
from real_mamba.eval_lm import evaluate, peak_memory_gb, reset_peak_memory
from real_mamba.model import DEFAULT_MODEL, iter_mixers, load_model, recommended_dtype
from real_mamba.patch import patch_transition, set_transition
from real_mamba.transitions import build_per_head_transitions


class IntervalProbe:
    """Counts how often z lands outside the interval its head was fitted on."""

    def __init__(self, transitions: dict):
        # (layer, head) -> (lo, hi) of the interval that head's polynomial covers
        self.iv = {l: t.intervals for l, t in transitions.items()}
        self.reset()

    def reset(self):
        self.n = 0
        self.n_below = 0          # z < lo  -- the dangerous side, polynomial diverges
        self.n_above = 0          # z > hi
        self.a_min = math.inf
        self.a_max = -math.inf
        self.n_lt0 = 0
        self.n_gt1 = 0
        self.n_nan = 0
        self.z_min = math.inf
        self.per_layer_below = {}

    def __call__(self, layer_idx, z, a):
        zf = z.detach().float()
        af = a.detach().float()
        ivs = self.iv.get(layer_idx)
        if ivs is not None:
            lo = torch.tensor([v[0] for v in ivs], device=zf.device)
            hi = torch.tensor([v[1] for v in ivs], device=zf.device)
            below = (zf < lo).sum()
            above = (zf > hi).sum()
            self.n_below += int(below)
            self.n_above += int(above)
            self.per_layer_below[layer_idx] = (
                self.per_layer_below.get(layer_idx, 0) + int(below))
        self.n += zf.numel()
        self.z_min = min(self.z_min, float(zf.min()))
        self.n_nan += int(torch.isnan(af).sum())
        fin = af[torch.isfinite(af)]
        if fin.numel():
            self.a_min = min(self.a_min, float(fin.min()))
            self.a_max = max(self.a_max, float(fin.max()))
            self.n_lt0 += int((fin < 0).sum())
            self.n_gt1 += int((fin > 1).sum())

    def to_dict(self) -> dict:
        n = max(self.n, 1)
        return {
            "n_z_values": self.n,
            "frac_z_below_interval": self.n_below / n,
            "frac_z_above_interval": self.n_above / n,
            "frac_z_outside_interval": (self.n_below + self.n_above) / n,
            "z_min": self.z_min,
            "a_min": self.a_min, "a_max": self.a_max,
            "frac_a_lt_0": self.n_lt0 / n, "frac_a_gt_1": self.n_gt1 / n,
            "frac_a_nan": self.n_nan / n,
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=("auto", "float32", "bfloat16", "float16"))
    ap.add_argument("--fit-data", default="wikitext103",
                    help="corpus the per-head intervals are MEASURED on")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--fit-blocks", type=int, default=64)
    ap.add_argument("--eval-data", nargs="+", default=["wikitext2", "pile", "lambada"],
                    choices=DATASETS)
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--eval-blocks", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--fit-method", default="chebyshev")
    ap.add_argument("--pin-zero", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("runs/e1_cross_domain"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)

    # ---- 1. measure z on the FIT corpus only -------------------------------
    stats_path = args.outdir / f"stats_fitted_on_{args.fit_data}.json"
    print(f"\n[E1] measuring z on {args.fit_data}/{args.fit_split} "
          f"({args.fit_blocks} x {args.seq_len} tokens) -- the ONLY corpus the "
          f"intervals see")
    import collect_transition_stats as cts
    cts.main([
        "--model", args.model, "--backend", backend, "--device", args.device,
        "--dtype", args.dtype if args.dtype != "auto" else "float32",
        "--data", args.fit_data, "--split", args.fit_split,
        "--blocks", str(args.fit_blocks), "--seq-len", str(args.seq_len),
        "--chunk-size", str(args.chunk_size), "--seed", str(args.seed),
        "--outdir", str(args.outdir / "fit_stats"),
    ])
    (args.outdir / "fit_stats" / "transition_stats.json").replace(stats_path)

    # ---- 2. build one set of per-head polynomials from those stats ---------
    transitions = {d: build_per_head_transitions(
        stats_path, d, method=args.fit_method, pin_zero=args.pin_zero,
        margin=args.margin) for d in args.degrees}
    (args.outdir / "polynomial_coefficients.json").write_text(json.dumps(
        {f"poly{d}": {str(k): v.to_dict() for k, v in t.items()}
         for d, t in transitions.items()}, indent=2) + "\n")

    # ---- 3. evaluate on every corpus, WITHOUT refitting --------------------
    probes = {d: IntervalProbe(t) for d, t in transitions.items()}
    rows = []
    handle = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)

    for corpus in args.eval_data:
        split = "test" if corpus == "lambada" else args.eval_split
        try:
            blocks, _, info = get_blocks(corpus, split, args.seq_len)
        except Exception as e:                                   # noqa: BLE001
            print(f"[skip] {corpus}: {type(e).__name__}: {e}")
            continue
        blocks = blocks[: args.eval_blocks]
        print(f"\n=== {corpus}/{split}: {blocks.shape[0]} x {args.seq_len} tokens ===")

        # exact baseline ON THIS CORPUS -- the only fair reference. Comparing a
        # polynomial's perplexity on `pile` against the exact model's perplexity
        # on `wikitext2` would be meaningless; each corpus gets its own baseline.
        set_transition(handle, ExactExp())
        reset_peak_memory(args.device)
        base = evaluate(model, blocks, device=args.device, batch_size=args.batch_size,
                        vocab_size=cfg.vocab_size)
        print(f"  exact : loss {base['loss']:.4f}  ppl {base['perplexity']:.4f}")
        rows.append({"corpus": corpus, "transition": "exact", "degree": "",
                     "loss": base["loss"], "perplexity": base["perplexity"],
                     "delta_ppl": 0.0, "n_eval_tokens": info["n_tokens"],
                     **{k: "" for k in ("frac_z_outside_interval", "frac_z_below_interval",
                                        "a_min", "frac_a_lt_0", "frac_a_gt_1", "z_min")}})

        for d in args.degrees:
            probes[d].reset()
            # re-patch with the probe attached for this degree
            handle.restore()
            handle = patch_transition(model, transitions[d], chunk_size=args.chunk_size,
                                      collector=probes[d])
            r = evaluate(model, blocks, device=args.device, batch_size=args.batch_size,
                         vocab_size=cfg.vocab_size)
            p = probes[d].to_dict()
            print(f"  poly{d} : loss {r['loss']:.4f}  ppl {r['perplexity']:.4f}  "
                  f"(delta {r['perplexity'] - base['perplexity']:+.4f})  "
                  f"z outside interval {p['frac_z_outside_interval']:.2e}  "
                  f"min a {p['a_min']:.4f}  frac a>1 {p['frac_a_gt_1']:.2e}")
            rows.append({"corpus": corpus, "transition": f"poly{d}", "degree": d,
                         "loss": r["loss"], "perplexity": r["perplexity"],
                         "delta_ppl": r["perplexity"] - base["perplexity"],
                         "n_eval_tokens": info["n_tokens"], **p})
            handle.restore()
            handle = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)

    handle.restore()

    # ---- 4. report ---------------------------------------------------------
    print()
    print("=" * 112)
    print(f"E1 -- PER-HEAD INTERVALS FITTED ON {args.fit_data.upper()}, EVALUATED ELSEWHERE")
    print(f"model {args.model} ({backend}, {dtype}) | margin {args.margin} | "
          f"no refitting on any eval corpus")
    print("=" * 112)
    hdr = (f"{'corpus':>12} {'trans':>7} {'ppl':>10} {'delta ppl':>10} "
           f"{'z outside iv':>13} {'z below iv':>11} {'min a':>10} {'frac a<0':>9} {'frac a>1':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        f = lambda k, w=9, p=2: (f"{r[k]:{w}.{p}e}" if isinstance(r[k], float) else f"{'':>{w}}")  # noqa: E731
        print(f"{r['corpus']:>12} {r['transition']:>7} {r['perplexity']:10.4f} "
              f"{r['delta_ppl']:+10.4f} {f('frac_z_outside_interval',13)} "
              f"{f('frac_z_below_interval',11)} "
              + (f"{r['a_min']:10.4f}" if isinstance(r['a_min'], float) else f"{'':>10}")
              + f" {f('frac_a_lt_0')} {f('frac_a_gt_1')}")
    print("-" * len(hdr))
    print("'z outside iv' is the number that matters: it is the fraction of tokens whose")
    print("z left the interval its head's polynomial was fitted on. Outside that interval")
    print("a polynomial diverges rather than decaying (Part 4), so a large value here")
    print("would mean the per-head result does not transfer across domains.")

    # union of keys: the `exact` rows carry fewer fields than the poly rows
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with (args.outdir / "cross_domain.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader(); w.writerows(rows)
    (args.outdir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"backend": backend, "dtype": str(dtype)}, indent=2) + "\n")
    (args.outdir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.outdir}/cross_domain.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
