"""Does Mamba-2 still work when EVERY non-polynomial operator is a polynomial, with
CKKS bootstrap noise at every refresh? The go/no-go for end-to-end FHE.

    python -m allpoly.eval_allpoly --device cuda                 # full run (GPU, ~hours)
    python -m allpoly.eval_allpoly --device cpu --smoke          # tiny pipeline check

Protocol (each step exists because skipping it has cost this project before):
  1. CALIBRATE on wikitext-2 TRAIN with the exact model: every operator's input
     range, per channel / head / layer, plus each refresh point's max |x|.
     Measure before you fit.
  2. FIT every polynomial to those measured ranges (allpoly/forward.py: PolyOps).
  3. EVALUATE on wikitext-2 VALIDATION -- disjoint from calibration -- as PAIRED
     per-shard differences against the exact model on the same tokens. The
     unpaired spread is the wrong error bar for an operator swap (eval/README).
  4. Report, per config: mean d ppl +/- 2 SEM at EACH length separately, the
     fraction of inputs that fell OUTSIDE a fitted interval, and the OpenFHE depth.

Configs, in order:
  only:<op>                     -> one operator at a time: which operator costs what
  gates                         -> exp + softplus + both SiLUs, RMSNorm left exact
  all                           -> gates + all three RMSNorm sites
  exp/gates/all + noise@<b>     -> what bootstrap precision costs. exp+noise is the
                                   noise-only baseline: the smallest model that can run
                                   encrypted (exact exp + noise overflows; see below)
Noise levels default to the MEAN-abs precision measured in fhe/EPOCH.md at
N = 2^17: 11.35 bits (uniform secret) and 18.23 bits (sparse secret).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from allpoly.forward import (OPS, ExactOps, PolyOps, allpoly_forward, calibrate,
                             load_calibration, save_calibration)
from allpoly.ops import Refresher
from eval.noise_floor import make_shards
from real_mamba.data import get_blocks, get_tokenizer, load_text
from real_mamba.model import load_model

ALL_SITES = ("block_in", "scan_in_x", "scan_in_B", "scan_in_C", "scan_in_dt", "scan_out_y")


def shard_mean_nll(model, ops, refresh, blocks, device, vocab, chunk_size):
    tot, n = 0.0, 0
    for i in range(blocks.shape[0]):
        ids = blocks[i:i + 1].to(device)
        logits = allpoly_forward(model, ids, ops, refresh=refresh, chunk_size=chunk_size)
        lg = logits[:, :-1, :vocab]
        nll = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                                ids[:, 1:].reshape(-1), reduction="sum")
        if not torch.isfinite(nll):
            return float("inf")
        tot += float(nll)
        n += ids[:, 1:].numel()
    return tot / n


def range_report(calib: ExactOps) -> dict:
    """Max over layers of hi/lo for v at each norm site, overall vs by position."""
    out = {}
    for site in ("norm_pre", "norm_gated", "norm_f"):
        row = {}
        for suffix in ("", "@pos0", "@rest"):
            st = calib.stats.get(site + suffix, {})
            if not st:
                continue
            ratios = [float(d["max"].max() / d["min"].min()) for d in st.values()]
            row[(suffix or "@all").lstrip("@")] = {"max_ratio_over_layers": max(ratios),
                                                   "median_ratio": float(np.median(ratios))}
        out[site] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="local", choices=("local", "official", "auto"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--calib-blocks", type=int, default=64,
                    help="calibration blocks of --calib-len tokens; Part 6 fitted the exp gate on 64")
    ap.add_argument("--calib-len", type=int, default=2048)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--degree-silu", type=int, default=27)
    ap.add_argument("--degree-silu-conv", type=int, default=None,
                    help="SiLU after conv1d; defaults to --degree-silu. The two sites need not match")
    ap.add_argument("--degree-silu-norm", type=int, default=None,
                    help="SiLU inside the gated norm; defaults to --degree-silu")
    ap.add_argument("--degree-softplus-q", type=int, default=13)
    ap.add_argument("--degree-invsqrt", type=int, default=27)
    ap.add_argument("--invsqrt-target", type=float, default=2.0 ** -12)
    ap.add_argument("--margin", type=float, default=0.05, help="per-channel interval widening, fraction of width")
    ap.add_argument("--v-margin", type=float, default=2.0, help="multiplicative widening of v's range")
    ap.add_argument("--noise-bits", type=float, nargs="*", default=[11.35, 18.23])
    ap.add_argument("--refresh-sites", nargs="+", default=list(ALL_SITES))
    ap.add_argument("--refresh-per-tensor", action="store_true",
                    help="one noise scale per tensor instead of per channel (pessimistic; for comparison)")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset of config names to run (default: all)")
    ap.add_argument("--calib-cache", type=Path, default=None,
                    help="load calibration from here if it exists, else compute and save it; "
                         "a degree sweep should calibrate once")
    ap.add_argument("--max-blocks-per-shard", type=int, default=None,
                    help="cap blocks per shard (default: all). The shards split the WHOLE validation "
                         "set, so without a cap a 2-shard run is ~250k tokens")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU check: 2 calib blocks, 2 shards x 3 blocks, L=64")
    ap.add_argument("--out", type=Path, default=Path("allpoly/results/allpoly.json"))
    args = ap.parse_args()
    if args.smoke:
        args.calib_blocks, args.calib_len, args.shards, args.lengths = 2, 128, 2, [64]
        args.chunk_size = 32
        args.max_blocks_per_shard = 3

    if args.smoke:
        print("=" * 88)
        print("SMOKE TEST: calibration on 256 tokens, 6 evaluation blocks. This only checks that")
        print("the pipeline RUNS. With so little calibration data, inputs fall outside the fitted")
        print("ranges, so verdicts and warnings below are expected to look bad and MEAN NOTHING.")
        print("It passes if it reaches 'wrote ...' with no traceback.")
        print("=" * 88)
    torch.manual_seed(0)
    dev = args.device
    model, cfg, backend = load_model(backend=args.backend, device=dev, dtype=torch.float32)
    model.eval()
    vocab = cfg.vocab_size
    out = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "backend": backend, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1. calibrate --------------------------------------------------------
    t0 = time.time()
    if args.calib_cache is not None and args.calib_cache.exists():
        calib = load_calibration(args.calib_cache)
        nblk = None
        print(f"[calib] loaded {args.calib_cache}")
    else:
        cblocks = get_blocks("wikitext2", "train", args.calib_len, verbose=False)[0][: args.calib_blocks]
        calib = calibrate(model, cblocks, dev, chunk_size=args.chunk_size)
        nblk = int(cblocks.shape[0])
        if args.calib_cache is not None:
            args.calib_cache.parent.mkdir(parents=True, exist_ok=True)
            save_calibration(calib, args.calib_cache)
    out["calibration"] = {"blocks": nblk, "len": args.calib_len, "cache": str(args.calib_cache),
                          "seconds": round(time.time() - t0, 1),
                          "v_range_by_position": range_report(calib)}
    print(f"[calib] done in {time.time() - t0:.0f}s")
    print(json.dumps(out["calibration"]["v_range_by_position"], indent=1))

    # 2. configs ----------------------------------------------------------
    pk = dict(degree_silu=args.degree_silu, degree_silu_conv=args.degree_silu_conv,
              degree_silu_norm=args.degree_silu_norm, degree_softplus_q=args.degree_softplus_q,
              degree_invsqrt=args.degree_invsqrt, invsqrt_target=args.invsqrt_target,
              margin=args.margin, v_margin=args.v_margin)
    configs = []
    for op in OPS:
        configs.append((f"only:{op}", (op,), None))
    # "gates" = the four gates with RMSNorm left exact (RMSNorm is a separate track);
    # "all"   = gates + all three RMSNorm sites.
    GATES = ("exp", "softplus", "silu_conv", "silu_norm")
    configs.append(("gates", GATES, None))
    configs.append(("all", OPS, None))
    # Noise-only baseline is "exp+noise", NOT "exact+noise". With EXACT exp, a tiny
    # negative error on Delta at a token where Delta ~ 0, times |A| up to ~2.6e5,
    # gives z ~ +158 and exp(+158) = inf. The circuit never evaluates exact exp --
    # it evaluates the bounded degree-4 polynomial, where the same error moves a by
    # ~1e-3. So exact+noise is a configuration that cannot occur under FHE.
    for b in args.noise_bits:
        configs.append((f"exp+noise@{b:g}", ("exp",), b))
        configs.append((f"gates+noise@{b:g}", GATES, b))
        configs.append((f"all+noise@{b:g}", OPS, b))
    if args.configs:
        configs = [c for c in configs if c[0] in set(args.configs)]

    ids = get_tokenizer()(load_text("wikitext2", "validation"), return_tensors=None)["input_ids"]
    exact = ExactOps()
    out["results"] = {}
    for L in args.lengths:
        shards = make_shards(ids, args.shards, L)
        if args.max_blocks_per_shard:
            shards = [s[: args.max_blocks_per_shard] for s in shards]
        print(f"\n=== L={L}: {len(shards)} shards x {shards[0].shape[0]} blocks ===")
        exact.reset_decay()
        with torch.no_grad():
            base = [shard_mean_nll(model, exact, None, b, dev, vocab, args.chunk_size) for b in shards]
        base_ppl = np.exp(base)
        print(f"  exact (explicit forward) ppl {base_ppl.mean():.4f}   decay a_max {exact.a_max:.4f}")
        res = {"exact_ppl_mean": float(base_ppl.mean()), "exact_decay": exact.decay_report(),
               "configs": {}}
        for name, which, bits in configs:
            t1 = time.time()
            ops = PolyOps(calib, which, **pk).to(dev) if which else exact
            rf = (Refresher(bits, calib.absmax, args.refresh_sites, seed=1,
                            per_tensor=args.refresh_per_tensor)
                  if bits is not None else None)
            if isinstance(ops, PolyOps):
                ops.reset_counts()
            else:
                ops.reset_decay()
            with torch.no_grad():
                nll = [shard_mean_nll(model, ops, rf, b, dev, vocab, args.chunk_size) for b in shards]
            ppl = np.exp(nll)
            d = ppl - base_ppl
            finite = bool(np.all(np.isfinite(d)))
            sem = float(d.std(ddof=1) / math.sqrt(len(d))) if finite and len(d) > 1 else float("nan")
            mean = float(d.mean()) if finite else float("inf")
            verdict = ("BROKEN (inf/nan)" if not finite else
                       "not distinguishable" if abs(mean) <= 2 * sem else "SIGNIFICANT")
            r = {"d_ppl_mean": mean, "d_ppl_2sem": 2 * sem, "verdict": verdict,
                 "ppl_mean": float(ppl.mean()) if finite else float("inf"),
                 "seconds": round(time.time() - t1, 1)}
            r["decay"] = ops.decay_report()
            if isinstance(ops, PolyOps):
                r["operators"] = ops.report()
            res["configs"][name] = r
            fo = max((v.get("frac_out_max", 0.0) for v in r.get("operators", {}).values()), default=0.0)
            dr = r["decay"]
            # Flag on the FRACTION only. a_max is a maximum over ~1e7-1e8 decays and is
            # set by one rare token outside the fitted interval, so it grows with how
            # much text is evaluated: same calibration, 8x more text took it from 1.05
            # to 17.3 while frac(a>1) stayed ~1e-7 (2026-10-14). The exp gate alone sits
            # at 1e-8..1e-7; with 11.35-bit noise ~5.5e-3.
            flag = ("  <-- frac(a>1) far above baseline: recurrence can expand"
                    if dr["frac_a_gt_1"] > 1e-2 and not args.smoke else "")
            if args.smoke:
                verdict = "(smoke: ignore)"
            print(f"  {name:22s} d ppl {mean:+10.4f} +/- {2 * sem:8.4f}  {verdict:20s} "
                  f"frac_out {fo:.1e}  a>1 {dr['frac_a_gt_1']:.1e} (max {dr['a_max']:.3f})  "
                  f"a<0 {dr['frac_a_lt_0']:.1e}  ({r['seconds']:.0f}s){flag}", flush=True)
            out["results"][str(L)] = res
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\nwrote {args.out}" + ("   <- SMOKE TEST PASSED (pipeline runs)" if args.smoke else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
