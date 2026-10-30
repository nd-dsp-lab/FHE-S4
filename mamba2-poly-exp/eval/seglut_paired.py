#!/usr/bin/env python
"""ROSETTA-style segmented LUTs on the gates polynomials could not handle -- PAIRED test.

    python eval/seglut_paired.py --gate invsqrt fused_dt softplus silu_conv silu_norm all \\
        --oob clamp --shards 16 --lengths 512 2048

WHAT IS BEING TESTED
--------------------
Each gate below is evaluated by real_mamba/seglut.py, a plaintext simulation of
the segmented lookup table in ROSETTA (Yu et al., CCS '26). The model is the
pretrained mamba2-130m, untouched; one family of operators is swapped at a time,
then all of them together, and the PAIRED perplexity difference is reported --
the same statistic, shards and verdict rule as eval/paired_significance.py, so
the rows are directly comparable with the polynomial results in GATES.md.

    invsqrt_pre    1/sqrt(.) in the 24 block pre-norms            1 scalar/token each
    invsqrt_gated  1/sqrt(.) in the 24 gated norms                1 scalar/token each
    invsqrt_f      1/sqrt(.) in norm_f                            1 scalar/token
    invsqrt        all 49 of the above
    fused_dt       a = exp(A_h softplus(x + b_h)), one table/head 24 scalars/token
    softplus       Delta = softplus(x), one table/head            24 scalars/token
    silu_conv      SiLU after conv1d, one table/channel           1792/token
    silu_norm      SiLU inside the gated norm, one table/channel  1536/token
    all            every one of the above at once -- every non-arithmetic
                   operator in the mixer and the norms

In the polynomial work `fused_dt` cost +134 ppl, softplus +0.59, SiLU-norm +4.55,
SiLU-conv +5.85, all four together inf, and RMSNorm was never replaced at all.

RANGES
------
SiLU, softplus and dt tables take their per-channel domain from
runs/gate_stats/gate_stats.json -- the SAME file the polynomial fits used, so
the comparison is like for like. The norms have no entry there, so their
per-instance range of v = mean(x^2) + eps is measured here first, on the same
blocks collect_gate_stats.py used (--calib-split validation, 6 x 1024 tokens).
That calibration set is a subset of the evaluation text, as it was for the
polynomials; the fraction of evaluation inputs that fall OUTSIDE each table is
reported, so the effect of the narrow calibration is visible rather than hidden.

PAIRING
-------
The baseline runs through the same replacement modules in exact mode, with
fused_add_norm switched off for both, so the two sides of every pair differ in
the swapped operator and in nothing else. A one-block logit check against the
untouched model is printed first.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baby_mamba.transition import ExactExp
from norm.install_norms import NormHandle, _unfuse_add_norm, find_norm_sites
from real_mamba.data import get_blocks, get_tokenizer, load_text
from real_mamba.model import DEFAULT_MODEL, load_model, recommended_dtype
from real_mamba.patch import mamba2_reference_forward, patch_transition
from real_mamba.seglut import (
    OOB_MODES,
    SegLUTGatedNorm,
    SegLUTRMSNorm,
    build_seglut,
    frac,
    summarize_precision,
)

from eval.noise_floor import make_shards, shard_nll

FAMILIES = ("norm_pre", "norm_gated", "norm_f", "fused_dt", "softplus",
            "silu_conv", "silu_norm")
GATE_FAMILIES = {
    "exact": (),
    "invsqrt_pre": ("norm_pre",),
    "invsqrt_gated": ("norm_gated",),
    "invsqrt_f": ("norm_f",),
    "invsqrt": ("norm_pre", "norm_gated", "norm_f"),
    "fused_dt": ("fused_dt",),
    "softplus": ("softplus",),
    "silu_conv": ("silu_conv",),
    "silu_norm": ("silu_norm",),
    "all": FAMILIES,
}
# ROSETTA's choices: log-spaced for 1/x and 1/sqrt x, uniform for activations
AUTO_BOUNDARIES = {"invsqrt": "log", "silu": "uniform", "softplus": "uniform",
                   "fused_dt": "uniform"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=("auto", "float32", "bfloat16", "float16"))
    ap.add_argument("--gate", nargs="+", default=["invsqrt", "fused_dt", "softplus",
                                                  "silu_conv", "silu_norm", "all"],
                    choices=[g for g in GATE_FAMILIES if g != "exact"])
    # the table
    ap.add_argument("--n-seg", type=int, default=4, help="ROSETTA uses 4")
    ap.add_argument("--n-seg-invsqrt", type=int, default=None,
                    help="override --n-seg for the 1/sqrt tables only")
    ap.add_argument("--n-br", type=int, default=2048, help="blind-rotation ring degree")
    ap.add_argument("--boundaries", default="auto",
                    choices=("auto", "uniform", "log", "adaptive"),
                    help="auto = log for 1/sqrt, uniform otherwise (ROSETTA's choice); "
                         "log only applies to 1/sqrt")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--oob", default="clamp", choices=OOB_MODES)
    ap.add_argument("--noise-bits", type=float, default=None)
    ap.add_argument("--noise-scale", default="abs", choices=("abs", "range", "rel"))
    ap.add_argument("--cmp-noise-bits", type=float, default=None)
    # the 1/sqrt tables as OpenFHE builds them (real_mamba/ss_invsqrt.py)
    ap.add_argument("--invsqrt-impl", default="seglut", choices=("seglut", "openfhe"),
                    help="openfhe = the fhe/seglut_invsqrt pipeline: ~5.5-bit table + "
                         "Newton; the noise/oob flags above then do not apply to 1/sqrt")
    ap.add_argument("--newton", type=int, default=2)
    ap.add_argument("--fhew-noise", type=float, default=9.7,
                    help="index and phase noise in table positions; sqrt(n/18) = 8.5 at "
                         "STD128, 9.7 the worst segment bench.cpp measured")
    ap.add_argument("--ckks-bits", type=float, default=15.3)
    ap.add_argument("--guard", type=float, default=0.05)
    # data
    ap.add_argument("--gate-stats", type=Path, default=Path("runs/gate_stats/gate_stats.json"))
    ap.add_argument("--calib-split", default="validation")
    ap.add_argument("--calib-blocks", type=int, default=6)
    ap.add_argument("--calib-seq-len", type=int, default=1024)
    ap.add_argument("--data", default="wikitext2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512])
    ap.add_argument("--max-blocks-per-shard", type=int, default=None,
                    help="truncate shards (smoke tests only)")
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eval/seglut_paired.json"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    model.eval()
    tok = get_tokenizer()
    ids = tok(load_text(args.data, args.split), return_tensors=None)["input_ids"]
    calib, _, calib_info = get_blocks(args.data, args.calib_split, args.calib_seq_len,
                                      max_tokens=args.calib_blocks * args.calib_seq_len,
                                      tokenizer=tok)
    calib = calib[: args.calib_blocks]

    def bkind(fn):
        if args.boundaries == "auto":
            return AUTO_BOUNDARIES[fn]
        if args.boundaries == "log" and fn != "invsqrt":
            return "uniform"
        return args.boundaries

    lut_kw = dict(n_br=args.n_br, margin=args.margin, oob=args.oob,
                  noise_bits=args.noise_bits, noise_scale=args.noise_scale,
                  cmp_noise_bits=args.cmp_noise_bits)
    ss_kw = dict(newton=args.newton, index_noise=args.fhew_noise, phase_noise=args.fhew_noise,
                 ckks_bits=args.ckks_bits, guard=args.guard)

    # ------------------------------------------------------------------ install
    ph = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)
    probe = calib[:1, :256].to(args.device)
    with torch.no_grad():
        ref_logits = model(probe).logits.float()

    nh = NormHandle(model=model)
    _unfuse_add_norm(model, nh)
    sites = find_norm_sites(model)
    NP, NF, GN = {}, {}, {}
    for layer, (parent, attr, old) in sorted(sites["norm_pre"].items()):
        NP[layer] = SegLUTRMSNorm(old.weight.data, getattr(old, "eps", 1e-5))
        setattr(parent, attr, NP[layer])
        nh.originals.append((parent, attr, old))
    for layer, (parent, attr, old) in sorted(sites["norm_f"].items()):
        NF[layer] = SegLUTRMSNorm(old.weight.data, getattr(old, "eps", 1e-5))
        setattr(parent, attr, NF[layer])
        nh.originals.append((parent, attr, old))
    for layer, (_mixer, _attr, old) in sorted(sites["norm_gated"].items()):
        GN[layer] = SegLUTGatedNorm(old.weight.data, getattr(old, "eps", 1e-5),
                                    group_size=getattr(old, "group_size", None))
    if not (len(NP) and len(NF) and len(GN)):
        raise RuntimeError(f"norm sites not found: pre={len(NP)} f={len(NF)} gated={len(GN)}")

    cur_kw = {}

    for mixer in ph.mixers:
        def bound(u, _m=mixer, inference_params=None, **e):
            li = _m.layer_idx
            return mamba2_reference_forward(_m, u, _m._transition, chunk_size=args.chunk_size,
                                            gated_norm_override=GN[li],
                                            **cur_kw.get(li, {}), **e)
        mixer.forward = bound

    # ------------------------------------------------------------ build tables
    st = json.loads(args.gate_stats.read_text())
    pc = st["per_channel"]
    SC, SN, SP, FD = {}, {}, {}, {}
    fam_idx = {f: i for i, f in enumerate(FAMILIES)}

    def seed_for(fam, layer):
        return args.seed * 1_000_003 + fam_idx[fam] * 1000 + int(layer)

    for layer, mixer in ((m.layer_idx, m) for m in ph.mixers):
        key = str(layer)
        sd = pc["silu_conv_in"][key]
        SC[layer] = build_seglut("silu", sd["min"], sd["max"], n_seg=args.n_seg,
                                 boundaries=bkind("silu"), seed=seed_for("silu_conv", layer),
                                 name=f"silu_conv_L{layer}", **lut_kw)
        sd = pc["silu_norm_in"][key]
        SN[layer] = build_seglut("silu", sd["min"], sd["max"], n_seg=args.n_seg,
                                 boundaries=bkind("silu"), seed=seed_for("silu_norm", layer),
                                 name=f"silu_norm_L{layer}", **lut_kw)
        sd = pc["softplus_in"][key]
        SP[layer] = build_seglut("softplus", sd["min"], sd["max"], n_seg=args.n_seg,
                                 boundaries=bkind("softplus"),
                                 seed=seed_for("softplus", layer),
                                 name=f"softplus_L{layer}", **lut_kw)
        sd = pc["dt_raw"][key]
        A = (-torch.exp(mixer.A_log.float())).detach().cpu().numpy()
        b = mixer.dt_bias.float().detach().cpu().numpy()
        FD[layer] = build_seglut("fused_dt", sd["min"], sd["max"], n_seg=args.n_seg,
                                 boundaries=bkind("fused_dt"), A=A, b=b,
                                 seed=seed_for("fused_dt", layer),
                                 name=f"fused_dt_L{layer}", **lut_kw)

    holder = nn.ModuleDict({
        **{f"norm_pre_{k}": v for k, v in NP.items()},
        **{f"norm_f_{k}": v for k, v in NF.items()},
        **{f"norm_gated_{k}": v for k, v in GN.items()},
        **{f"silu_conv_{k}": v for k, v in SC.items()},
        **{f"silu_norm_{k}": v for k, v in SN.items()},
        **{f"softplus_{k}": v for k, v in SP.items()},
        **{f"fused_dt_{k}": v for k, v in FD.items()},
    })
    model.seglut_modules = holder.to(args.device)

    def tables(fam):
        return {"norm_pre": lambda: [m.lut for m in NP.values()],
                "norm_f": lambda: [m.lut for m in NF.values()],
                "norm_gated": lambda: [m.lut for m in GN.values()],
                "fused_dt": lambda: list(FD.values()),
                "softplus": lambda: list(SP.values()),
                "silu_conv": lambda: list(SC.values()),
                "silu_norm": lambda: list(SN.values())}[fam]()

    def configure(gate):
        fams = set(GATE_FAMILIES[gate])
        for m in NP.values():
            m.use_lut = "norm_pre" in fams
        for m in NF.values():
            m.use_lut = "norm_f" in fams
        for li, m in GN.items():
            m.use_lut = "norm_gated" in fams
            m.silu = SN[li] if "silu_norm" in fams else None
        cur_kw.clear()
        for li in GN:
            kw = {}
            if "silu_conv" in fams:
                kw["silu_conv_gate"] = SC[li]
            if "softplus" in fams:
                kw["softplus_gate"] = SP[li]
            if "fused_dt" in fams:
                kw["fused_dt_gate"] = FD[li]
            cur_kw[li] = kw
        for fam in FAMILIES:
            if fam.startswith("norm_") and getattr(next(iter(NP.values())), "lut", None) is None:
                continue
            for lut in tables(fam):
                lut.reseed()
                lut.reset_counters()
                lut.track = fam in fams

    # ------------------------------------------- calibrate the norms, then check
    configure("exact")
    for m in list(NP.values()) + list(NF.values()) + list(GN.values()):
        m.calibrating = True
    with torch.no_grad():
        for i in range(calib.shape[0]):
            model(calib[i:i + 1].to(args.device))
        exact_logits = model(probe).logits.float()
    n_inv = args.n_seg_invsqrt or args.n_seg
    norm_ranges = {}
    for site, mods in (("norm_pre", NP), ("norm_f", NF), ("norm_gated", GN)):
        norm_ranges[site] = {}
        for layer, m in mods.items():
            m.calibrating = False
            norm_ranges[site][str(layer)] = [float(m.v_min), float(m.v_max)]
            m.build_lut(n_seg=n_inv, boundaries=bkind("invsqrt"),
                        seed=seed_for(site, layer), name=f"{site}_L{layer}",
                        impl=args.invsqrt_impl, ss_kw=ss_kw, **lut_kw)
    ref_diff = float((exact_logits - ref_logits).abs().max())
    print(f"[seglut] exact-mode replacement vs untouched model, max |d logit| on one "
          f"256-token block: {ref_diff:.3e}")
    if ref_diff > 1e-2:
        raise RuntimeError("exact-mode replacements do not reproduce the model; the "
                           "pairing would be meaningless")
    for site in norm_ranges:
        v = [x for lohi in norm_ranges[site].values() for x in lohi]
        print(f"[seglut] calibrated v range {site:10s}: [{min(v):.4g}, {max(v):.4g}] "
              f"over {len(norm_ranges[site])} instance(s)")

    # ------------------------------------------------------- table precision
    precision = {}
    for fam in FAMILIES:
        precision[fam] = summarize_precision(tables(fam))
        p = precision[fam]
        print(f"[seglut] {fam:10s} {p['n_tables']:6d} tables  bits median {p['bits_median']:5.1f} "
              f"worst {p['bits_worst']:5.1f}   max abs err worst {p['max_abs_worst']:.2e}")
        if fam.startswith("norm_") and args.invsqrt_impl == "openfhe":
            rb = [float(t.precision()["rel_bits"][0]) for t in tables(fam)]
            p["rel_bits_median"], p["rel_bits_worst"] = float(np.median(rb)), min(rb)
            print(f"[seglut] {'':10s} relative bits median {p['rel_bits_median']:5.1f} "
                  f"worst {p['rel_bits_worst']:5.1f}   max rel err worst {p['max_rel_worst']:.2e}")
    ops = next(iter(SC.values())).tfhe_ops_per_element
    print(f"[seglut] TFHE ops per element (n_seg={args.n_seg}, oob={args.oob}): {ops}")

    out = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "backend": backend, "calib_info": calib_info,
           "exact_vs_untouched_max_logit_diff": ref_diff,
           "norm_v_ranges": norm_ranges, "table_precision": precision,
           "tfhe_ops_per_element": ops, "results": {}}

    # --------------------------------------------------------------- evaluate
    with torch.no_grad():
        for L in args.lengths:
            shards = make_shards(ids, args.shards, L)
            if args.max_blocks_per_shard:
                shards = [s[: args.max_blocks_per_shard] for s in shards]
            print(f"\n=== L={L}: {len(shards)} shards x {shards[0].shape[0]} blocks ===")
            configure("exact")
            t0 = time.time()
            base = []
            for k, blk in enumerate(shards):
                n, t = shard_nll(model, blk, args.device, cfg.vocab_size)
                base.append(n / t)
                print(f"\r  exact          shard {k+1}/{len(shards)} ({time.time()-t0:.0f}s)",
                      end="", flush=True)
            print()
            res = {}
            for gname in args.gate:
                configure(gname)
                gl = []
                for k, blk in enumerate(shards):
                    n, t = shard_nll(model, blk, args.device, cfg.vocab_size)
                    gl.append(n / t)
                    print(f"\r  {gname:14s} shard {k+1}/{len(shards)} ({time.time()-t0:.0f}s)",
                          end="", flush=True)
                print()
                finite = all(math.isfinite(g) for g in gl)
                dppl = [math.exp(min(g, 700)) - math.exp(b) for g, b in zip(gl, base)]
                sd = statistics.stdev(dppl) if len(dppl) > 1 and finite else float("nan")
                mean = statistics.fmean(dppl) if finite else float("inf")
                sem = sd / math.sqrt(len(dppl)) if len(dppl) > 1 else float("nan")
                diag = {}
                for fam in GATE_FAMILIES[gname]:
                    ts = tables(fam)
                    seen = sum(t.n_seen for t in ts)
                    diag[fam] = {"frac_out_of_domain": frac(sum(t.n_oob for t in ts), seen),
                                 "frac_out_lt_0": frac(sum(t.n_neg for t in ts), seen),
                                 "frac_out_gt_1": frac(sum(t.n_gt1 for t in ts), seen),
                                 # openfhe 1/sqrt only: reads past the table, reflected (garbage)
                                 "frac_wrapped": frac(sum(getattr(t, "n_wrap", 0) for t in ts), seen),
                                 "frac_wrapped_low": frac(sum(getattr(t, "n_wrap_lo", 0) for t in ts), seen),
                                 # inf/nan inputs: an earlier norm already diverged
                                 "frac_nonfinite": frac(sum(getattr(t, "n_nonfinite", 0) for t in ts), seen),
                                 "n_seen": seen}
                res[gname] = {
                    "per_shard_delta_ppl": dppl,
                    "per_shard_delta_nll": [g - b for g, b in zip(gl, base)],
                    "mean_delta_ppl": mean, "std_delta_ppl": sd, "sem_delta": sem,
                    "significant_paired": (abs(mean) > 2 * sem) if finite and len(dppl) > 1 else True,
                    "n_shards": len(dppl),
                    "exact_mean_ppl": statistics.fmean([math.exp(b) for b in base]),
                    "diagnostics": diag,
                }
            out["results"][str(L)] = res
            configure("exact")

    nh.restore()
    ph.restore()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print()
    print("=" * 112)
    print(f"PAIRED, segmented LUT (n_seg={args.n_seg}, n_br={args.n_br}, oob={args.oob}, "
          f"noise={args.noise_bits} {args.noise_scale}, cmp_noise={args.cmp_noise_bits})")
    if args.invsqrt_impl == "openfhe":
        print(f"1/sqrt: OpenFHE pipeline, {args.newton} Newton step(s), FHEW noise "
              f"{args.fhew_noise}, CKKS floor {args.ckks_bits} bits, guard {args.guard}")
    print("=" * 112)
    hdr = (f"{'L':>6} {'gate':>14} {'mean d ppl':>12} {'2*SEM':>9} {'verdict':>22}   "
           f"{'out-of-domain':>13} {'out<0':>9} {'out>1':>9} {'wrapped':>9}")
    print(hdr); print("-" * len(hdr))
    for L, res in out["results"].items():
        for g, r in res.items():
            v = "SIGNIFICANT" if r["significant_paired"] else "not distinguishable"
            d = r["diagnostics"]
            worst = lambda k: max((x[k] for x in d.values() if x[k] == x[k]), default=float("nan"))  # noqa: E731
            print(f"{L:>6} {g:>14} {r['mean_delta_ppl']:+12.4f} {2*r['sem_delta']:9.4f} {v:>22}   "
                  f"{worst('frac_out_of_domain'):13.2e} {worst('frac_out_lt_0'):9.2e} "
                  f"{worst('frac_out_gt_1'):9.2e} {worst('frac_wrapped'):9.2e}")
    print("-" * len(hdr))
    print("out<0 matters for softplus (Delta) and fused_dt (a); out>1 matters for fused_dt only.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
