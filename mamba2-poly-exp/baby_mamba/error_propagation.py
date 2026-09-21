#!/usr/bin/env python
"""Part 4 -- a tiny error in `a` is not a tiny error in the model.

    python -m baby_mamba.error_propagation
    python -m baby_mamba.error_propagation --delta-scale 4 --lengths 16 64 256 1024

THE POINT
---------
If you replace a SiLU with a polynomial, an error of 0.03 stays an error of
about 0.03. If you replace `a = exp(A*delta)` with a polynomial, the error is
*multiplied into the state once per timestep*. Two things then happen:

  1. a systematic bias in `a` compounds geometrically. P(0) = 0.966 instead of
     1.0 means a token that should survive forever instead decays like
     0.966^t -- gone after a few hundred steps.
  2. if P(z) < 0 for some z, the state's sign flips at that step, and if
     |P(z)| > 1 it grows. Neither can happen with the real exp, because
     exp(z) in (0, 1] for z <= 0.

So the quantity to watch is not max|P - exp|. It is the state error as a
function of sequence length, plus the fraction of `a` values outside (0, 1].

This script measures exactly that and writes runs/part4_error_propagation/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from baby_mamba.polynomial import approximation_report, build_poly
from baby_mamba.transition import BabyConfig, BabyMamba2Transition, ExactExp

DEFAULT_LENGTHS = (16, 64, 256, 1024)


def measure(cfg: BabyConfig, transition, u: torch.Tensor, seed: int) -> dict:
    """Run one (length, transition) combination and pull out the diagnostics."""
    model = BabyMamba2Transition(cfg, transition=transition, seed=seed)
    with torch.no_grad():
        out = model(u)
    h, a = out["h"], out["a"]
    # per-timestep state norms, flattened over (nheads, headdim, d_state)
    norms = h.reshape(h.shape[0], h.shape[1], -1).norm(dim=-1)      # (B, L)
    return {
        "z": out["z"],
        "a": a,
        "h": h,
        "state_norms": norms,
        "max_state_norm": float(norms.max()),
        "final_state_norm_mean": float(norms[:, -1].mean()),
        "has_nan": bool(torch.isnan(h).any() or torch.isnan(a).any()),
        "has_inf": bool(torch.isinf(h).any() or torch.isinf(a).any()),
        "a_min": float(a.min()),
        "a_max": float(a.max()),
        "frac_a_lt_0": float((a < 0).float().mean()),
        "frac_a_gt_1": float((a > 1).float().mean()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xmin", type=float, default=-8.0)
    ap.add_argument("--xmax", type=float, default=0.0)
    ap.add_argument("--method", default="chebyshev")
    ap.add_argument("--pin-zero", action="store_true",
                    help="use polynomials with P(0) == 1 enforced")
    ap.add_argument("--delta-scale", type=float, default=1.0,
                    help="pedagogical knob: >1 pushes z more negative (see BabyConfig)")
    ap.add_argument("--outdir", type=Path, default=Path("runs/part4_error_propagation"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    polys = {}
    for d in args.degrees:
        p = build_poly(d, args.xmin, args.xmax, method=args.method, pin_zero=args.pin_zero)
        polys[f"P{d}"] = p

    print(f"interval       [{args.xmin:g}, {args.xmax:g}]   method={args.method}"
          f"{'  pin_zero' if args.pin_zero else ''}")
    print(f"delta_scale    {args.delta_scale}"
          + ("   (1.0 = the model as initialised)" if args.delta_scale == 1.0 else
             "   (>1: emulating a model with larger delta)"))
    print()
    print("--- pointwise approximation quality, before any recurrence ---")
    print(f"{'poly':>6} {'depth':>5} {'max|err|':>11} {'RMSE':>11} {'P(0)':>10} {'frac<0 on interval':>20}")
    for name, p in polys.items():
        r = approximation_report(p.coeff_list(), args.xmin, args.xmax)
        print(f"{name:>6} {p.ct_ct_depth:>5} {r['max_abs_error']:11.4e} {r['rmse']:11.4e} "
              f"{r['P_at_0']:10.6f} {r['frac_poly_lt_0']:20.4f}")

    rows = []
    curves = {}
    for L in args.lengths:
        cfg = BabyConfig(batch=args.batch, seqlen=L, delta_scale=args.delta_scale)
        u = torch.randn(args.batch, L, cfg.d_model,
                        generator=torch.Generator().manual_seed(args.seed + 1))

        exact = measure(cfg, ExactExp(), u, args.seed)
        curves[(L, "exact")] = exact["state_norms"][0].tolist()
        zt = exact["z"]
        rows.append({
            "seqlen": L, "transition": "exact", "degree": "", "ct_ct_depth": "",
            "z_min": float(zt.min()), "z_max": float(zt.max()),
            "rel_state_error_final": 0.0, "rel_state_error_overall": 0.0,
            "max_rel_state_error_over_t": 0.0,
            "mean_abs_transition_error": 0.0, "max_abs_transition_error": 0.0,
            "max_state_norm": exact["max_state_norm"],
            "final_state_norm_mean": exact["final_state_norm_mean"],
            "a_min": exact["a_min"], "a_max": exact["a_max"],
            "frac_a_lt_0": exact["frac_a_lt_0"], "frac_a_gt_1": exact["frac_a_gt_1"],
            "has_nan": exact["has_nan"], "has_inf": exact["has_inf"],
        })

        for name, p in polys.items():
            m = measure(cfg, p, u, args.seed)
            curves[(L, name)] = m["state_norms"][0].tolist()
            dh = m["h"] - exact["h"]
            # relative error of the whole state tensor, and per timestep
            rel_overall = float(dh.norm() / exact["h"].norm())
            # ALLOW-CLAMP: this guards the DENOMINATOR of a reported ratio, in a
            # measurement script. It never touches `a` or the state, so it cannot
            # hide instability -- it only stops a 0/0 in the diagnostic itself.
            per_t = (dh.reshape(*dh.shape[:2], -1).norm(dim=-1) /
                     exact["state_norms"].clamp_min(1e-20))
            da = (m["a"] - exact["a"]).abs()
            rows.append({
                "seqlen": L, "transition": name, "degree": p.degree,
                "ct_ct_depth": p.ct_ct_depth,
                "z_min": float(zt.min()), "z_max": float(zt.max()),
                "rel_state_error_final": float(per_t[:, -1].mean()),
                "rel_state_error_overall": rel_overall,
                "max_rel_state_error_over_t": float(per_t.max()),
                "mean_abs_transition_error": float(da.mean()),
                "max_abs_transition_error": float(da.max()),
                "max_state_norm": m["max_state_norm"],
                "final_state_norm_mean": m["final_state_norm_mean"],
                "a_min": m["a_min"], "a_max": m["a_max"],
                "frac_a_lt_0": m["frac_a_lt_0"], "frac_a_gt_1": m["frac_a_gt_1"],
                "has_nan": m["has_nan"], "has_inf": m["has_inf"],
            })

    print()
    print("--- inside the recurrence ---")
    hdr = (f"{'L':>5} {'trans':>6} {'z range':>18} {'mean|da|':>10} {'rel h err':>10} "
           f"{'max rel h err':>14} {'max||h||':>10} {'a_min':>9} {'frac a<0':>9} {'frac a>1':>9}")
    print(hdr)
    for r in rows:
        flag = ""
        if r["has_nan"] or r["has_inf"]:
            flag = "  <-- NaN/Inf !!"
        elif r["max_state_norm"] > 1e4:
            flag = "  <-- EXPLODING"
        print(f"{r['seqlen']:>5} {r['transition']:>6} "
              f"[{r['z_min']:7.3f},{r['z_max']:7.3f}] "
              f"{r['mean_abs_transition_error']:10.3e} {r['rel_state_error_final']:10.3e} "
              f"{r['max_rel_state_error_over_t']:14.3e} {r['max_state_norm']:10.3e} "
              f"{r['a_min']:9.4f} {r['frac_a_lt_0']:9.4f} {r['frac_a_gt_1']:9.4f}{flag}")

    # ---- what to take away --------------------------------------------------
    print()
    print("--- read this ---")
    for name, p in polys.items():
        errs = [r["rel_state_error_final"] for r in rows if r["transition"] == name]
        Ls = [r["seqlen"] for r in rows if r["transition"] == name]
        growth = (errs[-1] / errs[0]) if errs and errs[0] > 0 else float("nan")
        r0 = approximation_report(p.coeff_list(), args.xmin, args.xmax)
        print(f"{name}: pointwise max|P-exp| = {r0['max_abs_error']:.3e}, but the relative")
        print(f"      state error grew from {errs[0]:.3e} at L={Ls[0]} to {errs[-1]:.3e} at "
              f"L={Ls[-1]}  ({growth:.1f}x)")
    print()
    print("The polynomial does not merely approximate one activation.")
    print("Its output controls how memory is repeatedly multiplied through time.")
    print("Therefore long-sequence stability matters more than pointwise")
    print("approximation error alone.")

    # ---- save ---------------------------------------------------------------
    csv_path = args.outdir / "error_propagation.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (args.outdir / "config.json").write_text(json.dumps(vars(args) | {"outdir": str(args.outdir)},
                                                        indent=2, default=str) + "\n")
    (args.outdir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.outdir / "polynomial_coefficients.json").write_text(
        json.dumps({n: p.to_dict() for n, p in polys.items()}, indent=2) + "\n")
    print(f"\nwrote {csv_path} (+ config.json, metrics.json, polynomial_coefficients.json)")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plots "
                  "(pip install matplotlib, or pass --no-plot)")
            return 0

        names = ["exact"] + list(polys)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

        # (a) the approximation itself
        zz = torch.linspace(args.xmin, args.xmax, 600)
        axes[0].plot(zz, torch.exp(zz), "k-", lw=2, label="exp(z)")
        for name, p in polys.items():
            axes[0].plot(zz, p(zz).detach(), "--", label=f"{name} (depth {p.ct_ct_depth})")
        axes[0].axhline(0, color="r", lw=0.8, ls=":")
        axes[0].set_xlabel("z = A*delta"); axes[0].set_ylabel("a")
        axes[0].set_title("(a) the gate, pointwise")
        axes[0].legend(fontsize=8)

        # (b) error vs length
        for name in names[1:]:
            xs = [r["seqlen"] for r in rows if r["transition"] == name]
            ys = [r["rel_state_error_final"] for r in rows if r["transition"] == name]
            axes[1].plot(xs, ys, "o-", label=name)
        axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
        axes[1].set_xlabel("sequence length"); axes[1].set_ylabel("relative error of final state")
        axes[1].set_title("(b) error compounds with length")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        # (c) state norm trajectory at the longest length
        Lmax = max(args.lengths)
        for name in names:
            y = curves[(Lmax, name)]
            axes[2].plot(range(len(y)), y, label=name, lw=1.2)
        axes[2].set_yscale("log")
        axes[2].set_xlabel("timestep"); axes[2].set_ylabel("||h_t||")
        axes[2].set_title(f"(c) state norm over time, L={Lmax}")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

        fig.tight_layout()
        png = args.outdir / "error_propagation.png"
        fig.savefig(png, dpi=130)
        print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
