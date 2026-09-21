#!/usr/bin/env python
"""Part 2 -- fit a low-degree polynomial to exp(z) on a negative interval.

    python fit_exp_polynomial.py --degree 4 --xmin -8 --xmax 0

Prints max abs error, mean abs error and RMSE, and writes the coefficients to
JSON (default: coefficients/exp_poly<deg>_<xmin>_<xmax>[_tag].json).

Fitting methods (none of them is Taylor -- see baby_mamba/polynomial.py):
    chebyshev  near-minimax, never fails                      [default]
    lobatto    interpolates the endpoints too (P(0)=1 if xmax=0)
    remez      true minimax; raises if it cannot converge
    lstsq      weighted least squares; pair with --weight relative

Flags worth knowing:
    --pin-zero        force P(0) == 1 exactly. exp(0)=1 means "remember
                      everything"; getting it wrong costs you long-range memory
                      at a rate of P(0)^L. Not supported with --method remez.
    --weight relative optimise relative rather than absolute error (lstsq only)
    --compare         also print every degree from 2..degree in one table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baby_mamba.polynomial import (
    FIT_METHODS,
    PowerSchedule,
    approximation_report,
    fit_exp_poly,
    save_coefficients,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Fit exp(z) with a low-degree, FHE-friendly polynomial.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--degree", type=int, required=True, help="polynomial degree (2-4 is the interesting range)")
    ap.add_argument("--xmin", type=float, default=-8.0, help="left end of the fit interval (default -8)")
    ap.add_argument("--xmax", type=float, default=0.0, help="right end of the fit interval (default 0)")
    ap.add_argument("--method", choices=FIT_METHODS, default="chebyshev")
    ap.add_argument("--weight", choices=("uniform", "relative"), default="uniform",
                    help="error weighting; only used by --method lstsq")
    ap.add_argument("--pin-zero", action="store_true", help="enforce P(0) == 1 exactly")
    ap.add_argument("--out", type=Path, default=None, help="output JSON path")
    ap.add_argument("--outdir", type=Path, default=Path("coefficients"))
    ap.add_argument("--tag", type=str, default="", help="suffix for the output filename")
    ap.add_argument("--compare", action="store_true", help="also table every degree from 2 up to --degree")
    ap.add_argument("--quiet", action="store_true")
    return ap


def default_path(args) -> Path:
    def fmt(v: float) -> str:
        return f"{v:g}".replace("-", "m").replace(".", "p")
    bits = [f"exp_poly{args.degree}", fmt(args.xmin), fmt(args.xmax), args.method]
    if args.pin_zero:
        bits.append("pin0")
    if args.method == "lstsq" and args.weight != "uniform":
        bits.append(args.weight)
    if args.tag:
        bits.append(args.tag)
    return args.outdir / ("_".join(bits) + ".json")


HEADER = (f"{'deg':>3} {'depth':>5} {'mults':>5} {'max|err|':>11} {'mean|err|':>11} "
          f"{'RMSE':>11} {'P(0)':>10} {'min P':>10} {'frac<0':>7} {'frac>1':>7}")


def row(deg: int, rep: dict) -> str:
    s = PowerSchedule.binary(deg)
    return (f"{deg:>3} {s.ct_ct_depth:>5} {s.ct_ct_mults:>5} "
            f"{rep['max_abs_error']:11.4e} {rep['mean_abs_error']:11.4e} {rep['rmse']:11.4e} "
            f"{rep['P_at_0']:10.6f} {rep['poly_min']:10.6f} "
            f"{rep['frac_poly_lt_0']:7.3f} {rep['frac_poly_gt_1']:7.3f}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.xmin >= args.xmax:
        raise SystemExit(f"--xmin ({args.xmin}) must be < --xmax ({args.xmax})")

    coeffs = fit_exp_poly(args.degree, args.xmin, args.xmax,
                          method=args.method, weight=args.weight, pin_zero=args.pin_zero)
    report = approximation_report(coeffs, args.xmin, args.xmax)
    sched = PowerSchedule.binary(args.degree)

    out = args.out or default_path(args)
    save_coefficients(out, coeffs, args.xmin, args.xmax, args.method, args.weight,
                      pin_zero=args.pin_zero, report=report)

    if not args.quiet:
        print(f"target        exp(z)  on  [{args.xmin:g}, {args.xmax:g}]")
        print(f"method        {args.method}"
              + (f" (weight={args.weight})" if args.method == "lstsq" else "")
              + (", P(0)=1 pinned" if args.pin_zero else ""))
        print()
        print("--- FHE cost (Part 3) ---")
        print(sched.explain())
        print()
        print("--- coefficients, lowest power first ---")
        for k, c in enumerate(coeffs):
            print(f"  c{k} = {c: .10f}")
        print()
        print("--- approximation error on the fit interval ---")
        print(f"  max  abs error : {report['max_abs_error']:.6e}   (at z = {report['argmax_abs_error_z']:.4f})")
        print(f"  mean abs error : {report['mean_abs_error']:.6e}")
        print(f"  RMSE           : {report['rmse']:.6e}")
        print(f"  max  rel error : {report['max_rel_error']:.6e}")
        print()
        print("--- gate diagnostics (why this is not just an activation) ---")
        print(f"  P(0)           : {report['P_at_0']:.8f}   (exp(0) = 1; this is the 'remember everything' case)")
        print(f"  P({args.xmin:g})".ljust(17) + f": {report['P_at_xmin']:.8f}   (exp = {__import__('math').exp(args.xmin):.8f})")
        print(f"  min P / max P  : {report['poly_min']:.6f} / {report['poly_max']:.6f}")
        print(f"  frac P(z) < 0  : {report['frac_poly_lt_0']:.4f}   <-- a negative decay flips the sign of memory every step")
        print(f"  frac P(z) > 1  : {report['frac_poly_gt_1']:.4f}   <-- a decay above 1 makes the recurrence expand")
        if report["P_at_0"] < 0.999:
            drift = report["P_at_0"] ** 1024
            print(f"  NOTE: P(0) = {report['P_at_0']:.4f}, so a perfectly-remembered token decays to "
                  f"{drift:.3e} after 1024 steps. Try --pin-zero.")
        print()
        print(f"wrote {out}")

        if args.compare:
            print()
            print(f"--- every degree from 2 to {args.degree}, same interval and method ---")
            print(HEADER)
            for d in range(2, args.degree + 1):
                try:
                    c = fit_exp_poly(d, args.xmin, args.xmax, method=args.method,
                                     weight=args.weight, pin_zero=args.pin_zero)
                except Exception as e:                      # noqa: BLE001
                    print(f"{d:>3}  failed: {e}")
                    continue
                print(row(d, approximation_report(c, args.xmin, args.xmax)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
