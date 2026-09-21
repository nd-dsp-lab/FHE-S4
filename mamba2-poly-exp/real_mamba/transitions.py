"""The `--transition` switch: exact | poly2 | poly3 | poly4.

One place that turns a command-line string into a `z -> a` module, so that every
script in this project accepts the same flag and records the same metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn

from baby_mamba.polynomial import PolyExp, approximation_report, build_poly, load_poly
from baby_mamba.transition import ExactExp

TRANSITION_CHOICES = ("exact", "poly2", "poly3", "poly4")

# We allow polyN for any N so that a student can ask "what would degree 6 cost?"
# without editing code. Part 13's acceptance criteria only accept degree <= 4.
_POLY_RE = re.compile(r"^poly(\d+)$")


def add_transition_args(ap):
    """Attach the standard transition flags to an argparse parser."""
    g = ap.add_argument_group("transition (the thing we are replacing)")
    g.add_argument("--transition", default="exact",
                   help="exact | poly2 | poly3 | poly4 (polyN also works for N>4)")
    g.add_argument("--xmin", type=float, default=-8.0,
                   help="left end of the polynomial fit interval (default -8; "
                        "run collect_transition_stats.py before trusting this)")
    g.add_argument("--xmax", type=float, default=0.0)
    g.add_argument("--fit-method", default="chebyshev",
                   choices=("chebyshev", "lobatto", "remez", "lstsq"))
    g.add_argument("--fit-weight", default="uniform", choices=("uniform", "relative"))
    g.add_argument("--pin-zero", action="store_true",
                   help="enforce P(0)=1 exactly; strongly recommended (see Part 4)")
    g.add_argument("--coeff-json", type=Path, default=None,
                   help="load coefficients from a fit_exp_polynomial.py JSON instead of fitting")
    g.add_argument("--trainable-poly", action="store_true",
                   help="make the polynomial coefficients trainable (Part 9 MODE B)")
    return ap


def make_transition(spec: str = "exact", xmin: float = -8.0, xmax: float = 0.0,
                    method: str = "chebyshev", weight: str = "uniform",
                    pin_zero: bool = False, trainable: bool = False,
                    coeff_json: str | Path | None = None) -> nn.Module:
    """Build the transition module named by `spec`."""
    if spec == "exact":
        if coeff_json or trainable:
            raise ValueError("--coeff-json / --trainable-poly make no sense with --transition exact")
        return ExactExp()

    m = _POLY_RE.match(spec)
    if not m:
        raise ValueError(f"unknown --transition {spec!r}; expected one of {TRANSITION_CHOICES}")
    degree = int(m.group(1))
    if degree < 1:
        raise ValueError("polynomial degree must be >= 1")

    if coeff_json is not None:
        poly = load_poly(coeff_json, trainable=trainable)
        if poly.degree != degree:
            raise ValueError(f"--transition poly{degree} but {coeff_json} holds degree {poly.degree}")
        return poly
    return build_poly(degree, xmin, xmax, method=method, weight=weight,
                      pin_zero=pin_zero, trainable=trainable)


def from_args(args) -> nn.Module:
    """Build the transition from a parser populated by add_transition_args."""
    return make_transition(
        args.transition, args.xmin, args.xmax,
        method=args.fit_method, weight=args.fit_weight,
        pin_zero=args.pin_zero, trainable=args.trainable_poly,
        coeff_json=args.coeff_json,
    )


def describe(transition: nn.Module) -> dict:
    """Metadata for config.json / results_summary.csv."""
    if isinstance(transition, PolyExp):
        d = transition.to_dict()
        if transition.interval:
            d["error_report"] = approximation_report(transition.coeff_list(), *transition.interval)
        return d
    return {"name": "exact", "degree": None, "ct_ct_depth": None, "coeffs_lowest_first": None,
            "interval": None, "trainable": False}


def depth_estimate(transition: nn.Module) -> str:
    """The 'FHE depth estimate' column of the Part 7 table."""
    if isinstance(transition, PolyExp):
        return str(transition.ct_ct_depth)
    return "n/a (transcendental)"


# =============================================================================
# Per-head intervals, driven by the Part 6 measurements
# =============================================================================
# Part 6 found that |A| spans 4e-4 .. 3.6e4 across the 576 heads of the
# pretrained checkpoint, so `z = A*delta` per head ranges from [-0.004, 0] to
# [-1.8e5, -0.47]. One global interval cannot serve both. Since `A` is a WEIGHT
# it is plaintext under FHE, so per-head coefficients are free -- same degree,
# same ct-ct depth, just different plaintext constants. This is the single
# biggest lever we have, and it is a direct consequence of measuring first.
#
# Coverage of the pretrained model (from runs/part6_transition_stats):
#   322 / 576 heads:  z_min >= -1     degree-4 max err 1.1e-05
#   171 / 576 heads:  z_min in [-4, -1)
#    30 / 576 heads:  z_min in [-8, -4)
#    53 / 576 heads:  z_min <  -8     <- the hard ones; 38 of them z_min < -32

import json as _json

from baby_mamba.polynomial import PerHeadPolyExp, fit_per_head


def head_intervals_from_stats(stats: dict, layer: int, margin: float = 0.25,
                              percentile: str | None = None,
                              near_zero: float = 0.05) -> list[tuple[float, float]]:
    """Turn Part 6's per-head z statistics into one fit interval per head.

    margin      widen the lower end by this fraction, so tokens slightly beyond
                anything we measured still land inside the interval.
    percentile  None -> use the observed min/max. Otherwise a key such as "p0.1"
                -> use that trimmed quantile instead, which gives a much tighter
                interval at the cost of leaving a measured tail outside it. State
                which you used; it changes the result.
    near_zero   if a head's observed z_max is closer to 0 than this, extend the
                interval all the way to 0 (so P(0) can be pinned). Heads that
                never approach z=0 keep their own upper end, because pinning
                P(0)=1 on a head that never evaluates z=0 wastes a coefficient.
    """
    h = stats["per_head_z"][str(layer)]
    lo_key = percentile or "min"
    hi_key = {"p0.1": "p99.9", "p1": "p99", "p0.01": "p99"}.get(percentile, "max") \
        if percentile else "max"
    los, his = h[lo_key], h[hi_key]
    out = []
    for k in range(len(los)):
        lo = float(los[k]) * (1.0 + margin)
        hi = float(his[k])
        hi = 0.0 if hi > -near_zero else hi * (1.0 - margin)
        if lo >= hi:                      # a head with a degenerate range
            lo = hi - max(abs(hi) * 0.1, 1e-6)
        out.append((lo, min(hi, 0.0)))
    return out


def build_per_head_transitions(stats_path, degree: int, method: str = "chebyshev",
                               weight: str = "uniform", pin_zero: bool = False,
                               margin: float = 0.25, percentile: str | None = None,
                               zero_degenerate: bool = True,
                               trainable: bool = False) -> dict:
    """{layer_idx: PerHeadPolyExp} built from a Part 6 transition_stats.json."""
    stats = _json.loads(open(stats_path).read())
    n_layers = int(stats["n_layers"])
    out = {}
    for layer in range(n_layers):
        ivs = head_intervals_from_stats(stats, layer, margin=margin, percentile=percentile)
        out[layer] = fit_per_head(ivs, degree, method=method, weight=weight,
                                  pin_zero=pin_zero, zero_degenerate=zero_degenerate,
                                  trainable=trainable,
                                  name=f"perhead_poly{degree}_L{layer}")
    return out


def add_interval_mode_args(ap):
    g = ap.add_argument_group("fit interval (Part 6 tells you what to put here)")
    g.add_argument("--interval-mode", default="global", choices=("global", "per-head"),
                   help="global: one [xmin,xmax] for all 576 heads (the naive baseline). "
                        "per-head: each head gets its own interval from --stats-json. "
                        "Per-head costs NOTHING extra under FHE because A is plaintext.")
    g.add_argument("--stats-json", type=Path,
                   default=Path("runs/part6_transition_stats/transition_stats.json"),
                   help="output of collect_transition_stats.py; required for per-head")
    g.add_argument("--interval-margin", type=float, default=0.25,
                   help="widen each per-head interval by this fraction (default 0.25)")
    g.add_argument("--interval-percentile", default=None,
                   choices=(None, "p0.01", "p0.1", "p1"),
                   help="use a trimmed quantile instead of the observed min")
    g.add_argument("--no-zero-degenerate", action="store_true",
                   help="do NOT give constant-zero polynomials to heads whose exp(z) "
                        "underflows over their whole range")
    return ap


def transition_from_args(args):
    """The one entry point every script uses. Returns a module or {layer: module}."""
    if getattr(args, "interval_mode", "global") == "global" or args.transition == "exact":
        return from_args(args)
    m = _POLY_RE.match(args.transition)
    if not m:
        raise ValueError(f"--interval-mode per-head needs --transition polyN, got {args.transition!r}")
    if not Path(args.stats_json).exists():
        raise SystemExit(
            f"--interval-mode per-head needs {args.stats_json}, which does not exist.\n"
            f"Run:  python collect_transition_stats.py --blocks 32"
        )
    return build_per_head_transitions(
        args.stats_json, int(m.group(1)), method=args.fit_method, weight=args.fit_weight,
        pin_zero=args.pin_zero, margin=args.interval_margin,
        percentile=args.interval_percentile,
        zero_degenerate=not args.no_zero_degenerate,
        trainable=args.trainable_poly,
    )


def describe_any(transition) -> dict:
    """describe() that also handles a {layer: module} dict."""
    if isinstance(transition, dict):
        first = next(iter(transition.values()))
        d = {"name": f"{first.name.rsplit('_L', 1)[0]}",
             "kind": "per_head_per_layer",
             "degree": first.degree,
             "ct_ct_depth": first.ct_ct_depth,
             "ct_ct_mults": first.ct_ct_mults,
             "n_layers": len(transition),
             "n_degenerate_heads_total": sum(t.n_degenerate for t in transition.values()),
             "trainable": any(isinstance(t.coeffs, torch.nn.Parameter)
                              for t in transition.values()),
             "per_layer": {str(k): v.to_dict() for k, v in transition.items()}}
        return d
    return describe(transition)


def depth_estimate_any(transition) -> str:
    if isinstance(transition, dict):
        return str(max(t.ct_ct_depth for t in transition.values()))
    return depth_estimate(transition)
