#!/usr/bin/env python
"""The Phase B verdict: a polynomial run vs its control, paired per shard.

    python -m allpoly.compare_phase_b runs/phase_b/<poly run> runs/phase_b/<control run>

Both runs are fine-tuned with the same recipe on the same text and evaluated on the
same shards, so the gain from fine-tuning itself cancels. What is left is the cost
of the polynomial. Every row is also shown against the untouched teacher.

Refuses to compare runs whose recipe or evaluation differ: a mismatched control is
the "number that looks too good" the rules warn about.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# recipe settings that must match between a polynomial run and its control
RECIPE = ("token_budget", "train_data", "train_split", "train_seq_len", "micro_batch", "grad_accum",
          "lr_backbone", "lambda_lm", "lambda_kd", "temperature", "warmup_frac", "grad_clip",
          "chunk_size", "seed", "shards", "lengths", "max_blocks_per_shard", "eval_chunk_size")


# An interval this wide is not a pass. Phase A: "not distinguishable" with a huge
# error bar (SiLU-conv d13, softplus Q27) meant the run was unstable.
WIDE = 0.5   # ppl; 2 SEM above this is reported as UNSTABLE instead of "not distinguishable"


def paired(a, b):
    d = np.exp(np.asarray(a)) - np.exp(np.asarray(b))
    if not np.all(np.isfinite(d)):
        return float("inf"), float("nan"), "BROKEN (inf/nan)"
    m = float(d.mean())
    s = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    if abs(m) > 2 * s:
        v = "SIGNIFICANT"
    elif 2 * s > WIDE:
        v = f"UNSTABLE (2 SEM > {WIDE}: not a pass)"
    else:
        v = "not distinguishable"
    return m, 2 * s, v


def fmt(t):
    m, s, v = t
    return f"{m:+9.4f} +/- {s:7.4f}  {v}"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    runs = [Path(p) for p in argv]
    (P, C), (pc, cc) = ([json.loads((r / f).read_text()) for r in runs] for f in ("metrics.json", "config.json"))
    if P.get("control") or not C.get("control"):
        raise SystemExit("usage: compare_phase_b.py <polynomial run> <control run>")

    bad = [k for k in RECIPE if pc.get(k) != cc.get(k)]
    if bad:
        raise SystemExit("the two runs used different recipes, so the control does not cancel anything:\n  "
                         + "\n  ".join(f"{k}: poly={pc.get(k)!r}  control={cc.get(k)!r}" for k in bad))
    p_init, c_init = pc.get("init_from"), cc.get("init_from")
    if (p_init is None) != (c_init is None):
        raise SystemExit(f"one run starts from a previous stage and the other does not "
                         f"(poly: {p_init}, control: {c_init}). Stage k's control starts from stage k-1's control.")

    PE, CE = P["eval"], C["eval"]
    print(f"polynomial run : {runs[0]}   ({P['kind']})")
    print(f"control run    : {runs[1]}")
    if p_init:
        print(f"started from   : poly {p_init}   control {c_init}")
    print()
    rows = {}
    for L in PE["teacher"]:
        tp, tc = PE["teacher"][L]["per_shard_nll"], CE["teacher"][L]["per_shard_nll"]
        if len(tp) != len(tc) or not np.allclose(tp, tc, rtol=0, atol=1e-6):
            raise SystemExit(f"L={L}: the two runs were not evaluated on the same shards")
        ctrl = CE["after"][L]["per_shard_nll"]
        print(f"L={L}   ({len(tp)} shards; d ppl +/- 2 SEM)")
        if "before" in PE:
            lab = "poly, before training" + ("  (= Phase A row)" if not p_init else "")
            print(f"  {lab:38s} vs teacher  {fmt(paired(PE['before'][L]['per_shard_nll'], tp))}")
        print(f"  {'control, after':38s} vs teacher  {fmt(paired(ctrl, tp))}")
        print(f"  {'poly, after (refit)':38s} vs teacher  {fmt(paired(PE['after_refit'][L]['per_shard_nll'], tp))}")
        v = paired(PE["after_refit"][L]["per_shard_nll"], ctrl)
        rows[L] = v
        print(f"  {'POLY (refit) vs CONTROL':38s}             {fmt(v)}   <- the verdict")
        print(f"  {'  diag: poly, fit it trained with':38s} vs control  "
              f"{fmt(paired(PE['after_trainfit'][L]['per_shard_nll'], ctrl))}")
        print(f"  {'  diag: trained weights, exact ops':38s} vs control  "
              f"{fmt(paired(PE['after_exact'][L]['per_shard_nll'], ctrl))}")
        for k in PE:
            if k.startswith("after_refit+noise"):
                print(f"  {'  ' + k:38s} vs control  {fmt(paired(PE[k][L]['per_shard_nll'], ctrl))}")
        print()

    # operators, decay, drift, overfitting ----------------------------------
    Ls = list(PE["teacher"])
    print("operators after the refit (worst length shown for frac_out):")
    ops = PE["after_refit"][Ls[0]].get("operators", {})
    for o, r in ops.items():
        fo = max(PE["after_refit"][L]["operators"][o].get("frac_out_max", 0.0) for L in Ls)
        dr = P.get("range_drift", {}).get(o, {})
        print(f"  {o:10s} degree {r.get('degree')}  levels {r.get('depth_openfhe')}  frac_out {fo:.1e}  "
              f"range width new/old {dr.get('width_ratio_median', float('nan')):.3f} "
              f"(old fit left by {dr.get('frac_channels_outside_old_fit', float('nan')):.1%} of channels)")
    print("\ndecay factor a (report for every run):")
    for name, E in (("poly after refit", PE["after_refit"]), ("control after", CE["after"]),
                    ("teacher", PE["teacher"])):
        for L in Ls:
            d = E[L]["decay"]
            print(f"  {name:18s} L={L:>5}  frac(a>1) {d['frac_a_gt_1']:.1e}  max a {d['a_max']:.3g}  "
                  f"frac(a<0) {d['frac_a_lt_0']:.1e}")
    print("\noverfitting monitor (held-out minus train-sample NLL; watch the CHANGE, not the level):")
    for name, R in (("poly", P), ("control", C)):
        m = R.get("overfit_monitor", [])
        if m:
            print(f"  {name:8s} gap {m[0]['gap']:+.4f} at step 0 -> {m[-1]['gap']:+.4f} at step {m[-1]['step']}   "
                  f"(held-out nll {m[0]['heldout_nll']:.4f} -> {m[-1]['heldout_nll']:.4f})")

    # one line for the table ------------------------------------------------
    print("\nrow for the Phase B table:")
    cells = " | ".join(f"{rows[L][0]:+.4f} ± {rows[L][1]:.4f} ({rows[L][2]})" for L in Ls)
    fo = max((PE["after_refit"][L]["operators"][o].get("frac_out_max", 0.0) for L in Ls for o in ops), default=0.0)
    d = PE["after_refit"][Ls[-1]]["decay"]
    # no nested f-strings: CRC's Python may be older than 3.12
    degs = ", ".join("%s %s" % (o, r.get("degree")) for o, r in ops.items())
    lv = ", ".join("%s %s" % (o, r.get("depth_openfhe")) for o, r in ops.items())
    print(f"| {P['kind']} | {degs} | {lv} | {cells} | "
          f"{fo:.1e} | a>1 {d['frac_a_gt_1']:.1e} (max {d['a_max']:.3g}), a<0 {d['frac_a_lt_0']:.1e} |")
    print(f"  columns: run | degrees | levels per operator (OpenFHE table; Q^2 adds 1) | poly vs control at L = {', '.join(Ls)} "
          f"| frac_out | decay at L={Ls[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
