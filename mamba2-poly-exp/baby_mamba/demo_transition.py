#!/usr/bin/env python
"""Part 1 -- ONE command that shows you the entire Mamba-2 state transition.

    python -m baby_mamba.demo_transition

Nothing is hidden. Every number printed below was computed by
baby_mamba/transition.py, which you can read top to bottom in five minutes.

Add --poly to see what happens when exp is replaced by a polynomial:

    python -m baby_mamba.demo_transition --poly 4
"""

from __future__ import annotations

import argparse

import torch

from baby_mamba.polynomial import build_poly
from baby_mamba.transition import BabyConfig, BabyMamba2Transition, ExactExp

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

RULE = "=" * 78


def section(title: str) -> None:
    print()
    print(RULE)
    print(title)
    print(RULE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqlen", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poly", type=int, default=None, metavar="DEGREE",
                    help="also run with a degree-DEGREE polynomial instead of exp")
    ap.add_argument("--xmin", type=float, default=-8.0)
    ap.add_argument("--xmax", type=float, default=0.0)
    ap.add_argument("--pin-zero", action="store_true")
    args = ap.parse_args(argv)

    cfg = BabyConfig(batch=args.batch, seqlen=args.seqlen)
    torch.manual_seed(args.seed)
    model = BabyMamba2Transition(cfg, transition=ExactExp(), seed=args.seed)
    u = torch.randn(cfg.batch, cfg.seqlen, cfg.d_model,
                    generator=torch.Generator().manual_seed(args.seed + 1))

    # no_grad: this is a demo, and it keeps every printed number a plain float
    with torch.no_grad():
        out = model(u)

    section("0. THE DIMENSIONS (tiny on purpose; upstream values in brackets)")
    c = cfg.to_dict()
    print(f"  batch    = {c['batch']}")
    print(f"  seqlen   = {c['seqlen']}")
    print(f"  d_model  = {c['d_model']}      [130M checkpoint: 768]")
    print(f"  d_inner  = {c['d_inner']}      [1536]   = expand * d_model")
    print(f"  headdim  = {c['headdim']}      [64]     'P' in the paper")
    print(f"  nheads   = {c['nheads']}      [24]     = d_inner / headdim")
    print(f"  d_state  = {c['d_state']}      [128]    'N' in the paper")

    section("1. u  -- the input to the layer            (batch, seqlen, d_model)")
    print(f"  shape {tuple(u.shape)}")
    print(u)

    section("2. A  -- learned decay rate, ONE SCALAR PER HEAD        (nheads,)")
    print("  A = -exp(A_log), so A < 0 always.  More negative = forgets faster.")
    print("  NOTE the exp here is applied to a WEIGHT, not to data. Under FHE")
    print("  weights are plaintext, so this exp is free. It is not our target.")
    print(f"  shape {tuple(out['A'].shape)}")
    print(out["A"])

    section("3. delta -- input-dependent timestep, always > 0   (b, l, nheads)")
    print("  delta = softplus(projection(u) + dt_bias).  We are NOT replacing")
    print("  softplus in this project. Large delta = 'this token matters, take a")
    print("  big step'; small delta = 'barely move'.")
    print(f"  shape {tuple(out['delta'].shape)}")
    print(out["delta"])

    section("4. z = A * delta      negative x positive  =>  z <= 0   (b, l, nheads)")
    print("  THIS is the input to the nonlinearity we want to make FHE-cheap.")
    print(f"  shape {tuple(out['z'].shape)}")
    print(out["z"])
    z = out["z"]
    print(f"\n  z range in this sample: [{z.min():.4f}, {z.max():.4f}]")
    print(f"  (with freshly-initialised weights z sits near 0, because dt_bias is")
    print(f"   initialised small -- see mamba2.py:118-127. On the REAL pretrained")
    print(f"   model the range is an empirical question: that is Part 6.)")

    section("5. a = exp(z)   the memory-survival factor, in (0, 1]   (b, l, nheads)")
    print("  a = 1  ->  keep the whole previous state")
    print("  a = 0  ->  throw the previous state away")
    print(f"  shape {tuple(out['a'].shape)}")
    print(out["a"])

    section("6. the recurrence:  h_t = a_t * h_{t-1} + b_t")
    print("  a_t is a scalar per (batch, head); it multiplies ALL")
    print(f"  headdim*d_state = {cfg.headdim * cfg.d_state} numbers of that head's state at once.")
    print(f"  h shape {tuple(out['h'].shape)}   (batch, seqlen, nheads, headdim, d_state)")
    print()
    print("  Here is batch 0, head 0, flattened to (seqlen, headdim*d_state):")
    h00 = out["h"][0, :, 0].reshape(cfg.seqlen, -1)
    for t in range(cfg.seqlen):
        print(f"    t={t}  a={out['a'][0, t, 0]:.6f}   h_t = {h00[t].tolist()}")
    print()
    print("  state 2-norm per timestep (batch 0, head 0):")
    print("   ", [round(float(v), 5) for v in h00.norm(dim=-1)])

    section("7. y -- the layer output read out through C, plus the D skip")
    print(f"  shape {tuple(out['y'].shape)}   (batch, seqlen, nheads, headdim)")
    print(out["y"][0, :, 0])

    if args.poly is not None:
        poly = build_poly(args.poly, args.xmin, args.xmax, pin_zero=args.pin_zero)
        model_p = BabyMamba2Transition(cfg, transition=poly, seed=args.seed)
        with torch.no_grad():
            out_p = model_p(u)

        section(f"8. SAME INPUT, exp REPLACED BY A DEGREE-{args.poly} POLYNOMIAL")
        print(f"  {poly}")
        print(f"  sequential ciphertext-ciphertext multiplication depth: {poly.ct_ct_depth}")
        print()
        print("  z (unchanged):")
        print(out["z"][0, :, 0].tolist())
        print("  exact exp(z):")
        print([round(float(v), 6) for v in out["a"][0, :, 0]])
        print(f"  P{args.poly}(z):")
        print([round(float(v), 6) for v in out_p["a"][0, :, 0]])
        print("  a error:")
        print([round(float(v), 6) for v in (out_p["a"] - out["a"])[0, :, 0]])
        print()
        print("  resulting state norms (batch 0, head 0):")
        ne = out["h"][0, :, 0].reshape(cfg.seqlen, -1).norm(dim=-1)
        npoly = out_p["h"][0, :, 0].reshape(cfg.seqlen, -1).norm(dim=-1)
        print("    exact :", [round(float(v), 5) for v in ne])
        print("    poly  :", [round(float(v), 5) for v in npoly])
        rel = (out_p["h"] - out["h"]).norm() / out["h"].norm()
        print(f"\n  relative state error over the whole tensor: {rel:.6e}")
        print()
        print("  Look at how a small error in `a` becomes a larger error in `h`.")
        print("  Then run   python -m baby_mamba.error_propagation   to watch that")
        print("  gap grow with sequence length. That is Part 4, and it is the")
        print("  whole reason this project is not just 'fit a polynomial'.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
