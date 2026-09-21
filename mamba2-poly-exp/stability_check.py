#!/usr/bin/env python
"""Part 8 -- is the polynomial transition actually STABLE, or just lucky?

    python stability_check.py --transition poly4 --interval-mode per-head --pin-zero
    python stability_check.py --sweep

Part 7 reports perplexity. Perplexity can look fine while the state is quietly
doing something insane, and it can be NaN for reasons that are hard to localise.
This script instruments the transition and the SSM state directly, at sequence
lengths 128 / 512 / 1024 / 2048.

WE DO NOT USE torch.clamp ANYWHERE.
Clamping `a` into (0, 1] would make every table below look healthy, and it would
be a lie: `min(max(a, 0), 1)` is a comparison, and comparisons are exactly what
CKKS cannot do cheaply (you need a high-degree polynomial sign approximation,
which costs far more depth than the polynomial we are trying to save). A
clamped result would not be implementable under FHE, so clamping here would
invalidate the whole experiment. Instability is reported, never hidden.
`baby_mamba/tests/test_polynomial.py::test_no_clamp_anywhere` enforces this by
scanning the source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from real_mamba.data import get_blocks
from real_mamba.model import DEFAULT_MODEL, iter_mixers, load_model, recommended_dtype
from real_mamba.patch import patch_transition, set_transition
from real_mamba.reference_ssd import ssd_product_form
from real_mamba.transitions import (
    add_interval_mode_args,
    add_transition_args,
    depth_estimate_any,
    describe_any,
    transition_from_args,
)

LENGTHS = (128, 512, 1024, 2048)


class StabilityProbe:
    """Per-layer transition and state diagnostics, accumulated over a run."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.reset()

    def reset(self):
        z = lambda: [0] * self.n_layers                     # noqa: E731
        self.n = z()
        self.a_min = [math.inf] * self.n_layers
        self.a_max = [-math.inf] * self.n_layers
        self.n_lt0 = z()
        self.n_gt1 = z()
        self.n_nan = z()
        self.n_inf = z()
        self.state_norm_max = [0.0] * self.n_layers
        self.state_norm_final = [0.0] * self.n_layers
        self.z_min = [math.inf] * self.n_layers

    def __call__(self, layer_idx, zv, a):
        i = layer_idx
        af = a.detach().float()
        self.n[i] += af.numel()
        self.n_nan[i] += int(torch.isnan(af).sum())
        self.n_inf[i] += int(torch.isinf(af).sum())
        fin = af[torch.isfinite(af)]
        if fin.numel():
            self.a_min[i] = min(self.a_min[i], float(fin.min()))
            self.a_max[i] = max(self.a_max[i], float(fin.max()))
            self.n_lt0[i] += int((fin < 0).sum())
            self.n_gt1[i] += int((fin > 1).sum())
        self.z_min[i] = min(self.z_min[i], float(zv.detach().float().min()))

    def note_state(self, layer_idx, norms):
        self.state_norm_max[layer_idx] = max(self.state_norm_max[layer_idx],
                                             float(norms.max()))
        self.state_norm_final[layer_idx] = float(norms[..., -1].max())

    def summary(self) -> dict:
        n = max(sum(self.n), 1)
        return {
            "a_min": min(self.a_min), "a_max": max(self.a_max),
            "frac_a_lt_0": sum(self.n_lt0) / n,
            "frac_a_gt_1": sum(self.n_gt1) / n,
            "frac_a_nan": sum(self.n_nan) / n,
            "frac_a_inf": sum(self.n_inf) / n,
            "z_min": min(self.z_min),
            "max_state_norm": max(self.state_norm_max) if any(self.state_norm_max) else float("nan"),
            "worst_layer_a_min": int(min(range(self.n_layers), key=lambda i: self.a_min[i])),
            "per_layer_a_min": self.a_min,
            "per_layer_frac_a_lt_0": [c / max(k, 1) for c, k in zip(self.n_lt0, self.n)],
        }


@torch.no_grad()
def measure_state_norms(model, ids, transition, chunk_size, device):
    """Re-run every mixer's scan and record the SSM state norm trajectory.

    We call `ssd_product_form` a second time with `return_a` so we can read the
    final state; running the scan again is cheaper and much simpler than
    threading state tensors out through the patched forward.
    """
    from real_mamba.patch import mamba2_reference_forward   # noqa: F401
    norms = {}

    def hook_factory(idx):
        def hook(mod, inp, out):
            norms[idx] = out
        return hook

    # Instead of hooking, recompute per layer from the layer inputs we capture.
    captured = {}
    handles = []
    for idx, mixer in iter_mixers(model):
        def pre(mod, args, kwargs=None, _i=idx):
            captured[_i] = args[0].detach()
        handles.append(mixer.register_forward_pre_hook(pre))
    try:
        model(ids)
    finally:
        for h in handles:
            h.remove()

    import torch.nn.functional as F
    from einops import rearrange
    from real_mamba.nn_ref import causal_depthwise_conv1d_ref
    out = {}
    for idx, mixer in iter_mixers(model):
        u = captured[idx]
        zxbcdt = mixer.in_proj(u)
        A = -torch.exp(mixer.A_log.float())
        d_ssm = getattr(mixer, "d_ssm", mixer.d_inner)
        nh, ng, ds = mixer.nheads, mixer.ngroups, mixer.d_state
        d_mlp = (zxbcdt.shape[-1] - 2 * d_ssm - 2 * ng * ds - nh) // 2
        _, _, _, xBC, dt = torch.split(
            zxbcdt, [d_mlp, d_mlp, d_ssm, d_ssm + 2 * ng * ds, nh], dim=-1)
        xBC = causal_depthwise_conv1d_ref(xBC, mixer.conv1d, mixer.d_conv)
        x, Bm, Cm = torch.split(xBC, [d_ssm, ng * ds, ng * ds], dim=-1)
        dt = F.softplus(dt.float() + mixer.dt_bias.float())
        tr = mixer._transition
        _, _, _, final = ssd_product_form(
            rearrange(x.float(), "b l (h p) -> b l h p", p=mixer.headdim), dt, A,
            rearrange(Bm.float(), "b l (g n) -> b l g n", g=ng),
            rearrange(Cm.float(), "b l (g n) -> b l g n", g=ng),
            tr, D=None, chunk_size=chunk_size, return_a=True)
        out[idx] = float(final.reshape(final.shape[0], -1).norm(dim=-1).max())
    return out


SWEEP = [
    ("exact", "global", False),
    ("poly2", "global", False),
    ("poly4", "global", False),
    ("poly2", "per-head", True),
    ("poly3", "per-head", True),
    ("poly4", "per-head", True),
]


def main(argv=None) -> int:
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
    ap.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    ap.add_argument("--blocks", type=int, default=2, help="sequences per length")
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("runs/part8_stability"))
    add_transition_args(ap)
    add_interval_mode_args(ap)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    model, cfg, backend = load_model(args.model, args.backend, args.device,
                                     recommended_dtype(args.device) if args.dtype == 'auto' else getattr(torch, args.dtype))
    n_layers = sum(1 for _ in iter_mixers(model))
    probe = StabilityProbe(n_layers)

    combos = SWEEP if args.sweep else [(args.transition, args.interval_mode, args.pin_zero)]
    rows, handle = [], None
    for spec, mode, pin in combos:
        sub = argparse.Namespace(**vars(args))
        sub.transition, sub.interval_mode, sub.pin_zero = spec, mode, pin
        try:
            transition = transition_from_args(sub)
        except SystemExit as e:
            print(f"[skip] {spec}/{mode}: {e}")
            continue
        desc = describe_any(transition)
        if handle is None:
            handle = patch_transition(model, transition, chunk_size=args.chunk_size,
                                      collector=probe)
        else:
            set_transition(handle, transition)

        for L in args.lengths:
            blocks, _, _ = get_blocks(args.data, args.split, L, verbose=False)
            ids = blocks[: args.blocks].to(args.device)
            probe.reset()
            with torch.no_grad():
                model(ids)
            s = probe.summary()
            state = measure_state_norms(model, ids, transition, args.chunk_size, args.device)
            s["max_state_norm"] = max(state.values())
            s["max_state_norm_layer"] = int(max(state, key=state.get))
            label = f"{spec}" + ("" if mode == "global" else "/per-head") + ("/pin0" if pin else "")
            rows.append({"transition": spec, "label": label, "interval_mode": mode,
                         "pin_zero": pin, "degree": desc.get("degree") or "",
                         "fhe_depth": depth_estimate_any(transition), "seq_len": L,
                         **{k: v for k, v in s.items()
                            if not k.startswith("per_layer")},
                         "_per_layer": {k: v for k, v in s.items() if k.startswith("per_layer")}})
    if handle is not None:
        handle.restore()

    # ---- the table ---------------------------------------------------------
    print()
    print("=" * 128)
    print("PART 8 -- STABILITY.  No torch.clamp is used anywhere; see the module docstring.")
    print("=" * 128)
    hdr = (f"{'transition':>18} {'depth':>6} {'L':>6} {'min a':>16} {'max a':>12} "
           f"{'frac a<0':>10} {'frac a>1':>10} {'frac NaN':>9} {'max||h||':>12} {'verdict':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bad = []
        if r["frac_a_nan"] > 0 or r["frac_a_inf"] > 0 or not math.isfinite(r["max_state_norm"]):
            bad.append("NaN/Inf")
        if r["max_state_norm"] > 1e6:
            bad.append("EXPLODING")
        if r["a_min"] < -1.0:
            bad.append("a<<0")
        if r["frac_a_gt_1"] > 1e-3:
            bad.append("a>1")
        verdict = "FLAG: " + ",".join(bad) if bad else "ok"
        print(f"{r['label']:>18} {r['fhe_depth']:>6} {r['seq_len']:>6} {r['a_min']:16.4g} "
              f"{r['a_max']:12.6f} {r['frac_a_lt_0']:10.5f} {r['frac_a_gt_1']:10.5f} "
              f"{r['frac_a_nan']:9.5f} {r['max_state_norm']:12.4g}   {verdict}")
    print("-" * len(hdr))
    print("Reading this table:")
    print("  exp gives a in (0,1] ALWAYS. Any 'min a' < 0 or 'frac a>1' > 0 is a")
    print("  behaviour the real model cannot produce. Whether it MATTERS is what")
    print("  Part 7's perplexity column answers -- the two tables must be read together.")
    print("  'max||h||' growing with L is the compounding failure Part 4 predicted.")

    cols = [k for k in rows[0] if not k.startswith("_")]
    with (args.outdir / "stability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in cols} for r in rows])
    (args.outdir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.outdir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"backend": backend}, indent=2) + "\n")
    print(f"\nwrote {args.outdir}/stability.csv (+ metrics.json, config.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
