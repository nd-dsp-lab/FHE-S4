"""The Mamba-2 LM forward written out explicitly, so every operator is swappable
and every refresh point is visible.

Why not patch the model's own modules: mamba2-130m ships fused_add_norm=True, and
on that path neither residual RMSNorm is ever called as a module. Two separate
fixes in norm/ (collect_norm_stats.py, install_norms.py) exist only because a
swapped-in module was silently bypassed. Here there is nothing to bypass: the
loop below IS the model, and `test_exact_matches_model` checks that in exact mode
it reproduces model(ids).logits.

    hidden = embed(ids); residual = None
    for each layer:
        residual = hidden (+ residual)              <- "block_in" refresh point
        hidden   = mixer(norm_pre(residual))        <- "scan_in_*", "scan_out_y" inside
    logits = lm_head(norm_f(hidden + residual))
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from baby_mamba.polynomial import fit_per_head
from baby_mamba.transition import ExactExp
from real_mamba.patch import mamba2_reference_forward
from real_mamba.transitions import head_intervals_from_stats

from allpoly.ops import ChebPerChannel, PolyInvSqrt, cheb_depth, fit_cheb_interp

OPS = ("exp", "softplus", "silu_conv", "silu_norm", "norm_pre", "norm_gated", "norm_f")


def _rms_exact(x, w, eps):
    v = x.float().square().mean(-1, keepdim=True) + eps
    return x.float() * torch.rsqrt(v) * w.float(), v


# ---------------------------------------------------------------------------
# Operator sets
# ---------------------------------------------------------------------------
class ExactOps:
    """The real operators, optionally recording every input range (calibration)."""

    def __init__(self, record: bool = False):
        self.record = record
        self.stats = defaultdict(dict)   # name -> layer -> {"min": tensor, "max": tensor}
        self.absmax = {}                 # "site:layer" -> float
        self.track_a = True              # decay-factor safety stats, always on
        self.reset_decay()

    # -- decay factor a: the recurrence must never expand ----------------
    def reset_decay(self):
        self.a_n = 0
        self.a_gt1 = 0
        self.a_lt0 = 0
        self.a_max = float("-inf")
        self.a_min = float("inf")

    def decay_report(self) -> dict:
        n = max(self.a_n, 1)
        return {"a_max": self.a_max, "a_min": self.a_min,
                "frac_a_gt_1": self.a_gt1 / n, "frac_a_lt_0": self.a_lt0 / n,
                "n_decays": self.a_n}

    # -- recording -------------------------------------------------------
    def _rec(self, name, layer, t, reduce_dims):
        if not self.record:
            return
        t = t.detach().float()
        lo = t.amin(dim=reduce_dims).cpu()
        hi = t.amax(dim=reduce_dims).cpu()
        d = self.stats[name].get(layer)
        if d is None:
            self.stats[name][layer] = {"min": lo, "max": hi}
        else:
            d["min"] = torch.minimum(d["min"], lo)
            d["max"] = torch.maximum(d["max"], hi)

    def gate_collector(self, layer, name, t):
        if name in ("silu_conv_in", "softplus_in", "silu_norm_in"):
            self._rec(name, layer, t, tuple(range(t.dim() - 1)))   # per channel

    def collector(self, layer, z, a):
        self._rec("z", layer, z, tuple(range(z.dim() - 1)))        # per head
        if self.track_a and a is not None:
            with torch.no_grad():
                a = a.detach().float()
                self.a_n += a.numel()
                self.a_gt1 += int((a > 1.0).sum())
                self.a_lt0 += int((a < 0.0).sum())
                self.a_max = max(self.a_max, float(a.max()))
                self.a_min = min(self.a_min, float(a.min()))

    def refresh(self, layer, site, x):
        # per-channel max |x| over batch and positions (last dim = channel/head)
        if self.record:
            k = f"{site}:{layer}"
            m = x.detach().float().abs().amax(dim=tuple(range(x.dim() - 1))).cpu()
            self.absmax[k] = m if k not in self.absmax else torch.maximum(self.absmax[k], m)
        return x

    # -- operators -------------------------------------------------------
    def transition(self, layer):
        return ExactExp()

    def softplus(self, layer):
        return None

    def silu_conv(self, layer):
        return None

    def _rec_v(self, site, layer, v):
        # v: (B, L, 1). Overall range, plus position 0 vs the rest: token position
        # is public under FHE, so if the extremes sit at a known position a
        # separate plaintext-selected range for it costs no levels.
        self._rec(site, layer, v.reshape(-1, 1), (0,))
        self._rec(site + "@pos0", layer, v[:, :1].reshape(-1, 1), (0,))
        if v.shape[1] > 1:
            self._rec(site + "@rest", layer, v[:, 1:].reshape(-1, 1), (0,))

    def norm(self, layer, site, x, w, eps):
        out, v = _rms_exact(x, w, eps)
        self._rec_v(site, layer, v)
        return out

    def gated_norm(self, layer, mixer):
        gs = getattr(mixer.norm, "group_size", None)
        d_ssm = getattr(mixer, "d_ssm", mixer.d_inner)
        if gs not in (None, d_ssm):
            raise NotImplementedError(f"gated norm with group_size={gs} != d_ssm={d_ssm}")
        if getattr(mixer, "norm_before_gate", False):
            raise NotImplementedError("norm_before_gate=True: this computes norm(x*silu(z)), not norm(x)*silu(z)")
        w, eps = mixer.norm.weight, mixer.norm.eps

        def f(y, gate):
            x = y.float() * F.silu(gate.float())
            out, v = _rms_exact(x, w, eps)
            self._rec_v("norm_gated", layer, v)
            return out.to(y.dtype)
        return f


class PolyOps(ExactOps):
    """Polynomial replacements, fitted to the ranges an ExactOps calibration measured.

    Each operator in OPS can be switched on independently (`which`), so the cost
    of each is attributable. Exact ones fall through to ExactOps.
    """

    def __init__(self, calib: ExactOps, which, *, degree_silu=27, degree_softplus_q=13,
                 degree_exp=4, degree_invsqrt=27, newton=None, invsqrt_target=2.0 ** -12,
                 margin=0.05, v_margin=2.0, exp_margin=0.25):
        super().__init__(record=False)
        self.which = set(which)
        bad = self.which - set(OPS)
        if bad:
            raise ValueError(f"unknown operators {bad}; choose from {OPS}")
        self.cfg = dict(degree_silu=degree_silu, degree_softplus_q=degree_softplus_q,
                        degree_exp=degree_exp, degree_invsqrt=degree_invsqrt, newton=newton,
                        invsqrt_target=invsqrt_target,
                        margin=margin, v_margin=v_margin, exp_margin=exp_margin)
        st = calib.stats
        self.mods = {}

        def widen(lo, hi):
            lo, hi = lo.double().numpy(), hi.double().numpy()
            w = np.maximum(hi - lo, 1e-6)
            return lo - margin * w, hi + margin * w

        layers = sorted(st["silu_conv_in"])
        if "silu_conv" in self.which:
            for l in layers:
                lo, hi = widen(st["silu_conv_in"][l]["min"], st["silu_conv_in"][l]["max"])
                c = fit_cheb_interp(lambda x: x / (1.0 + np.exp(-x)), lo, hi, degree_silu)
                self.mods[("silu_conv", l)] = ChebPerChannel(c, lo, hi, degree_silu, f"silu_conv:{l}")
        if "silu_norm" in self.which:
            for l in layers:
                lo, hi = widen(st["silu_norm_in"][l]["min"], st["silu_norm_in"][l]["max"])
                c = fit_cheb_interp(lambda x: x / (1.0 + np.exp(-x)), lo, hi, degree_silu)
                self.mods[("silu_norm", l)] = ChebPerChannel(c, lo, hi, degree_silu, f"silu_norm:{l}")
        if "softplus" in self.which:
            # Delta = Q(x)^2 with Q ~ sqrt(softplus): non-negative by construction
            for l in layers:
                lo, hi = widen(st["softplus_in"][l]["min"], st["softplus_in"][l]["max"])
                c = fit_cheb_interp(lambda x: np.sqrt(np.logaddexp(0.0, x)), lo, hi, degree_softplus_q)
                self.mods[("softplus", l)] = ChebPerChannel(c, lo, hi, degree_softplus_q,
                                                            f"softplus:{l}", square=True)
        if "exp" in self.which:
            # the validated per-head gate, intervals built the same way as Part 6
            pz = {"per_head_z": {str(l): {"min": st["z"][l]["min"].tolist(),
                                          "max": st["z"][l]["max"].tolist()} for l in st["z"]}}
            self.exp_iv = {}
            for l in sorted(st["z"]):
                ivs = head_intervals_from_stats(pz, l, margin=exp_margin)
                self.mods[("exp", l)] = fit_per_head(ivs, degree_exp, method="chebyshev",
                                                     pin_zero=True, zero_degenerate=True,
                                                     name=f"exp:{l}")
                self.exp_iv[l] = (torch.tensor([a for a, _ in ivs]), torch.tensor([b for _, b in ivs]))
        self.exp_n = 0
        self.exp_out = 0
        for site in ("norm_pre", "norm_gated", "norm_f"):
            if site not in self.which:
                continue
            for l in sorted(st[site]):
                lo = float(st[site][l]["min"].min()) / v_margin
                hi = float(st[site][l]["max"].max()) * v_margin
                self.mods[(site, l)] = PolyInvSqrt(lo, hi, degree_invsqrt, newton,
                                                   target_rel=invsqrt_target, name=f"{site}:{l}")

    def to(self, device):
        for m in self.mods.values():
            m.to(device)
        if hasattr(self, "exp_iv"):
            self.exp_iv = {l: (a.to(device), b.to(device)) for l, (a, b) in self.exp_iv.items()}
        return self

    def collector(self, layer, z, a):
        super().collector(layer, z, a)
        iv = getattr(self, "exp_iv", {}).get(layer)
        if iv is not None:
            with torch.no_grad():
                lo, hi = iv
                zz = z.detach().float()
                self.exp_n += zz.numel()
                self.exp_out += int(((zz < lo) | (zz > hi)).sum())

    # -- operators -------------------------------------------------------
    def transition(self, layer):
        m = self.mods.get(("exp", layer))
        return m if m is not None else ExactExp()

    def softplus(self, layer):
        return self.mods.get(("softplus", layer))

    def silu_conv(self, layer):
        return self.mods.get(("silu_conv", layer))

    def norm(self, layer, site, x, w, eps):
        m = self.mods.get((site, layer))
        if m is None:
            return _rms_exact(x, w, eps)[0]
        v = x.float().square().mean(-1, keepdim=True) + eps
        return x.float() * m(v) * w.float()

    def gated_norm(self, layer, mixer):
        silu = self.mods.get(("silu_norm", layer))
        inv = self.mods.get(("norm_gated", layer))
        w, eps = mixer.norm.weight, mixer.norm.eps

        def f(y, gate):
            g = silu(gate) if silu is not None else F.silu(gate.float())
            x = y.float() * g.float()
            v = x.square().mean(-1, keepdim=True) + eps
            r = inv(v) if inv is not None else torch.rsqrt(v)
            return (x * r * w.float()).to(y.dtype)
        return f

    # -- reporting -------------------------------------------------------
    def reset_counts(self):
        self.reset_decay()
        self.exp_n = self.exp_out = 0
        for m in self.mods.values():
            if hasattr(m, "reset_counts"):
                m.reset_counts()

    def report(self) -> dict:
        """Per operator: worst off-interval fraction, degree, OpenFHE depth."""
        out = {}
        for op in OPS:
            ms = [m for (o, _), m in self.mods.items() if o == op]
            if not ms:
                continue
            r = {"instances": len(ms)}
            if hasattr(ms[0], "frac_out"):
                fr = [m.frac_out for m in ms]
                r["frac_out_max"] = max(fr)
                r["frac_out_mean"] = float(np.mean(fr))
            if hasattr(ms[0], "depth"):
                r["depth_openfhe"] = ms[0].depth
                r["degree"] = ms[0].degree
            if op == "exp":
                r["frac_out_max"] = self.exp_out / max(self.exp_n, 1)
                r["frac_out_mean"] = r["frac_out_max"]
                r["degree"] = self.cfg["degree_exp"]
                r["depth_openfhe"] = cheb_depth(self.cfg["degree_exp"])
            if hasattr(ms[0], "init_rel_err_max"):
                r["invsqrt_init_rel_err_max"] = max(m.init_rel_err_max for m in ms)
                r["invsqrt_init_all_in_basin"] = all(m.init_in_basin for m in ms)
                r["newton_steps_max"] = max(m.newton for m in ms)
                r["depth_openfhe"] = max(m.depth for m in ms)          # worst layer
                r["range_ratio_max"] = max(m.hi / m.lo for m in ms)
            out[op] = r
        return out


# ---------------------------------------------------------------------------
# The forward
# ---------------------------------------------------------------------------
def allpoly_forward(model, ids, ops: ExactOps, refresh=None, chunk_size=None):
    """Logits of the Mamba-2 LM with `ops` supplying every non-polynomial operator.

    `refresh(layer, site, x) -> x` is applied at "block_in" (the residual stream
    entering each block) and, via the mixer, at the scan's inputs and output.
    Everything runs in fp32 so that polynomial and noise error are not confounded
    with fp16 rounding.
    """
    bb = model.backbone
    hidden = bb.embedding(ids).float()
    residual = None
    rf = refresh if refresh is not None else (lambda l, s, x: x)
    for i, layer in enumerate(bb.layers):
        residual = hidden if residual is None else residual + hidden
        residual = rf(i, "block_in", residual)
        h = ops.norm(i, "norm_pre", residual, layer.norm.weight, layer.norm.eps)
        mixer = layer.mixer
        hidden = mamba2_reference_forward(
            mixer, h.to(mixer.in_proj.weight.dtype), ops.transition(i),
            chunk_size=chunk_size,
            softplus_gate=ops.softplus(i), silu_conv_gate=ops.silu_conv(i),
            gated_norm_override=ops.gated_norm(i, mixer),
            gate_collector=ops.gate_collector if ops.record else None,
            collector=ops.collector if (ops.record or ops.track_a) else None,
            refresh=(lambda l, s, x: rf(l, s, x)) if (refresh is not None or ops.record) else None,
        ).float()
    residual = residual + hidden
    h = ops.norm(len(bb.layers), "norm_f", residual, bb.norm_f.weight, bb.norm_f.eps)
    return model.lm_head(h.to(model.lm_head.weight.dtype)).float()


def save_calibration(ops: ExactOps, path) -> None:
    torch.save({"stats": {k: dict(v) for k, v in ops.stats.items()}, "absmax": ops.absmax}, path)


def load_calibration(path) -> ExactOps:
    d = torch.load(path, weights_only=False)
    ops = ExactOps(record=False)
    for k, v in d["stats"].items():
        ops.stats[k] = v
    ops.absmax = d["absmax"]
    return ops


def calibrate(model, blocks, device, chunk_size=None) -> ExactOps:
    """Run the EXACT model over `blocks` and record every operator's input range
    and every refresh point's max |activation|."""
    ops = ExactOps(record=True)
    with torch.no_grad():
        for i in range(blocks.shape[0]):
            allpoly_forward(model, blocks[i:i + 1].to(device), ops,
                            refresh=ops.refresh, chunk_size=chunk_size)
    return ops
