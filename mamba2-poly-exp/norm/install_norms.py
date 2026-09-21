"""Install / uninstall the Path A and Path B norm replacements, per stage.

Three sites, two different invocation mechanisms, so two different swap methods:

    norm_pre    (x24)  invoked as a MODULE  -> swap the module on its parent
    norm_f      (x1)   invoked as a MODULE  -> swap the module on its parent
    norm_gated  (x24)  invoked as a FUNCTION from the patched mixer forward
                       -> pass an override into mamba2_reference_forward

Stages follow the brief: A1 = pre-norms only, A2 = + gated, A3 = + norm_f, so
each stage can be distilled and evaluated before the next operator is replaced.
Replacing all of them at once from the pretrained checkpoint is the thing most
likely to fail while telling you nothing about which one caused it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from real_mamba.model import iter_mixers

STAGES = {
    "exact": (),
    "A1": ("norm_pre",),
    "A2": ("norm_pre", "norm_gated"),
    "A3": ("norm_pre", "norm_gated", "norm_f"),
}

# Path B uses the same staging; only the operator differs.
B_STAGES = {k.replace("A", "B"): v for k, v in STAGES.items() if k != "exact"}


def find_norm_sites(model):
    """{'norm_pre': {layer: (parent, attr, module)}, 'norm_f': ..., 'norm_gated': {...}}"""
    sites = {"norm_pre": {}, "norm_f": {}, "norm_gated": {}}
    named = dict(model.named_modules())
    for name, mod in named.items():
        if not (hasattr(mod, "weight") and hasattr(mod, "eps")):
            continue
        if name.endswith("norm_f"):
            parent = named[name.rsplit(".", 1)[0]]
            sites["norm_f"][0] = (parent, "norm_f", mod)
        elif name.endswith(".norm") and ".mixer" not in name and ".layers." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            parent = named[name.rsplit(".", 1)[0]]
            sites["norm_pre"][layer] = (parent, "norm", mod)
    for layer, mixer in iter_mixers(model):
        if hasattr(mixer, "norm"):
            sites["norm_gated"][layer] = (mixer, "norm", mixer.norm)
    return sites


@dataclass
class NormHandle:
    """What was swapped, so it can be put back exactly."""
    model: object = None
    originals: list = field(default_factory=list)   # (parent, attr, old_module)
    replacements: nn.ModuleDict = None
    gated_overrides: dict = field(default_factory=dict)
    stage: str = "exact"

    def restore(self):
        for parent, attr, old in self.originals:
            setattr(parent, attr, old)
        self.originals.clear()
        self.gated_overrides.clear()
        if self.model is not None and hasattr(self.model, "norm_replacements"):
            del self.model.norm_replacements
        self.stage = "exact"


def install_path_a(model, stats, stage="A3", precision_bits=None,
                   train_weight=True, verbose=True):
    """Swap in ConstDivisorNorm / GatedConstDivisor for the sites in `stage`.

    `stats` is the Phase 1 JSON; each instance's c is initialised from that
    layer's median sqrt(v), which is the closest a constant can get to the
    divisor RMSNorm would have produced on typical text.
    """
    from norm.const_norm import ConstDivisorNorm, GatedConstDivisor

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}")
    sites = find_norm_sites(model)
    h = NormHandle(model=model, stage=stage)
    holder = {}

    def c_init_for(site, layer, fallback=1.0):
        try:
            return stats[site][str(layer)]["sqrt_median"]
        except (KeyError, TypeError):
            return fallback

    for site in STAGES[stage]:
        for layer, (parent, attr, old) in sorted(sites[site].items()):
            c0 = c_init_for(site, layer)
            if site == "norm_gated":
                new = GatedConstDivisor(old.weight.data, c0,
                                        group_size=getattr(old, "group_size", None),
                                        train_weight=train_weight,
                                        precision_bits=precision_bits)
                h.gated_overrides[layer] = new
            else:
                new = ConstDivisorNorm(old.weight.data, c0,
                                       eps=getattr(old, "eps", 1e-5),
                                       train_weight=train_weight,
                                       precision_bits=precision_bits)
                setattr(parent, attr, new)
                h.originals.append((parent, attr, old))
            holder[f"{site}_{layer}"] = new

    # register so model.parameters() sees the learned c's and gammas
    h.replacements = nn.ModuleDict(holder)
    ref = next(model.parameters())
    model.norm_replacements = h.replacements.to(device=ref.device)
    if verbose:
        cs = [float(m.c) for m in holder.values()]
        print(f"[norm] stage {stage}: replaced {len(holder)} instances "
              f"({', '.join(STAGES[stage])}); c init range "
              f"[{min(cs):.4g}, {max(cs):.4g}]; ct-ct depth 0 each")
    return h


@contextlib.contextmanager
def path_a(model, stats, stage="A3", **kw):
    h = install_path_a(model, stats, stage=stage, **kw)
    try:
        yield h
    finally:
        h.restore()


def norm_kwargs_for_mixer(handle, layer_idx):
    """The kwargs to pass into mamba2_reference_forward for this layer."""
    ov = handle.gated_overrides.get(layer_idx)
    return {"gated_norm_override": ov} if ov is not None else {}


def frac_outside_band(stats, handle, site, layer, factor=4.0):
    """Fraction of tokens whose true sqrt(v) falls outside [c/factor, factor*c].

    The direct measure of how wrong a constant divisor is. Computed from the
    Phase 1 reservoir summary rather than a fresh pass, so it is the p1/p99
    bracket rather than an exact count -- reported as such.
    """
    import math
    try:
        s = stats[site][str(layer)]
    except (KeyError, TypeError):
        return None
    key = f"{site}_{layer}"
    m = handle.replacements[key] if handle.replacements and key in handle.replacements else None
    if m is None:
        return None
    c = float(m.c)
    lo, hi = (c / factor) ** 2, (c * factor) ** 2      # compare in v-space
    return {"c": c, "band_v": [lo, hi],
            "p1_inside": s["p1"] >= lo, "p99_inside": s["p99"] <= hi,
            "v_p1": s["p1"], "v_p99": s["p99"], "v_median": s["median"]}


@contextlib.contextmanager
def exact_mode(handle: NormHandle, record: bool = False):
    """Run the replaced norms in EXACT mode -- i.e. as the teacher.

    Every other weight in the network is shared, so the teacher and the student
    differ in exactly one operator and nothing else. One model in memory.
    """
    mods = list(handle.replacements.values()) if handle.replacements else []
    prev = [(m.exact, m.record) for m in mods]
    for m in mods:
        m.exact, m.record = True, record
    try:
        yield
    finally:
        for m, (e, r) in zip(mods, prev):
            m.exact, m.record = e, r


@contextlib.contextmanager
def recording(handle: NormHandle):
    """Capture each replaced norm's output, for the auxiliary MSE term."""
    mods = list(handle.replacements.values()) if handle.replacements else []
    prev = [m.record for m in mods]
    for m in mods:
        m.record, m.last_out = True, None
    try:
        yield
    finally:
        for m, r in zip(mods, prev):
            m.record = r


def norm_outputs(handle: NormHandle):
    """{site_layer: tensor} of whatever the replaced norms last produced."""
    if not handle.replacements:
        return {}
    return {k: m.last_out for k, m in handle.replacements.items()
            if m.last_out is not None}


# =============================================================================
# Path B installation
# =============================================================================

class GatedNewtonInvSqrt(nn.Module):
    """Gated-norm variant of NewtonInvSqrtNorm.

    The gated site takes (x, z) and normalises `x * silu(z)`, so the mean-square
    argument is computed AFTER gating. The exact SiLU is kept deliberately: this
    step isolates the norm.
    """

    def __init__(self, weight, s_init, t_steps=2, eps=1e-5, group_size=None,
                 train_weight=True, precision_bits=None):
        super().__init__()
        import math as _m
        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad_(train_weight)
        self.register_buffer("weight_orig", weight.detach().clone())
        self.log_s = nn.Parameter(torch.tensor(_m.log(max(s_init, 1e-12)), dtype=torch.float32))
        self.log_y0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.t_steps, self.eps, self.group_size = t_steps, eps, group_size
        self.precision_bits = precision_bits
        self.exact = False
        self.record = False
        self.last_out = None
        self.last_vprime = None

    @property
    def s(self):
        return self.log_s.exp()

    @property
    def ct_ct_depth(self):
        from norm.newton_norm import newton_depth
        return newton_depth(self.t_steps)["total_ct_ct_depth"]

    def forward(self, x, z):
        import torch.nn.functional as _F

        from norm.const_norm import round_to_bits
        dtype = x.dtype
        xf = x.float() * _F.silu(z.float())
        if self.exact:
            rstd = 1.0 / torch.sqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
            out = xf * rstd * self.weight_orig.float()
        else:
            v = xf.square().mean(dim=-1, keepdim=True) + self.eps
            vp = v / self.s
            self.last_vprime = vp if self.record else None
            y = self.log_y0.exp().expand_as(vp)
            for _ in range(self.t_steps):
                y = y * (1.5 - 0.5 * vp * y * y)
            out = xf * (y / torch.sqrt(self.s)) * self.weight.float()
            if self.precision_bits:
                out = round_to_bits(out, self.precision_bits)
        if self.record:
            self.last_out = out
        return out.to(dtype)


def install_path_b(model, stats, stage="B3", t_steps=2, precision_bits=None,
                   train_weight=True, verbose=True):
    """Swap in the prescaled-Newton inverse sqrt for the sites in `stage`.

    s_layer is initialised from the Phase 1 per-layer MEDIAN of v, so that v/s
    starts centred at 1 -- the middle of the measured convergence basin
    v' in [0.25, 2.0]. Outside that basin the iteration DIVERGES and more steps
    make it worse, so the prescale is doing the real work here, not the degree.
    """
    from norm.newton_norm import NewtonInvSqrtNorm

    sites = STAGES[stage.replace("B", "A")] if stage != "exact" else ()
    h = NormHandle(model=model, stage=stage)
    found = find_norm_sites(model)
    holder = {}

    def s_init_for(site, layer, fallback=1.0):
        try:
            return stats[site][str(layer)]["median"]
        except (KeyError, TypeError):
            return fallback

    for site in sites:
        for layer, (parent, attr, old) in sorted(found[site].items()):
            s0 = s_init_for(site, layer)
            if site == "norm_gated":
                new = GatedNewtonInvSqrt(old.weight.data, s0, t_steps=t_steps,
                                         eps=getattr(old, "eps", 1e-5),
                                         group_size=getattr(old, "group_size", None),
                                         train_weight=train_weight,
                                         precision_bits=precision_bits)
                h.gated_overrides[layer] = new
            else:
                new = NewtonInvSqrtNorm(old.weight.data, s0, t_steps=t_steps,
                                        eps=getattr(old, "eps", 1e-5),
                                        train_weight=train_weight,
                                        precision_bits=precision_bits)
                setattr(parent, attr, new)
                h.originals.append((parent, attr, old))
            holder[f"{site}_{layer}"] = new

    h.replacements = nn.ModuleDict(holder)
    ref = next(model.parameters())
    model.norm_replacements = h.replacements.to(device=ref.device)
    if verbose and holder:
        d = next(iter(holder.values())).ct_ct_depth
        print(f"[norm] stage {stage}: replaced {len(holder)} instances with "
              f"Newton t={t_steps}; ct-ct depth {d} each "
              f"({d * len(holder) // max(len(holder), 1)} per instance)")
    return h
