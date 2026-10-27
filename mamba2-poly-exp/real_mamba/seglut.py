"""ROSETTA-style segmented lookup tables, simulated in PLAINTEXT.

What this answers: if the gates we could not make polynomial were instead sent to
TFHE and evaluated with ROSETTA's adaptive segmented LUT (Yu et al., CCS '26,
arXiv 2609.16915, section 3), what would the pretrained model's perplexity do?
No cryptography runs here. This is the same "ciphertext-aligned plaintext" test
the paper uses for its own PPL numbers: every replaced operator computes exactly
what the encrypted protocol would compute, plus optional noise.

THE CONSTRUCTION (ROSETTA section 3.1)
--------------------------------------
The input domain [tau_0, tau_S] is cut into N_seg segments. Each segment carries
its own table of N_br entries (N_br = 2048 = the blind-rotation ring degree),
spaced UNIFORMLY inside the segment, and each entry is a line a*x + b rather
than a constant. So the evaluated function is piecewise linear with
N_seg * N_br pieces, and the segments let those pieces bunch up where the
function bends. We use the chord through the two knots of each interval, which
makes the interpolant continuous.

The encrypted lookup (section 3.2) is three steps, mirrored one-for-one below:

    1. S = sum_j [x >= tau_j]                 N_seg-1 homomorphic comparisons
    2. I = floor((x - tau_S) / w_S * N_br)    one pt-ct mult, one PBS, ModSwitch
    3. f = a_{S,I} * x + b_{S,I}              construct-then-select, two rounds

Per element that is N_seg-1 HomComp, N_seg+1 pt-ct mults, N_seg+2 blind
rotations and N_seg+1 traces, plus the CKKS<->TFHE conversion, which costs about
one CKKS bootstrap (SlotToCoeff + ModRecover). ZERO CKKS levels are consumed by
the function itself, which is the entire attraction: `ct_ct_depth = 0`.

WHAT A POLYNOMIAL COULD NOT GIVE US, AND WHAT THIS DOES INSTEAD
--------------------------------------------------------------
Every gate failure in GATES.md was a polynomial leaving the range it must stay
in: Delta < 0 from softplus, a > 1 from the fused gate, SiLU-conv exploding just
off its interval. A table cannot invent values it does not store -- but only if
inputs OUTSIDE the table are handled, and the protocol as written does not
handle them. So `oob` is a first-class switch, not a detail:

    clamp   x is clamped into [tau_0, tau_S] first. Output is bounded by the
            stored values. Costs two extra comparisons per element in TFHE.
    extend  the end intervals' lines are extrapolated. What you get if the
            interval index is saturated but x is not.
    wrap    what the protocol LITERALLY computes. The segment index saturates
            (it is a sum of comparisons) but the local index does not, and blind
            rotation is negacyclic in X^N_br + 1: an index in [N_br, 2*N_br)
            selects the NEGATED entry I - N_br. An input just past the last
            boundary therefore returns roughly -f(tau_{S-1}). ROSETTA does not
            discuss this; it is the off-interval failure mode again.

NOISE
-----
ROSETTA reports precision as -log2(RMSE) and gets 12-19 bits. `noise_bits` adds
Gaussian noise to every output, standing in for PBS and scheme-conversion noise,
with one of two scalings:

    abs     std = 2^-noise_bits. ROSETTA's metric taken literally.
    range   std = 2^-noise_bits * max|f| over the table's domain. TFHE encodes
            the output as fixed point scaled to fit its LARGEST value, so the
            noise is absolute in that scale. For 1/sqrt(v) over a wide v range
            the small outputs then carry far fewer bits than the headline: over
            our 6e6 range, max/min of the output is ~2400, i.e. ~11 bits of the
            budget are spent on dynamic range alone.

`cmp_noise_bits` perturbs the input of the
step-1 comparisons only, which models HomComp picking the WRONG segment near a
boundary -- after which the local index lands outside [0, N_br) and, under
`wrap`, the sign flips. Both default to off.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FUNCTIONS = ("silu", "softplus", "invsqrt", "fused_dt")
OOB_MODES = ("clamp", "extend", "wrap")


def _apply_fn(name, k, A=None, b=None):
    """The exact function, evaluated at table knots. Offline in the real protocol."""
    if name == "silu":
        return F.silu(k)
    if name == "softplus":
        return F.softplus(k)
    if name == "invsqrt":
        return torch.rsqrt(k)
    if name == "fused_dt":
        # a = exp(A_h * softplus(x + b_h)); A_h, b_h are weights, hence plaintext
        return torch.exp(A * F.softplus(k + b))
    raise ValueError(f"unknown function {name!r}; have {FUNCTIONS}")


# =============================================================================
# segment boundaries
# =============================================================================

def _widen_linear(lo, hi, margin):
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * (1.0 + margin)
    half = np.maximum(half, 1e-3)              # degenerate channels: never zero-width
    return mid - half, mid + half


def make_boundaries(lo, hi, n_seg, kind, fn=None, A=None, b=None, margin=0.05,
                    n_grid=4097):
    """Per-channel boundaries tau, shape (C, n_seg + 1), float64.

    lo, hi: (C,) measured input range per channel.
    kind:
      uniform   equal-width segments (ROSETTA's choice for SiLU / GELU)
      log       geometric segments, needs lo > 0 (ROSETTA's choice for 1/x, 1/sqrt x)
      adaptive  equidistribute sqrt|f''|, the density that minimises the max
                error of a piecewise-linear interpolant. Segments then sit where
                the function bends, which is the point of the "adaptive" in the
                paper's title. Used for the fused dt gate, which for large |A_h|
                is nearly a step.
    """
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    if kind == "log":
        if np.any(lo <= 0):
            raise ValueError("log boundaries need a strictly positive domain")
        llo, lhi = _widen_linear(np.log(lo), np.log(hi), margin)
        return np.exp(np.linspace(llo, lhi, n_seg + 1, axis=1))
    lo, hi = _widen_linear(lo, hi, margin)
    if kind == "uniform" or n_seg == 1:
        return np.linspace(lo, hi, n_seg + 1, axis=1)
    if kind != "adaptive":
        raise ValueError(f"unknown boundary kind {kind!r}")

    grid = np.linspace(lo, hi, n_grid, axis=1)                         # (C, G)
    At = None if A is None else torch.as_tensor(A, dtype=torch.float64)[:, None]
    bt = None if b is None else torch.as_tensor(b, dtype=torch.float64)[:, None]
    f = _apply_fn(fn, torch.as_tensor(grid), At, bt).numpy()
    dx = (hi - lo)[:, None] / (n_grid - 1)
    f2 = np.abs(f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / dx ** 2          # interior points
    dens = np.sqrt(np.pad(f2, ((0, 0), (1, 1)), mode="edge"))            # back to (C, G)
    # a floor, so a flat stretch still gets a segment edge eventually
    dens = dens + 1e-3 * dens.max(axis=1, keepdims=True) + 1e-12
    cdf = np.concatenate([np.zeros((len(lo), 1)),
                          np.cumsum(0.5 * (dens[:, 1:] + dens[:, :-1]), axis=1)], axis=1)
    cdf = cdf / cdf[:, -1:]
    q = np.linspace(0.0, 1.0, n_seg + 1)
    tau = np.stack([np.interp(q, cdf[c], grid[c]) for c in range(len(lo))])
    tau[:, 0], tau[:, -1] = lo, hi
    # boundaries must stay strictly increasing for the segment index to be valid
    min_gap = 1e-6 * (hi - lo)[:, None]
    for j in range(1, n_seg + 1):
        tau[:, j] = np.maximum(tau[:, j], tau[:, j - 1] + min_gap[:, 0])
    return tau


# =============================================================================
# the gate
# =============================================================================

class SegLUT(nn.Module):
    """f(x) by ROSETTA's segmented LUT, one table per channel of the last dim.

    Per-channel tables are free for the same reason per-head polynomials were:
    the tables are plaintext, and TFHE processes every element on its own anyway.
    """

    degree = 1          # piecewise linear; read by patch._FusedTransition
    ct_ct_depth = 0     # no CKKS levels: the function is evaluated in TFHE

    def __init__(self, fn, tau, n_br=2048, oob="clamp", A=None, b=None,
                 noise_bits=None, noise_scale="abs", cmp_noise_bits=None, seed=0,
                 name=None):
        super().__init__()
        if fn not in FUNCTIONS:
            raise ValueError(f"fn must be one of {FUNCTIONS}")
        if oob not in OOB_MODES:
            raise ValueError(f"oob must be one of {OOB_MODES}")
        if noise_scale not in ("abs", "range"):
            raise ValueError("noise_scale must be 'abs' or 'range'")
        tau = torch.as_tensor(np.asarray(tau, dtype=np.float64), dtype=torch.float32)
        if tau.ndim != 2 or tau.shape[1] < 2:
            raise ValueError("tau must be (C, n_seg + 1)")
        self.fn, self.n_br, self.oob = fn, int(n_br), oob
        self.nchannels, self.n_seg = int(tau.shape[0]), int(tau.shape[1] - 1)
        self.noise_bits, self.noise_scale = noise_bits, noise_scale
        self.cmp_noise_bits, self.seed = cmp_noise_bits, seed
        self._name = name or f"seglut_{fn}"
        self.register_buffer("tau", tau)
        if fn == "fused_dt":
            self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
            self.register_buffer("b", torch.as_tensor(b, dtype=torch.float32))
        else:
            self.A = self.b = None
        # max|f| over the domain, per channel: the fixed-point scale for `range` noise
        with torch.no_grad():
            g = torch.linspace(0, 1, 4097, dtype=torch.float64)[:, None]
            t64 = tau.double()
            xs = t64[:, 0] + (t64[:, -1] - t64[:, 0]) * g                       # (G, C)
            fx = _apply_fn(fn, xs, None if self.A is None else self.A.double(),
                           None if self.b is None else self.b.double())
            self.register_buffer("out_scale", fx.abs().max(dim=0).values.float())
        self._gen = None
        self.track = False
        self.reset_counters()

    @property
    def name(self):
        return self._name

    @property
    def tfhe_ops_per_element(self):
        """ROSETTA section 3.2, Algorithm 1. `clamp` adds two comparisons."""
        s = self.n_seg
        return {"homcomp": s - 1 + (2 if self.oob == "clamp" else 0),
                "pt_ct_mult": s + 1, "blind_rotation": s + 2, "trace": s + 1,
                "scheme_conversion": "~1 CKKS bootstrap per batch (StC + ModRecover)"}

    def reset_counters(self):
        self.n_seen = self.n_oob = self.n_neg = self.n_gt1 = 0

    def reseed(self, seed=None):
        self.seed = self.seed if seed is None else seed
        self._gen = None

    def _noise(self, like, bits):
        if self._gen is None or self._gen.device != like.device:
            self._gen = torch.Generator(device=like.device)
            self._gen.manual_seed(int(self.seed))
        return torch.randn(like.shape, generator=self._gen, device=like.device,
                           dtype=like.dtype) * (2.0 ** -float(bits))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        if xf.shape[-1] != self.nchannels and self.nchannels != 1:
            raise ValueError(f"expected last dim {self.nchannels}, got {tuple(x.shape)}")
        tau = self.tau
        t_lo, t_hi = tau[:, 0], tau[:, -1]
        if self.track:
            oob = (xf < t_lo) | (xf > t_hi)
            self.n_oob += int(oob.sum())
            self.n_seen += oob.numel()

        xin = torch.minimum(torch.maximum(xf, t_lo), t_hi) if self.oob == "clamp" else xf

        # step 1: segment index from comparisons. It saturates by construction.
        xcmp = xin if not self.cmp_noise_bits else xin + self._noise(xin, self.cmp_noise_bits)
        seg_lo = torch.zeros_like(xin) + tau[:, 0]
        seg_hi = torch.zeros_like(xin) + tau[:, 1]
        for j in range(1, self.n_seg):
            ge = xcmp >= tau[:, j]
            seg_lo = torch.where(ge, tau[:, j], seg_lo)
            seg_hi = torch.where(ge, tau[:, j + 1], seg_hi)

        # step 2: local interval index. It does NOT saturate.
        w = (seg_hi - seg_lo) / self.n_br
        idx = torch.floor((xin - seg_lo) / w)
        if self.oob == "wrap":
            m = torch.remainder(idx, 2 * self.n_br)
            sign = torch.where(m >= self.n_br, -1.0, 1.0)
            idx = torch.remainder(m, self.n_br)
        else:
            sign = 1.0
            idx = idx.clamp(0, self.n_br - 1)

        # step 3: the selected line a*x + b, through the interval's two knots
        k0 = seg_lo + idx * w
        f0 = _apply_fn(self.fn, k0, self.A, self.b)
        f1 = _apply_fn(self.fn, k0 + w, self.A, self.b)
        out = sign * (f0 + (f1 - f0) / w * (xin - k0))

        if self.noise_bits:
            eps = self._noise(out, self.noise_bits)
            out = out + (eps * self.out_scale if self.noise_scale == "range" else eps)
        if self.track:
            self.n_neg += int((out < 0).sum())
            self.n_gt1 += int((out > 1).sum())
        return out.to(dtype)

    @torch.no_grad()
    def precision(self, per_segment=512, seed=0):
        """-log2 RMSE (ROSETTA's 'bits') per channel, over its own domain, noise off.

        Points are drawn uniformly at random inside EACH segment, so every
        segment counts equally and no point sits on a knot. (A uniform grid over
        the whole domain lines up with the knots of a single-segment table and
        reports zero error, which is how the first version of this read 25 bits
        for one 2048-entry table over a 6e6 range.)

        Returns per-channel arrays: bits, max abs error, max relative error.
        This is the deterministic table error only; it is what the noise budget
        has to be compared against.
        """
        saved = (self.noise_bits, self.cmp_noise_bits, self.track)
        self.noise_bits = self.cmp_noise_bits = None
        self.track = False
        try:
            tau = self.tau.double()
            gen = torch.Generator(device="cpu").manual_seed(seed)
            u = torch.rand(per_segment, self.n_seg, generator=gen,
                           dtype=torch.float64).to(tau.device)
            w = tau[:, 1:] - tau[:, :-1]                                   # (C, S)
            xs = (tau[None, :, :-1] + u[:, None, :] * w[None])             # (P, C, S)
            xs = xs.permute(0, 2, 1).reshape(-1, self.nchannels).float()   # (P*S, C)
            approx = self.forward(xs).double()
            A = None if self.A is None else self.A.double()
            b = None if self.b is None else self.b.double()
            exact = _apply_fn(self.fn, xs.double(), A, b)
        finally:
            self.noise_bits, self.cmp_noise_bits, self.track = saved
        err = (approx - exact).abs()
        rmse = err.square().mean(dim=0).sqrt().clamp_min(1e-30)
        return {"bits": (-torch.log2(rmse)).cpu().numpy(),
                "max_abs": err.max(dim=0).values.cpu().numpy(),
                "max_rel": (err / exact.abs().clamp_min(1e-12)).max(dim=0).values.cpu().numpy()}

    def __repr__(self):
        return (f"SegLUT({self.fn}, channels={self.nchannels}, n_seg={self.n_seg}, "
                f"n_br={self.n_br}, oob={self.oob}, noise_bits={self.noise_bits})")


def build_seglut(fn, lo, hi, n_seg=4, n_br=2048, boundaries="uniform", margin=0.05,
                 A=None, b=None, **kw):
    tau = make_boundaries(lo, hi, n_seg, boundaries, fn=fn, A=A, b=b, margin=margin)
    return SegLUT(fn, tau, n_br=n_br, A=A, b=b, **kw)


# =============================================================================
# RMSNorm with the inverse square root, and only that, in the table
# =============================================================================

class _InvSqrtNormBase(nn.Module):
    """Shared machinery: an exact mode, a calibration mode, and a LUT mode.

    ROSETTA's decomposition (section 4.1, Figure 8a): of RMSNorm's seven
    primitives only 1/sqrt(.) is non-arithmetic. x^2, the mean, +eps and the two
    final multiplies stay in CKKS. So the LUT sees exactly one scalar per token
    per norm, which is the workload TFHE is good at.
    """

    def __init__(self, weight, eps):
        super().__init__()
        # a Parameter so `self.norm.weight.dtype` keeps working in Block.forward
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.eps = float(eps)
        self.lut = None
        self.use_lut = False
        self.calibrating = False
        self.register_buffer("v_min", torch.tensor(float("inf")))
        self.register_buffer("v_max", torch.tensor(float("-inf")))

    def _rstd(self, v):
        if self.calibrating:
            self.v_min = torch.minimum(self.v_min, v.min().float())
            self.v_max = torch.maximum(self.v_max, v.max().float())
        if self.use_lut:
            return self.lut(v)
        return torch.rsqrt(v)

    def build_lut(self, n_seg=4, n_br=2048, boundaries="log", margin=0.05, **kw):
        if not torch.isfinite(self.v_min):
            raise RuntimeError("no calibration samples reached this norm -- the "
                               "module was never called (fused_add_norm still on?)")
        self.lut = build_seglut("invsqrt", [float(self.v_min)], [float(self.v_max)],
                                n_seg=n_seg, n_br=n_br, boundaries=boundaries,
                                margin=margin, **kw).to(self.weight.device)
        return self.lut


class SegLUTRMSNorm(_InvSqrtNormBase):
    """Replacement for the block pre-norms and norm_f (called as modules)."""

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        xf = x.float()
        rstd = self._rstd(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rstd * self.weight.float()).to(dtype)


class SegLUTGatedNorm(_InvSqrtNormBase):
    """Replacement for the mixer's gated norm, norm(x * silu(z)).

    Passed as `gated_norm_override`. `silu` is the SiLU used inside it, exact by
    default, so the SiLU-norm table can be tested here too without touching the
    norm.
    """

    def __init__(self, weight, eps, group_size=None):
        super().__init__(weight, eps)
        self.group_size = group_size
        self.silu = None

    def forward(self, x, z):
        dtype = x.dtype
        zf = z.float()
        xf = x.float() * (self.silu(zf).float() if self.silu is not None else F.silu(zf))
        D = xf.shape[-1]
        gs = self.group_size or D
        if gs == D:
            out = xf * self._rstd(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        else:
            xg = xf.unflatten(-1, (D // gs, gs))
            out = (xg * self._rstd(xg.square().mean(dim=-1, keepdim=True) + self.eps)
                   ).flatten(-2)
        return (out * self.weight.float()).to(dtype)


def summarize_precision(luts):
    """Median / worst of `SegLUT.precision()` over a family of tables."""
    bits, mabs, mrel = [], [], []
    for lut in luts:
        p = lut.precision()
        bits.append(p["bits"]); mabs.append(p["max_abs"]); mrel.append(p["max_rel"])
    bits, mabs, mrel = map(np.concatenate, (bits, mabs, mrel))
    return {"n_tables": int(bits.size),
            "bits_median": float(np.median(bits)), "bits_worst": float(bits.min()),
            "max_abs_median": float(np.median(mabs)), "max_abs_worst": float(mabs.max()),
            "max_rel_median": float(np.median(mrel)), "max_rel_worst": float(mrel.max())}


def frac(n, d):
    return (n / d) if d else float("nan")


__all__ = ["SegLUT", "SegLUTRMSNorm", "SegLUTGatedNorm", "build_seglut",
           "make_boundaries", "summarize_precision", "FUNCTIONS", "OOB_MODES", "frac"]
