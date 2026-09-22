"""FHE-friendly replacements for the OTHER non-polynomial gates in Mamba-2.

The exp gate is handled in real_mamba/transitions.py. This file covers what is
left, in increasing order of difficulty:

    softplus   -> Delta          easy-ish, but see the amplification note below
    SiLU (x2)  -> activations    easy, bounded, well-studied
    RMSNorm    -> 1/sqrt(.)      hard: needs a reciprocal AND a square root.
                                 NOT implemented here yet, deliberately.

WHY DEPTH-PER-LAYER IS THE CURRENCY
-----------------------------------
The 130m has 24 blocks in series, so whatever one block costs is paid 24 times.
Counting only what is already verified -- our depth-2 exp gate, the depth-10
prefix-product tree over a 1024-token sequence, ~2 for the einsums -- a block is
already ~14 levels, i.e. ~336 for the network. A CKKS setup typically affords
10-30 levels before a bootstrap. So bootstrapping is structural, not an
optimisation, and the thing worth minimising is **levels per layer**: a gate that
costs 2 instead of 6 saves 96 levels network-wide.

Each class below reports `ct_ct_depth` for exactly that accounting.

THE AMPLIFICATION PROBLEM, AND THE FUSED GATE
---------------------------------------------
softplus is not just another activation here. Its output feeds
`z = A * Delta`, and |A| reaches 3.6e4 in the trained 130m. So an absolute error
of 1e-4 in Delta becomes an error of 3.6 in z, and exp(z) then moves by a factor
of e^3.6 ~ 36. Approximating softplus and exp separately pays that amplification
AND two lots of depth.

The alternative is to approximate the whole per-head chain as ONE function:

    a_h(x) = exp( A_h * softplus(x + dt_bias_h) )

which maps R -> (0, 1], is smooth and monotone decreasing, and needs
**one** polynomial and **one** lot of depth instead of two. `A_h` and
`dt_bias_h` are weights, so they are plaintext and can be folded in for free --
exactly the argument that made the per-head exp gate free.

Both routes are implemented (`PolySoftplus` for the separate route, `FusedDtGate`
for the fused one) because which wins is an empirical question: for heads with
very large |A| the fused function is nearly a step, and a low-degree polynomial
cannot represent a step. Measure, then choose.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_mamba.polynomial import PowerSchedule, fit_general


# =============================================================================
# exact references (what we are replacing)
# =============================================================================

class ExactSoftplus(nn.Module):
    """Delta = softplus(x). The real thing."""
    name, degree, ct_ct_depth = "exact_softplus", None, None

    def forward(self, x):
        return F.softplus(x)

    def __repr__(self):
        return "ExactSoftplus()"


class ExactSiLU(nn.Module):
    """SiLU(x) = x * sigmoid(x). The real thing."""
    name, degree, ct_ct_depth = "exact_silu", None, None

    def forward(self, x):
        return F.silu(x)

    def __repr__(self):
        return "ExactSiLU()"


# =============================================================================
# a shared polynomial gate over one scalar variable
# =============================================================================

class ScalarPolyGate(nn.Module):
    """P(x) ~= f(x) on a measured interval, evaluated at minimal ct-ct depth.

    Same evaluation discipline as baby_mamba.polynomial.PolyExp: explicit powers
    rather than Horner, so the depth is ceil(log2(degree)) instead of `degree`.
    Coefficients are ciphertext-times-PLAINTEXT, so they add no ct-ct depth.

    `scale` normalises the input to roughly [-1, 1] before evaluation. It is a
    plaintext constant, so it is free, and it keeps the coefficients O(1) --
    without it a degree-4 fit over a wide interval produces coefficients spanning
    many orders of magnitude, which is numerically fragile and untrainable. We
    learned that the hard way on the exp gate.
    """

    def __init__(self, coeffs, interval, scale=1.0, trainable=False, name="poly_gate"):
        super().__init__()
        c = torch.as_tensor(np.asarray(coeffs, dtype=np.float64), dtype=torch.float32)
        if c.ndim != 1:
            raise ValueError("coeffs must be 1-D, lowest power first")
        self.degree = int(c.numel() - 1)
        self.schedule = PowerSchedule.binary(self.degree)
        self.interval = (float(interval[0]), float(interval[1]))
        self._name = name
        self.register_buffer("inv_scale", torch.tensor(1.0 / float(scale)))
        self.register_buffer("scale", torch.tensor(float(scale)))
        if trainable:
            self.coeffs = nn.Parameter(c)
        else:
            self.register_buffer("coeffs", c)

    @property
    def ct_ct_depth(self):
        return self.schedule.ct_ct_depth

    @property
    def ct_ct_mults(self):
        return self.schedule.ct_ct_mults

    @property
    def name(self):
        return self._name

    def forward(self, x):
        c = self.coeffs.to(device=x.device, dtype=x.dtype)
        t = x * self.inv_scale.to(device=x.device, dtype=x.dtype)          # ct x pt, no ct-ct depth
        powers = {1: t}
        for target, lo, hi in self.schedule.steps:
            powers[target] = powers[lo] * powers[hi]        # one ct-ct each
        out = torch.zeros_like(t) + c[0]
        for k in range(1, self.degree + 1):
            out = out + c[k] * powers[k]
        return out

    def to_dict(self):
        return {"name": self.name, "degree": self.degree,
                "ct_ct_depth": self.ct_ct_depth, "ct_ct_mults": self.ct_ct_mults,
                "interval": list(self.interval), "scale": float(self.scale),
                "coeffs_lowest_first": [float(v) for v in self.coeffs.detach().cpu()],
                "trainable": isinstance(self.coeffs, nn.Parameter)}

    def __repr__(self):
        return (f"{type(self).__name__}(degree={self.degree}, "
                f"ct_ct_depth={self.ct_ct_depth}, interval={self.interval})")


def _fit_scalar(fn, degree, lo, hi, method="chebyshev", weight="uniform"):
    """Fit `fn` on [lo, hi] in a normalised variable t = x / s, s = max|lo|,|hi|."""
    s = max(abs(lo), abs(hi)) or 1.0
    coeffs = fit_general(lambda t: fn(np.asarray(t, dtype=np.float64) * s),
                         degree, lo / s, hi / s, method=method, weight=weight)
    return coeffs, s


# =============================================================================
# softplus
# =============================================================================

class PolySoftplus(ScalarPolyGate):
    """P(x) ~= softplus(x) = log(1 + e^x).

    INVARIANT WE MUST NOT BREAK: Delta > 0. A negative Delta flips the sign of
    z = A*Delta, which makes exp(z) > 1, which makes the recurrence EXPAND. That
    is the same failure mode Part 8 exists to catch, so `frac(Delta < 0)` is
    reported everywhere this is used and never clamped away.
    """

    @classmethod
    def fit(cls, lo, hi, degree=4, method="chebyshev", trainable=False):
        def softplus(x):
            x = np.asarray(x, dtype=np.float64)
            # numerically safe: log1p(exp(x)) overflows for large x
            return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))
        c, s = _fit_scalar(softplus, degree, lo, hi, method=method)
        return cls(c, (lo, hi), scale=s, trainable=trainable,
                   name=f"poly_softplus{degree}")


# =============================================================================
# SiLU
# =============================================================================

class PolySiLU(ScalarPolyGate):
    """P(x) ~= SiLU(x) = x * sigmoid(x).

    Bounded below by about -0.2785 (at x ~ -1.278) and unbounded above, where it
    approaches x. The upper tail is the easy part -- a polynomial can follow a
    line. The hard part is the knee near 0 and the flat left tail.
    """

    @classmethod
    def fit(cls, lo, hi, degree=4, method="chebyshev", trainable=False):
        def silu(x):
            x = np.asarray(x, dtype=np.float64)
            return x / (1.0 + np.exp(-np.clip(x, -700, 700)))   # ALLOW-CLAMP: overflow guard in the OFFLINE fit only; never applied to data
        c, s = _fit_scalar(silu, degree, lo, hi, method=method)
        return cls(c, (lo, hi), scale=s, trainable=trainable, name=f"poly_silu{degree}")


# =============================================================================
# the fused dt gate: x -> a, one polynomial, one lot of depth
# =============================================================================

class FusedDtGate(nn.Module):
    """One polynomial PER HEAD for the whole chain  x -> exp(A_h * softplus(x + b_h)).

    Replaces softplus AND exp with a single evaluation, so it costs one lot of
    depth instead of two, and it sidesteps the error amplification described in
    the module docstring (an error in Delta is multiplied by |A_h| when it
    reaches z; here the error is controlled directly in `a`-space, where it
    matters).

    `A_h` and `dt_bias_h` are weights, hence plaintext, hence free to fold in.

    The catch: for heads with very large |A_h| the fused function is close to a
    step in x, and no low-degree polynomial represents a step. For those heads
    the fit degenerates -- which may be exactly right if the head's observed x
    never sits near the step, and badly wrong if it does. That is why this has to
    be measured rather than assumed.
    """

    def __init__(self, coeffs, scales, intervals, trainable=False, name="fused_dt_gate"):
        super().__init__()
        c = torch.as_tensor(np.asarray(coeffs, dtype=np.float64), dtype=torch.float32)
        if c.ndim != 2:
            raise ValueError("coeffs must be (nheads, degree+1)")
        self.nheads, self.degree = int(c.shape[0]), int(c.shape[1] - 1)
        self.schedule = PowerSchedule.binary(self.degree)
        self.intervals = [tuple(map(float, iv)) for iv in intervals]
        self._name = name
        s = torch.as_tensor(np.asarray(scales, dtype=np.float64), dtype=torch.float32)
        self.register_buffer("inv_scales", 1.0 / s)
        self.register_buffer("scales", s)
        if trainable:
            self.coeffs = nn.Parameter(c)
        else:
            self.register_buffer("coeffs", c)

    @property
    def ct_ct_depth(self):
        return self.schedule.ct_ct_depth

    @property
    def name(self):
        return self._name

    def forward(self, x):
        """x: (..., nheads) -- the RAW dt projection, before softplus and before
        dt_bias. Returns `a` directly."""
        if x.shape[-1] != self.nheads:
            raise ValueError(f"expected last dim {self.nheads}, got {tuple(x.shape)}")
        c = self.coeffs.to(device=x.device, dtype=x.dtype)
        t = x * self.inv_scales.to(device=x.device, dtype=x.dtype)
        powers = {1: t}
        for target, lo, hi in self.schedule.steps:
            powers[target] = powers[lo] * powers[hi]
        out = torch.zeros_like(t) + c[:, 0]
        for k in range(1, self.degree + 1):
            out = out + c[:, k] * powers[k]
        return out

    @classmethod
    def fit(cls, A, dt_bias, intervals, degree=4, method="chebyshev", trainable=False):
        """A, dt_bias: (nheads,). intervals: per-head (lo, hi) of the RAW dt input."""
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(dt_bias, dtype=np.float64)
        rows, scales, ivs = [], [], []
        for h, (lo, hi) in enumerate(intervals):
            lo, hi = float(lo), float(hi)
            if hi <= lo:
                hi = lo + 1e-6
            s = max(abs(lo), abs(hi)) or 1.0

            def a_of_x(t, _h=h, _s=s):
                x = np.asarray(t, dtype=np.float64) * _s + b[_h]
                sp = np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))
                return np.exp(np.clip(A[_h] * sp, -700, 0))   # ALLOW-CLAMP: offline fit only, guards exp underflow

            rows.append(fit_general(a_of_x, degree, lo / s, hi / s, method=method))
            scales.append(s)
            ivs.append((lo, hi))
        return cls(np.stack(rows), scales, ivs, trainable=trainable,
                   name=f"fused_dt_gate{degree}")

    def to_dict(self):
        return {"name": self.name, "kind": "fused_softplus_exp_per_head",
                "degree": self.degree, "nheads": self.nheads,
                "ct_ct_depth": self.ct_ct_depth,
                "intervals": self.intervals,
                "scales": [float(v) for v in self.scales.detach().cpu()],
                "coeffs_lowest_first": self.coeffs.detach().cpu().tolist(),
                "trainable": isinstance(self.coeffs, nn.Parameter)}

    def __repr__(self):
        return (f"FusedDtGate(degree={self.degree}, nheads={self.nheads}, "
                f"ct_ct_depth={self.ct_ct_depth})")


# =============================================================================
# per-channel gates, fitted from measured statistics
# =============================================================================

class PerChannelPolyGate(nn.Module):
    """One polynomial PER CHANNEL of the last dimension.

    Same argument as the per-head exp gate: the coefficients are plaintext, so
    letting them differ per channel costs nothing under FHE -- they become one
    plaintext vector aligned to the slot layout, applied in a single
    ciphertext-times-plaintext multiply. Depth is unchanged.

    MEASURED on mamba2-130m (runs/gate_stats): going per-channel narrows the
    interval a SiLU has to cover by 26.5x after the conv1d and 6.6x in the gated
    norm, and a softplus by 2.9x. That is the difference between a usable fit and
    an unusable one -- one global interval gives degree-4 SiLU a max error of
    2.29, which is larger than most of the values it is approximating.
    """

    def __init__(self, coeffs, scales, intervals, trainable=False, name="perch_gate"):
        super().__init__()
        c = torch.as_tensor(np.asarray(coeffs, dtype=np.float64), dtype=torch.float32)
        if c.ndim != 2:
            raise ValueError("coeffs must be (nchannels, degree+1)")
        self.nchannels, self.degree = int(c.shape[0]), int(c.shape[1] - 1)
        self.schedule = PowerSchedule.binary(self.degree)
        self.intervals = [tuple(map(float, iv)) for iv in intervals]
        self._name = name
        s = torch.as_tensor(np.asarray(scales, dtype=np.float64), dtype=torch.float32)
        self.register_buffer("inv_scales", 1.0 / s)
        self.register_buffer("scales", s)
        if trainable:
            self.coeffs = nn.Parameter(c)
        else:
            self.register_buffer("coeffs", c)

    @property
    def ct_ct_depth(self):
        return self.schedule.ct_ct_depth

    @property
    def name(self):
        return self._name

    def forward(self, x):
        if x.shape[-1] != self.nchannels:
            raise ValueError(f"expected last dim {self.nchannels}, got {tuple(x.shape)}")
        c = self.coeffs.to(device=x.device, dtype=x.dtype)
        t = x * self.inv_scales.to(device=x.device, dtype=x.dtype)
        powers = {1: t}
        for target, lo, hi in self.schedule.steps:
            powers[target] = powers[lo] * powers[hi]
        out = torch.zeros_like(t) + c[:, 0]
        for k in range(1, self.degree + 1):
            out = out + c[:, k] * powers[k]
        return out

    @classmethod
    def fit(cls, fn, intervals, degree=4, method="chebyshev", margin=0.15,
            trainable=False, name="perch_gate"):
        rows, scales, ivs = [], [], []
        for lo, hi in intervals:
            lo, hi = float(lo), float(hi)
            # widen symmetrically: tokens slightly outside what we measured must
            # still land inside the interval, or the polynomial diverges there.
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * (1.0 + margin)
            lo, hi = mid - half, mid + half
            if hi - lo < 1e-6:
                lo, hi = mid - 1e-3, mid + 1e-3
            s = max(abs(lo), abs(hi)) or 1.0
            rows.append(fit_general(lambda t, _s=s: fn(np.asarray(t, dtype=np.float64) * _s),
                                    degree, lo / s, hi / s, method=method))
            scales.append(s)
            ivs.append((lo, hi))
        return cls(np.stack(rows), scales, ivs, trainable=trainable, name=name)

    def to_dict(self):
        return {"name": self.name, "kind": "per_channel", "degree": self.degree,
                "nchannels": self.nchannels, "ct_ct_depth": self.ct_ct_depth,
                "intervals": self.intervals,
                "scales": [float(v) for v in self.scales.detach().cpu()],
                "trainable": isinstance(self.coeffs, nn.Parameter)}

    def __repr__(self):
        return (f"PerChannelPolyGate({self._name}, degree={self.degree}, "
                f"nchannels={self.nchannels}, ct_ct_depth={self.ct_ct_depth})")


def _np_silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-np.clip(x, -700, 700)))   # ALLOW-CLAMP: offline fit only


def _np_softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


_GATE_FN = {"silu_conv_in": _np_silu, "silu_norm_in": _np_silu,
            "softplus_in": _np_softplus}


def build_per_channel_gates(stats_path, gate, degree=4, method="chebyshev",
                            margin=0.15, trainable=False):
    """{layer_idx: PerChannelPolyGate} from a collect_gate_stats.py JSON."""
    import json
    st = json.loads(open(stats_path).read())
    if gate not in st["per_channel"]:
        raise ValueError(f"{gate!r} not in {stats_path}; have "
                         f"{sorted(st['per_channel'])}")
    fn = _GATE_FN[gate]
    out = {}
    for layer, sd in st["per_channel"][gate].items():
        ivs = list(zip(sd["min"], sd["max"]))
        out[int(layer)] = PerChannelPolyGate.fit(
            fn, ivs, degree=degree, method=method, margin=margin,
            trainable=trainable, name=f"{gate}_poly{degree}_L{layer}")
    return out


# =============================================================================
# softplus, done properly: non-negative BY CONSTRUCTION
# =============================================================================

class SquaredPolySoftplus(nn.Module):
    """Delta = Q(x)^2, with Q fitted to sqrt(softplus(x)).

    WHY THIS SHAPE, MEASURED
    ------------------------
    A plain per-head degree-4 polynomial fit to softplus gives INFINITE
    perplexity, and the reason is not accuracy. On mamba2-130m,
    **415 of 576 heads produce a NEGATIVE Delta somewhere inside their own
    fitted interval** (median 11% of the interval). And Delta < 0 is not a small
    error, it is a sign error with a catastrophic consequence:

        Delta < 0  ->  z = A*Delta > 0  ->  a = exp(z) > 1  ->  the recurrence EXPANDS

    That is the exact failure mode Part 8 was built to catch. Exact softplus can
    never return a negative, so no amount of extra degree fixes a form that can.

    A square cannot be negative. So we approximate sqrt(softplus(x)) -- which is
    smooth, positive, and just as easy to fit -- and square the result. The
    invariant then holds for every input, including inputs we never measured and
    inputs outside the fitted interval. It is structural, not statistical.

    Cost: depth(Q) + 1 for the squaring. With Q of degree 2 (depth 1) the result
    is an effective degree-4 approximation at depth 2 -- the SAME depth as the
    broken direct degree-4 fit. The guarantee is free.
    """

    def __init__(self, coeffs, scales, intervals, trainable=False,
                 name="sq_poly_softplus"):
        super().__init__()
        c = torch.as_tensor(np.asarray(coeffs, dtype=np.float64), dtype=torch.float32)
        if c.ndim != 2:
            raise ValueError("coeffs must be (nheads, q_degree+1)")
        self.nheads, self.q_degree = int(c.shape[0]), int(c.shape[1] - 1)
        self.schedule = PowerSchedule.binary(self.q_degree)
        self.intervals = [tuple(map(float, iv)) for iv in intervals]
        self._name = name
        s = torch.as_tensor(np.asarray(scales, dtype=np.float64), dtype=torch.float32)
        self.register_buffer("inv_scales", 1.0 / s)
        self.register_buffer("scales", s)
        if trainable:
            self.coeffs = nn.Parameter(c)
        else:
            self.register_buffer("coeffs", c)

    @property
    def ct_ct_depth(self):
        """depth of Q, plus one for the final squaring."""
        return self.schedule.ct_ct_depth + 1

    @property
    def effective_degree(self):
        return 2 * self.q_degree

    @property
    def name(self):
        return self._name

    def forward(self, x):
        if x.shape[-1] != self.nheads:
            raise ValueError(f"expected last dim {self.nheads}, got {tuple(x.shape)}")
        c = self.coeffs.to(device=x.device, dtype=x.dtype)
        t = x * self.inv_scales.to(device=x.device, dtype=x.dtype)
        powers = {1: t}
        for target, lo, hi in self.schedule.steps:
            powers[target] = powers[lo] * powers[hi]
        q = torch.zeros_like(t) + c[:, 0]
        for k in range(1, self.q_degree + 1):
            q = q + c[:, k] * powers[k]
        return q * q                      # one more ct-ct mult; non-negative always

    @classmethod
    def fit(cls, intervals, q_degree=2, method="chebyshev", margin=0.15,
            trainable=False):
        def sqrt_softplus(x):
            x = np.asarray(x, dtype=np.float64)
            sp = np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))
            return np.sqrt(np.maximum(sp, 0.0))
        rows, scales, ivs = [], [], []
        for lo, hi in intervals:
            lo, hi = float(lo), float(hi)
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * (1.0 + margin)
            lo, hi = mid - half, mid + half
            if hi - lo < 1e-6:
                lo, hi = mid - 1e-3, mid + 1e-3
            s = max(abs(lo), abs(hi)) or 1.0
            rows.append(fit_general(lambda t, _s=s: sqrt_softplus(np.asarray(t) * _s),
                                    q_degree, lo / s, hi / s, method=method))
            scales.append(s)
            ivs.append((lo, hi))
        return cls(np.stack(rows), scales, ivs, trainable=trainable,
                   name=f"sq_poly_softplus_q{q_degree}")

    def to_dict(self):
        return {"name": self.name, "kind": "squared_polynomial_nonnegative",
                "q_degree": self.q_degree, "effective_degree": self.effective_degree,
                "nheads": self.nheads, "ct_ct_depth": self.ct_ct_depth,
                "intervals": self.intervals,
                "scales": [float(v) for v in self.scales.detach().cpu()],
                "coeffs_lowest_first": self.coeffs.detach().cpu().tolist(),
                "trainable": isinstance(self.coeffs, nn.Parameter)}

    def __repr__(self):
        return (f"SquaredPolySoftplus(q_degree={self.q_degree}, "
                f"effective_degree={self.effective_degree}, nheads={self.nheads}, "
                f"ct_ct_depth={self.ct_ct_depth})")


def build_squared_softplus(stats_path, q_degree=2, method="chebyshev", margin=0.15,
                           trainable=False):
    """{layer_idx: SquaredPolySoftplus} from a collect_gate_stats.py JSON."""
    import json
    st = json.loads(open(stats_path).read())
    out = {}
    for layer, sd in st["per_channel"]["softplus_in"].items():
        out[int(layer)] = SquaredPolySoftplus.fit(
            list(zip(sd["min"], sd["max"])), q_degree=q_degree, method=method,
            margin=margin, trainable=trainable)
    return out


def build_fused_dt_gates(stats_path, model, degree=4, method="chebyshev",
                         margin=0.15, trainable=False):
    """{layer_idx: FusedDtGate} from a collect_gate_stats.py JSON + the model.

    Needs the model because A_h and dt_bias_h are folded into the fitted
    function -- they are weights, so plaintext, so free to fold.
    """
    import json

    from real_mamba.model import iter_mixers
    st = json.loads(open(stats_path).read())
    per = st["per_channel"]["dt_raw"]
    out = {}
    for layer, mixer in iter_mixers(model):
        sd = per.get(str(layer))
        if sd is None:
            continue
        ivs = []
        for lo, hi in zip(sd["min"], sd["max"]):
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * (1.0 + margin)
            ivs.append((mid - half, mid + half))
        A = (-torch.exp(mixer.A_log.float())).detach().cpu().numpy()
        b = mixer.dt_bias.float().detach().cpu().numpy()
        out[layer] = FusedDtGate.fit(A, b, ivs, degree=degree, method=method,
                                     trainable=trainable)
    return out
