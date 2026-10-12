"""Every non-polynomial operator of Mamba-2, as the polynomial an FHE circuit would run.

This is the PLAINTEXT emulator for end-to-end FHE: the same function the
encrypted circuit computes, in ordinary floating point, plus bootstrap-sized
noise wherever a refresh would happen. It answers "does the all-polynomial model
still work, and how much precision does it need?" before anyone writes CKKS code.

Design rules, all learned the hard way in this repo:

* CHEBYSHEV BASIS, CLENSHAW EVALUATION. Power-basis coefficients break down in
  fp32 above degree ~6 (see FINDINGS.md). Chebyshev on a per-channel interval
  mapped to [-1, 1] is stable at any degree, and it is also what OpenFHE's
  EvalChebyshevSeries evaluates, so the emulator matches the circuit.
* NO CLAMPS, NO COMPARISONS. A value outside the fitted interval is evaluated
  anyway, exactly as the circuit would. Off-interval inputs are COUNTED
  (`frac_out`) instead of hidden, because a polynomial outside its interval is
  the failure mode that produced every `inf` in this project.
* DEPTH FROM OPENFHE'S TABLE, NOT OUR GRAPH COUNT. Our old "degree 4 = depth 2"
  counted only ciphertext x ciphertext multiplies. In CKKS a plaintext multiply
  also consumes a level (rescale). OpenFHE documents EvalChebyshevFunction's
  depth by degree (src/pke/examples/FUNCTION_EVALUATION.md); `cheb_depth` is that
  table, so a degree 3-5 polynomial costs 4 levels, not 2.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# OpenFHE EvalChebyshevFunction: degree -> multiplicative depth
# (src/pke/examples/FUNCTION_EVALUATION.md, checked 2026-10-07).
# "If we use a range (a, b) = (-1, 1), the multiplicative depth is 1 less."
# ---------------------------------------------------------------------------
_CHEB_DEPTH = [(5, 4), (13, 5), (27, 6), (59, 7), (119, 8), (247, 9), (495, 10),
               (1007, 11), (2031, 12)]


def cheb_depth(degree: int, unit_interval: bool = False) -> int:
    if degree < 3:
        raise ValueError("OpenFHE's table starts at degree 3")
    for hi, d in _CHEB_DEPTH:
        if degree <= hi:
            return d - (1 if unit_interval else 0)
    raise ValueError(f"degree {degree} beyond the table")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _cheb_nodes(n: int) -> np.ndarray:
    return np.cos(np.pi * (np.arange(n) + 0.5) / n)            # roots of T_n, in (-1, 1)


def fit_cheb_interp(fn, lo: np.ndarray, hi: np.ndarray, degree: int) -> np.ndarray:
    """Chebyshev INTERPOLATION of fn on [lo_c, hi_c] for every channel c at once.

    Interpolating at the n = degree+1 roots of T_n is near-minimax and exact
    for the discrete orthogonality of T_k at those nodes:
        c_k = (2/n) * sum_j f(x_j) T_k(t_j),   c_0 halved.
    Returns (C, degree+1) float64 coefficients in the Chebyshev basis.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    n = degree + 1
    t = _cheb_nodes(n)                                           # (n,)
    x = lo[:, None] + (t[None, :] + 1.0) * 0.5 * (hi - lo)[:, None]   # (C, n)
    y = np.asarray(fn(x), dtype=np.float64)
    T = np.cos(np.outer(np.arange(n), np.arccos(t)))             # (n_k, n_j) = T_k(t_j)
    c = (2.0 / n) * y @ T.T                                      # (C, n_k)
    c[:, 0] *= 0.5
    return c


def fit_cheb_relative(fn, lo: float, hi: float, degree: int, grid: int = 4096) -> np.ndarray:
    """Weighted least squares in the Chebyshev basis, minimising RELATIVE error.

    For 1/sqrt(v) the absolute-error fit is useless at the small-v end, where the
    function is largest; what Newton needs is y0*sqrt(v) inside its basin, i.e. a
    bounded RELATIVE error. Weight = 1/|f|.
    """
    t = _cheb_nodes(grid)
    x = lo + (t + 1.0) * 0.5 * (hi - lo)
    y = np.asarray(fn(x), dtype=np.float64)
    w = 1.0 / np.maximum(np.abs(y), 1e-300)
    V = np.polynomial.chebyshev.chebvander(t, degree)           # (grid, degree+1)
    c, *_ = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)
    return c


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def clenshaw(c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """sum_k c[..., k] T_k(t), broadcasting c's leading dims against t's last dim.

    c: (C, n) or (n,);  t: (..., C) or (...).  Stable for |t| <= 1 at any degree.
    """
    n = c.shape[-1]
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    t2 = 2.0 * t
    for k in range(n - 1, 0, -1):
        b1, b2 = t2 * b1 - b2 + c[..., k], b1
    return t * b1 - b2 + c[..., 0]


class _Counted(nn.Module):
    """Accumulates how often inputs land outside the fitted interval."""

    def __init__(self):
        super().__init__()
        self.n_in = 0
        self.n_out = 0

    def count(self, t: torch.Tensor):
        with torch.no_grad():
            self.n_in += t.numel()
            self.n_out += int((t.abs() > 1.0).sum())

    @property
    def frac_out(self) -> float:
        return self.n_out / max(self.n_in, 1)

    def reset_counts(self):
        self.n_in = self.n_out = 0


class ChebPerChannel(_Counted):
    """y_c = sum_k coeff[c, k] T_k(t_c),  t_c = (2 x_c - (lo_c + hi_c)) / (hi_c - lo_c).

    `square=True` returns the square, for the Delta = Q(x)^2 softplus form that is
    non-negative by construction (GATES.md: 415/576 heads went negative without
    it). Squaring is one more level.
    """

    def __init__(self, coeffs: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 degree: int, name: str = "", square: bool = False):
        super().__init__()
        self.register_buffer("coeffs", torch.tensor(coeffs, dtype=torch.float32))
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        self.register_buffer("mid", torch.tensor((hi + lo) / 2.0, dtype=torch.float32))
        self.register_buffer("inv_half", torch.tensor(2.0 / (hi - lo), dtype=torch.float32))
        self.degree, self.name, self.square = degree, name, square

    @property
    def depth(self) -> int:
        return cheb_depth(self.degree) + (1 if self.square else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = (x.float() - self.mid) * self.inv_half
        self.count(t)
        y = clenshaw(self.coeffs, t)
        return (y * y if self.square else y).to(x.dtype)


class PolyInvSqrt(_Counted):
    """1/sqrt(v) for v in [lo, hi]: Chebyshev starting guess, then Newton steps.

        y0 = s * P(v)                          degree-`degree` Chebyshev, relative-error fit
        y  <- y * (1.5 - 0.5 * v * y^2)        repeated

    Newton's basin is y*sqrt(v) in (0, sqrt 3). Overshooting it diverges;
    undershooting only costs extra steps (each step multiplies a too-small guess
    by up to 1.5). So the guess is scaled DOWN, only when needed, until its worst
    point sits at 1.6 -- and the number of Newton steps is then chosen offline, on
    a dense log grid over [lo, hi], as the fewest that reach `target_rel`.

    The cost is set almost entirely by hi/lo (allpoly/README.md has the table):
    ~5-7 levels for a 10-100x range, ~25-30 for 1e5, ~45 for 1e7. That is why the
    calibration records v's range by token position as well as overall.
    """

    SAFE = 1.6   # < sqrt(3)

    def __init__(self, lo: float, hi: float, degree: int, newton: int | None = None,
                 target_rel: float = 2.0 ** -12, name: str = "", max_newton: int = 40):
        super().__init__()
        c = fit_cheb_relative(lambda v: 1.0 / np.sqrt(v), lo, hi, degree)
        vv = np.geomspace(lo, hi, 40001)
        tt = (2 * vv - (lo + hi)) / (hi - lo)
        r = np.polynomial.chebyshev.chebval(tt, c) * np.sqrt(vv)
        if r.min() <= 0:
            raise ValueError(f"{name}: starting guess goes non-positive on [{lo:g}, {hi:g}] "
                             f"at degree {degree}; Newton cannot recover from that")
        scale = min(1.0, self.SAFE / float(r.max()))
        c = c * scale
        r = r * scale
        self.init_rel_err_max = float(np.max(np.abs(r - 1.0)))
        self.init_in_basin = bool(np.all((r > 0) & (r < math.sqrt(3.0))))
        if newton is None:
            k = 0
            while np.abs(r - 1.0).max() > target_rel and k < max_newton:
                r = r * (1.5 - 0.5 * r * r)
                k += 1
            newton = k
        self.final_rel_err_on_grid = float(np.abs(r - 1.0).max()) if newton is not None else None
        self.register_buffer("coeffs", torch.tensor(c, dtype=torch.float32))
        self.lo, self.hi, self.degree, self.newton, self.name = float(lo), float(hi), degree, newton, name

    @property
    def depth(self) -> int:
        # each Newton step: y^2 (1), v*y^2 (1), y*(...) (1); 0.5 and 1.5 fold into
        # existing plaintext multiplies
        return cheb_depth(self.degree) + 3 * self.newton

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        t = (2.0 * v - (self.lo + self.hi)) / (self.hi - self.lo)
        self.count(t)
        y = clenshaw(self.coeffs, t)
        for _ in range(self.newton):
            y = y * (1.5 - 0.5 * v * y * y)
        return y


# ---------------------------------------------------------------------------
# Bootstrap noise
# ---------------------------------------------------------------------------
class Refresher:
    """Adds CKKS-bootstrap-sized noise at named refresh points.

    A bootstrap needs the message scaled into [-1, 1], and its error is relative to
    that scale. A real circuit scales PER SLOT before refreshing (one plaintext
    vector multiply), so each channel's error is relative to that channel's own
    range: per (site, layer, channel), S_c = the calibrated max |activation|, and
    the error is Gaussian with mean |error| = 2^-bits * S_c.

    Why per channel and not per tensor: with one scale for the whole tensor, noise
    on Delta is sized by the LARGEST head's Delta, then multiplied by A (|A| up to
    ~2.6e5 in some heads) inside the exp gate -- exp(+263) = inf even with exact
    operators. That is an artefact of the scaling choice, not of FHE, and the first
    version of this emulator had it. `per_tensor=True` keeps it for comparison.

    `bits` is the MEAN-abs precision measured in fhe/EPOCH.md (uniform secret at
    N = 2^17: 11.35; sparse: 18.23). Gaussian mean |e| = sigma * sqrt(2/pi).
    A refresh point with no calibrated scale raises: a silently skipped refresh
    would make the model look more robust than it is.
    """

    def __init__(self, bits: float | None, scales: dict, sites: tuple, seed: int = 0,
                 per_tensor: bool = False):
        self.bits, self.scales, self.sites = bits, scales, set(sites)
        self.per_tensor = per_tensor
        self.gen = None
        self.seed = seed
        self.calls = 0
        self._dev_scales = {}

    def __call__(self, layer: int, site: str, x: torch.Tensor) -> torch.Tensor:
        if self.bits is None or site not in self.sites:
            return x
        key = f"{site}:{layer}"
        if key not in self.scales:
            raise KeyError(f"no calibrated scale for refresh point {key}")
        if self.gen is None or self.gen.device != x.device:
            self.gen = torch.Generator(device=x.device).manual_seed(self.seed)
        S = self._dev_scales.get(key)
        if S is None:
            S = torch.as_tensor(self.scales[key], dtype=torch.float32).to(x.device)
            if self.per_tensor:
                S = S.max()
            self._dev_scales[key] = S
        sigma = (2.0 ** -self.bits) * math.sqrt(math.pi / 2.0) * S
        self.calls += 1
        noise = torch.randn(x.shape, generator=self.gen, device=x.device, dtype=torch.float32)
        return (x.float() + sigma * noise).to(x.dtype)
