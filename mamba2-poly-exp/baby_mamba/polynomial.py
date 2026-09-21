"""FHE-friendly polynomial stand-ins for exp(z), plus honest depth accounting.

TWO SEPARATE JOBS LIVE IN THIS FILE
-----------------------------------
1. FITTING (numpy, offline, plaintext): find coefficients c0..cn such that
   P(z) ~= exp(z) on an interval. Done once, on your laptop, in the clear.
2. EVALUATION (torch, online, would-be-encrypted): given z, compute P(z) using
   only additions and multiplications, arranged so that the *multiplicative
   depth* is minimal and visible.

Job 2 is the one that costs money under FHE, so it is written in a deliberately
un-clever way.

WHY NOT HORNER
--------------
Horner's rule is the standard way to evaluate a polynomial:

    P4(z) = c0 + z*(c1 + z*(c2 + z*(c3 + z*c4)))

It uses the fewest multiplications (4) -- but they are *sequential*, so the
multiplicative depth is 4. Under CKKS, depth is what forces you to pick bigger
parameters or to bootstrap; the raw multiplication count barely matters. So we
use the "baby-step" form instead:

    z2 = z * z        depth 1
    z3 = z2 * z       depth 2
    z4 = z2 * z2      depth 2
    P4 = c0 + c1*z + c2*z2 + c3*z3 + c4*z4      still depth 2

5 multiplications instead of 4, depth 2 instead of 4. That is the trade you want.

Multiplying by c_i is a ciphertext-times-PLAINTEXT multiply. Those are cheap and
consume (at most) one rescale, not a fresh level of ct-ct depth, so we do not
count them in `ct_ct_depth`. `ct_ct_depth` counts only ciphertext-times-
ciphertext multiplications on the critical path -- the expensive kind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Part 3: the evaluation graph and its depth
# =============================================================================

@dataclass
class PowerSchedule:
    """An explicit plan for building z^2 ... z^n out of ct-ct multiplications.

    Each entry is (target_power, left_power, right_power), meaning
    `z^target = z^left * z^right`. `z^1` is the input and has depth 0.

    We keep this as data rather than as code so the depth can be *computed and
    audited* instead of asserted in a comment.
    """

    degree: int
    steps: list[tuple[int, int, int]] = field(default_factory=list)

    @staticmethod
    def binary(degree: int) -> "PowerSchedule":
        """Minimal-depth schedule: always split a power as evenly as possible.

        For power k, build it from floor(k/2) and ceil(k/2). That gives
        depth(k) = ceil(log2(k)), which is optimal for a single power, and the
        whole schedule therefore has depth ceil(log2(degree)).
        """
        steps: list[tuple[int, int, int]] = []
        done = {1}

        def build(k: int) -> None:
            if k in done:
                return
            lo, hi = k // 2, k - k // 2
            build(lo)
            build(hi)
            steps.append((k, lo, hi))
            done.add(k)

        for k in range(2, degree + 1):
            build(k)
        steps.sort()
        return PowerSchedule(degree=degree, steps=steps)

    def depths(self) -> dict[int, int]:
        """Multiplicative depth of each power. z^1 is free (depth 0)."""
        d = {1: 0}
        for target, lo, hi in self.steps:
            d[target] = max(d[lo], d[hi]) + 1
        return d

    @property
    def ct_ct_depth(self) -> int:
        """Longest chain of ciphertext-ciphertext multiplications."""
        if self.degree < 2:
            return 0
        return max(self.depths().values())

    @property
    def ct_ct_mults(self) -> int:
        """How many ct-ct multiplications in total (not the same as depth)."""
        return len(self.steps)

    def explain(self) -> str:
        d = self.depths()
        lines = [f"# power schedule for degree {self.degree}", "z1 = z            # input, depth 0"]
        for target, lo, hi in self.steps:
            l = "z" if lo == 1 else f"z{lo}"
            r = "z" if hi == 1 else f"z{hi}"
            lines.append(f"z{target} = {l} * {r}".ljust(20) + f"# ct-ct, depth {d[target]}")
        lines.append(
            f"P  = c0 + c1*z + " + " + ".join(f"c{k}*z{k}" for k in range(2, self.degree + 1))
            + "   # ct-pt only, no extra ct-ct depth"
        )
        lines.append(f"=> ct-ct multiplications: {self.ct_ct_mults}, sequential ct-ct depth: {self.ct_ct_depth}")
        return "\n".join(lines)


class PolyExp(nn.Module):
    """P(z) ~= exp(z), evaluated as an explicit FHE-shaped computation graph.

    Coefficients are stored in the *power basis*, lowest first:
        P(z) = c[0] + c[1]*z + c[2]*z^2 + ...

    Set `trainable=True` to let fine-tuning move the coefficients (Part 9 MODE B).
    """

    def __init__(
        self,
        coeffs,
        interval: tuple[float, float] | None = None,
        trainable: bool = False,
        name: str | None = None,
    ):
        super().__init__()
        c = torch.as_tensor(np.asarray(coeffs, dtype=np.float64), dtype=torch.float32)
        if c.ndim != 1 or c.numel() < 1:
            raise ValueError("coeffs must be a 1-D sequence, lowest power first")
        self.degree = int(c.numel() - 1)
        self.schedule = PowerSchedule.binary(self.degree)
        self.interval = tuple(interval) if interval is not None else None
        self._name = name or f"poly{self.degree}"
        if trainable:
            self.coeffs = nn.Parameter(c)
        else:
            self.register_buffer("coeffs", c)

    # -- the property the whole project is trying to keep small ---------------
    @property
    def ct_ct_depth(self) -> int:
        return self.schedule.ct_ct_depth

    @property
    def ct_ct_mults(self) -> int:
        return self.schedule.ct_ct_mults

    @property
    def name(self) -> str:
        return self._name

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate P(z). Deliberately NOT Horner -- see the module docstring."""
        c = self.coeffs.to(z.dtype)

        # powers[k] holds z^k. powers[1] = z is the input ciphertext.
        powers: dict[int, torch.Tensor] = {1: z}
        for target, lo, hi in self.schedule.steps:
            # each of these lines is one ciphertext-ciphertext multiplication
            powers[target] = powers[lo] * powers[hi]

        # Now fold in the plaintext coefficients. These are ct-pt multiplies and
        # ct-ct-free additions, so they add no ct-ct depth.
        out = torch.full_like(z, 0.0) + c[0]
        for k in range(1, self.degree + 1):
            out = out + c[k] * powers[k]
        return out

    # -- bookkeeping ----------------------------------------------------------
    def coeff_list(self) -> list[float]:
        return [float(v) for v in self.coeffs.detach().cpu()]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "degree": self.degree,
            "coeffs_lowest_first": self.coeff_list(),
            "interval": list(self.interval) if self.interval else None,
            "ct_ct_depth": self.ct_ct_depth,
            "ct_ct_mults": self.ct_ct_mults,
            "trainable": isinstance(self.coeffs, nn.Parameter),
        }

    def __repr__(self) -> str:
        iv = f", interval={self.interval}" if self.interval else ""
        return (f"PolyExp(degree={self.degree}, ct_ct_depth={self.ct_ct_depth}{iv}, "
                f"coeffs={[round(v, 6) for v in self.coeff_list()]})")


# =============================================================================
# Part 2: fitting
# =============================================================================
# Four methods. All of them are "numerically sensible" in the sense you asked
# for -- none of them is a Taylor series. Taylor is deliberately absent: it is
# accurate only near its expansion point and its error grows like |z|^(n+1),
# which is exactly the wrong behaviour on an interval like [-8, 0].
#
#   chebyshev  Chebyshev interpolation at the roots of T_{n+1}. Near-minimax,
#              never fails, no iteration. THE DEFAULT.
#   lobatto    Chebyshev-Lobatto interpolation (nodes INCLUDE both endpoints).
#              Slightly worse max-error than `chebyshev`, but exact at z = xmax,
#              so with xmax = 0 you get P(0) = 1 for free. See `pin_zero` below.
#   remez      True minimax (equioscillating), via our own Remez exchange.
#              Best max-error. Raises rather than returning garbage.
#   lstsq      Weighted least squares. Best RMSE, and the only method that lets
#              you say "I care about relative error, not absolute".


def _pad_to_degree(coef: np.ndarray, degree: int) -> np.ndarray:
    """Pad a coefficient vector out to exactly degree+1 entries.

    numpy's `Chebyshev.convert(kind=Polynomial).coef` TRIMS trailing zeros. On a
    very wide interval the degree-4 fit to exp is numerically the zero polynomial,
    so it comes back with length 1 -- and a PolyExp built from it would then
    report degree 0 and depth 0, which would be a silently wrong FHE cost.
    Everything goes through here so the declared degree is always the real one.
    """
    coef = np.asarray(coef, dtype=np.float64).ravel()
    if coef.size > degree + 1:
        raise ValueError(f"got {coef.size} coefficients for degree {degree}")
    out = np.zeros(degree + 1, dtype=np.float64)
    out[: coef.size] = coef
    return out


def _chebyshev_roots(n: int, xmin: float, xmax: float) -> np.ndarray:
    """n Chebyshev points of the FIRST kind (roots of T_n). Excludes endpoints."""
    k = np.arange(n)
    t = np.cos(np.pi * (2 * k + 1) / (2 * n))
    return np.sort(0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * t)


def _chebyshev_lobatto(n: int, xmin: float, xmax: float) -> np.ndarray:
    """n Chebyshev points of the SECOND kind. INCLUDES both endpoints."""
    if n == 1:
        return np.array([0.5 * (xmin + xmax)])
    k = np.arange(n)
    t = np.cos(np.pi * k / (n - 1))
    return np.sort(0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * t)


def _interpolate_at(nodes: np.ndarray, fn) -> np.ndarray:
    """Polynomial through (nodes, fn(nodes)), returned in the power basis.

    Solved in the Chebyshev basis and converted, because a raw Vandermonde solve
    in the power basis is badly conditioned once the degree passes ~6.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    deg = len(nodes) - 1
    lo, hi = float(nodes.min()), float(nodes.max())
    if hi <= lo:
        raise ValueError("interval must be non-degenerate")
    # map nodes to [-1, 1] and build the Chebyshev Vandermonde
    t = (2.0 * nodes - (lo + hi)) / (hi - lo)
    V = np.polynomial.chebyshev.chebvander(t, deg)
    c_cheb = np.linalg.solve(V, fn(nodes))
    series = np.polynomial.chebyshev.Chebyshev(c_cheb, domain=[lo, hi])
    return series.convert(kind=np.polynomial.Polynomial).coef


def fit_chebyshev(degree: int, xmin: float, xmax: float) -> np.ndarray:
    """Chebyshev interpolation of exp on [xmin, xmax] (power-basis coefficients).

    Interpolating at Chebyshev roots is within a small factor
    (~(2/pi)*log(n) + 1) of the true minimax polynomial, needs no iteration and
    cannot fail to converge. Reliable by default.
    """
    return _interpolate_at(_chebyshev_roots(degree + 1, xmin, xmax), np.exp)


def fit_lobatto(degree: int, xmin: float, xmax: float) -> np.ndarray:
    """Chebyshev-Lobatto interpolation of exp. EXACT at both endpoints.

    With xmax = 0 this guarantees P(0) = exp(0) = 1 exactly, which matters a lot
    for a recurrent decay gate -- see `fit_exp_poly(..., pin_zero=True)`.
    """
    return _interpolate_at(_chebyshev_lobatto(degree + 1, xmin, xmax), np.exp)


def fit_least_squares(degree: int, xmin: float, xmax: float,
                      weight: str = "uniform", n_grid: int = 4001,
                      pin_zero: bool = False) -> np.ndarray:
    """Weighted least-squares fit of exp on a dense grid.

    weight="uniform"  -> minimise  sum (P(z) - exp(z))^2
    weight="relative" -> minimise  sum ((P(z) - exp(z)) / exp(z))^2

    Why "relative" is worth having: `a = exp(z)` is a decay factor that gets
    multiplied L times. An absolute error of 0.01 at z = -8 (where exp(z) =
    0.00034) is a 3000% relative error in the gate; the same 0.01 at z = 0 is a
    1% error. A uniform fit spends most of its accuracy budget where the function
    is tiny and the model barely cares. Whether that trade is worth making is an
    empirical question -- Part 4 and Part 7 answer it.

    pin_zero=True adds the hard linear constraint P(0) = 1. Because
    P(z) = 1 + z*Q(z), the constrained problem is still a plain linear least
    squares, just on the columns z^1..z^n with target (exp(z) - 1). Note this
    minimises the error in *P*, with the weights you asked for -- it is NOT the
    same as fitting Q to (exp(z)-1)/z under the same weights, which would
    implicitly reweight everything by |z|.
    """
    x = np.linspace(xmin, xmax, n_grid)
    y = np.exp(x)
    if weight == "uniform":
        w = np.ones_like(x)
    elif weight == "relative":
        w = 1.0 / np.maximum(y, 1e-12)
    else:
        raise ValueError(f"unknown weight {weight!r}")
    V = np.vander(x, degree + 1, increasing=True)
    if pin_zero:
        # c0 is fixed at 1; solve only for c1..cn against the reduced target.
        Vw = V[:, 1:] * w[:, None]
        rest, *_ = np.linalg.lstsq(Vw, (y - 1.0) * w, rcond=None)
        coef = np.concatenate(([1.0], rest))
    else:
        Vw = V * w[:, None]
        coef, *_ = np.linalg.lstsq(Vw, y * w, rcond=None)
    return coef


def _remez_generic(fn, degree: int, xmin: float, xmax: float,
                   iters: int = 100, tol: float = 1e-13) -> np.ndarray:
    """Remez exchange -> the true minimax polynomial for a smooth `fn`.

    No trustworthy general-purpose Remez exists in numpy/scipy
    (`scipy.signal.remez` approximates FIR filter responses, not arbitrary
    functions), so this is our own implementation. It raises on
    non-convergence, so it can never silently hand back a bad polynomial.
    """
    n = degree + 2                               # reference points
    x = _chebyshev_lobatto(n, xmin, xmax)
    g = np.linspace(xmin, xmax, 40001)
    fg = fn(g)
    coef = None
    for _ in range(iters):
        # Solve  P(x_i) + (-1)^i E = fn(x_i)  for the coefficients and level E.
        lo, hi = xmin, xmax
        t = (2.0 * x - (lo + hi)) / (hi - lo)
        M = np.zeros((n, n))
        M[:, :degree + 1] = np.polynomial.chebyshev.chebvander(t, degree)
        M[:, -1] = (-1.0) ** np.arange(n)
        try:
            sol = np.linalg.solve(M, fn(x))
        except np.linalg.LinAlgError:
            break
        c_cheb, E = sol[:degree + 1], sol[-1]
        coef = np.polynomial.chebyshev.Chebyshev(
            c_cheb, domain=[lo, hi]).convert(kind=np.polynomial.Polynomial).coef

        err = np.polyval(coef[::-1], g) - fg
        aerr = np.abs(err)
        emax = float(aerr.max())
        # converged when the worst error equals the levelled error
        if abs(emax - abs(E)) <= tol * max(1.0, emax):
            return coef
        # new reference = the n largest local extrema of |err| (endpoints count)
        interior = np.where((aerr[1:-1] >= aerr[:-2]) & (aerr[1:-1] >= aerr[2:]))[0] + 1
        cand = np.unique(np.concatenate(([0], interior, [len(g) - 1])))
        if len(cand) < n:
            break
        x = np.sort(g[cand[np.argsort(-aerr[cand])][:n]])
    if coef is not None:
        err = np.polyval(coef[::-1], g) - fg
        emax = float(np.abs(err).max())
        # accept a near-equioscillating answer, reject anything else
        if abs(emax - abs(E)) <= 1e-6 * max(1.0, emax):
            return coef
    raise RuntimeError(
        f"Remez did not converge for degree={degree} on [{xmin}, {xmax}]. "
        f"Use --method chebyshev (always works) or --method lobatto."
    )


def fit_remez(degree: int, xmin: float, xmax: float) -> np.ndarray:
    """True minimax polynomial approximation to exp on [xmin, xmax]."""
    return _remez_generic(np.exp, degree, xmin, xmax)


def fit_pinned_chebyshev(degree: int, xmin: float, xmax: float) -> np.ndarray:
    """P(z) = 1 + z*Q(z) with Q a Chebyshev interpolant of (exp(z)-1)/z.

    Exact at z = 0 by construction. Cheap, no iteration. The caveat: this
    minimises the error in Q, and the error in P is |z| times the error in Q, so
    the max error of P is worst near z = xmin. If you want P(0) = 1 *and* a good
    max error, prefer --method lobatto (with xmax = 0) or
    --method lstsq --pin-zero.
    """
    def g(z):
        z = np.asarray(z, dtype=np.float64)
        out = np.empty_like(z)
        small = np.abs(z) < 1e-8
        out[small] = 1.0 + z[small] / 2.0                 # series, avoids 0/0
        out[~small] = np.expm1(z[~small]) / z[~small]
        return out

    q = _interpolate_at(_chebyshev_roots(degree, xmin, xmax), g)
    coef = np.zeros(degree + 1, dtype=np.float64)
    coef[0] = 1.0
    coef[1:] = q
    return coef


def fit_general(fn, degree: int, xmin: float, xmax: float,
                method: str = "chebyshev", weight: str = "uniform",
                pin_at_xmax: bool = False) -> np.ndarray:
    """Fit an arbitrary smooth `fn` on [xmin, xmax]; power-basis, lowest first.

    Same four methods as `fit_exp_poly`. `pin_at_xmax` switches to Chebyshev-
    Lobatto nodes, which include both endpoints, so the result is EXACT at xmax
    (and at xmin). We use that to pin P(0) = 1 without a separate constrained
    solve -- it always works and never fails to converge.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if pin_at_xmax or method == "lobatto":
        c = _interpolate_at(_chebyshev_lobatto(degree + 1, xmin, xmax), fn)
    elif method == "chebyshev":
        c = _interpolate_at(_chebyshev_roots(degree + 1, xmin, xmax), fn)
    elif method == "remez":
        c = _remez_generic(fn, degree, xmin, xmax)
    elif method == "lstsq":
        x = np.linspace(xmin, xmax, 4001)
        y = np.asarray(fn(x), dtype=np.float64)
        w = np.ones_like(x) if weight == "uniform" else 1.0 / np.maximum(np.abs(y), 1e-12)
        V = np.vander(x, degree + 1, increasing=True) * w[:, None]
        c, *_ = np.linalg.lstsq(V, y * w, rcond=None)
    else:
        raise ValueError(f"method must be one of {FIT_METHODS}")
    return _pad_to_degree(c, degree)


FIT_METHODS = ("chebyshev", "lobatto", "remez", "lstsq")


def fit_exp_poly(degree: int, xmin: float, xmax: float,
                 method: str = "chebyshev", weight: str = "uniform",
                 pin_zero: bool = False) -> np.ndarray:
    """Fit a degree-`degree` polynomial to exp on [xmin, xmax].

    Returns power-basis coefficients, LOWEST POWER FIRST.

    pin_zero=True asks for P(0) == 1 exactly. WHY YOU PROBABLY WANT IT:
    exp(0) = 1 means "keep the whole previous state". An unconstrained degree-4
    Chebyshev fit on [-8, 0] gives P(0) ~= 0.9663. That looks like a harmless 3%
    error -- but `a` is multiplied once per timestep, so a token that should have
    been remembered perfectly is damped by 0.9663 per step:

        0.9663 ** 1024  ~=  1e-15

    i.e. long-range memory quietly disappears. Pinning P(0) = 1 spends one degree
    of freedom to remove that failure mode. It does NOT change the ct-ct depth.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if method not in FIT_METHODS:
        raise ValueError(f"method must be one of {FIT_METHODS}")

    if not pin_zero:
        if method == "chebyshev":
            c = fit_chebyshev(degree, xmin, xmax)
        elif method == "lobatto":
            c = fit_lobatto(degree, xmin, xmax)
        elif method == "remez":
            c = fit_remez(degree, xmin, xmax)
        else:
            c = fit_least_squares(degree, xmin, xmax, weight=weight)
        return _pad_to_degree(c, degree)

    # --- pin_zero == True ---------------------------------------------------
    if method == "lobatto":
        if abs(xmax) > 1e-12:
            raise ValueError(
                "--method lobatto --pin-zero requires xmax == 0 (0 must be a node). "
                "Either set --xmax 0, or use --method lstsq --pin-zero."
            )
        return _pad_to_degree(fit_lobatto(degree, xmin, xmax), degree)   # exact at xmax == 0
    if method == "chebyshev":
        return _pad_to_degree(fit_pinned_chebyshev(degree, xmin, xmax), degree)
    if method == "lstsq":
        return _pad_to_degree(
            fit_least_squares(degree, xmin, xmax, weight=weight, pin_zero=True), degree)
    raise ValueError(
        "--pin-zero is not supported with --method remez: minimising max|P - exp| "
        "subject to P(0) = 1 is a weighted minimax problem whose weight |z| "
        "vanishes at the constrained point, so plain Remez does not apply. "
        "Use --method lobatto (xmax=0) or --method lstsq --pin-zero."
    )


# =============================================================================
# Error reporting
# =============================================================================

def approximation_report(coeffs, xmin: float, xmax: float, n_grid: int = 20001) -> dict:
    """Errors of P against exp on [xmin, xmax], plus the properties that matter
    for a *recurrent decay gate* as opposed to a generic activation."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    x = np.linspace(xmin, xmax, n_grid)
    y = np.exp(x)
    p = np.polyval(coeffs[::-1], x)
    err = p - y
    rel = np.abs(err) / np.maximum(y, 1e-300)
    return {
        "degree": int(len(coeffs) - 1),
        "interval": [float(xmin), float(xmax)],
        "max_abs_error": float(np.abs(err).max()),
        "argmax_abs_error_z": float(x[np.argmax(np.abs(err))]),
        "mean_abs_error": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "max_rel_error": float(rel.max()),
        "mean_rel_error": float(rel.mean()),
        # gate-specific diagnostics
        "poly_min": float(p.min()),
        "poly_max": float(p.max()),
        "frac_poly_lt_0": float((p < 0).mean()),
        "frac_poly_gt_1": float((p > 1).mean()),
        "P_at_0": float(np.polyval(coeffs[::-1], 0.0)),
        "P_at_xmin": float(np.polyval(coeffs[::-1], xmin)),
    }


def save_coefficients(path, coeffs, xmin, xmax, method, weight,
                      pin_zero: bool = False, report=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    degree = int(len(coeffs) - 1)
    sched = PowerSchedule.binary(degree)
    payload = {
        "target_function": "exp",
        "degree": degree,
        "coeffs_lowest_first": [float(c) for c in coeffs],
        "interval": [float(xmin), float(xmax)],
        "fit_method": method,
        "fit_weight": weight,
        "pin_zero": bool(pin_zero),
        "ct_ct_depth": sched.ct_ct_depth,
        "ct_ct_mults": sched.ct_ct_mults,
        "power_schedule": [list(s) for s in sched.steps],
        "evaluation": "explicit powers (NOT Horner); see baby_mamba/polynomial.py",
    }
    if report is not None:
        payload["error_report"] = report
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_poly(path, trainable: bool = False) -> PolyExp:
    """Rebuild a PolyExp from a coefficients JSON written by fit_exp_polynomial.py."""
    d = json.loads(Path(path).read_text())
    return PolyExp(
        d["coeffs_lowest_first"],
        interval=tuple(d["interval"]),
        trainable=trainable,
        name=f"poly{d['degree']}",
    )


def build_poly(degree: int, xmin: float = -8.0, xmax: float = 0.0,
               method: str = "chebyshev", weight: str = "uniform",
               pin_zero: bool = False, trainable: bool = False) -> PolyExp:
    """Convenience: fit and wrap in one call. Used by tests and demos."""
    c = fit_exp_poly(degree, xmin, xmax, method=method, weight=weight, pin_zero=pin_zero)
    return PolyExp(c, interval=(xmin, xmax), trainable=trainable, name=f"poly{degree}")


# =============================================================================
# Per-head polynomials (needed once you look at the real checkpoint -- Part 6)
# =============================================================================

class PerHeadPolyExp(nn.Module):
    """One polynomial PER HEAD, all of the same degree and depth.

    WHY THIS EXISTS
    ---------------
    Part 6 measured the real pretrained Mamba2-130M and found that `A`, which is
    one scalar per head, spans FIVE ORDERS OF MAGNITUDE across the 576 heads:
    |A| from 4.0e-04 to 3.6e+04. Consequently `z = A*delta` per head ranges from
    [-0.004, 0] for the gentlest head to [-1.8e5, -0.47] for the most extreme.

    A single global interval cannot serve both. It must be wide enough for the
    extreme head, and a degree-4 polynomial on [-1.8e5, 0] is worthless
    (max error 1.0 -- it is just the zero function).

    A per-head interval costs NOTHING under FHE, because `A` is a *weight*:
    plaintext. Each head's coefficients are plaintext constants, so ciphertext
    x plaintext multiplies are all we add, and the ct-ct depth is unchanged.

    Measured effect of narrowing the interval, degree 4 Chebyshev:

        [-8, 0]  max|err| 3.4e-02   P(0)=0.966   17% of the interval negative
        [-4, 0]  max|err| 3.7e-03   P(0)=0.996   0%  of the interval negative
        [-2, 0]  max|err| 2.4e-04   P(0)=0.9998  0%
        [-1, 0]  max|err| 1.1e-05   P(0)=0.99999 0%

    91% of heads (523/576) have z_min >= -8, and 322/576 have z_min >= -1.

    coeffs: (nheads, degree+1), lowest power first, one row per head.
    """

    def __init__(self, coeffs, scales=None, intervals=None, trainable: bool = False,
                 name: str | None = None, degenerate_mask=None):
        super().__init__()
        c = torch.as_tensor(np.asarray(coeffs, dtype=np.float64), dtype=torch.float32)
        if c.ndim != 2:
            raise ValueError("coeffs must be (nheads, degree+1)")
        self.nheads, self.degree = int(c.shape[0]), int(c.shape[1] - 1)
        self.schedule = PowerSchedule.binary(self.degree)
        self.intervals = [tuple(map(float, iv)) for iv in intervals] if intervals is not None else None
        self._name = name or f"perhead_poly{self.degree}"
        # which heads got the exact constant-zero polynomial (depth 0, see Part 6)
        mask = (torch.as_tensor(degenerate_mask, dtype=torch.bool)
                if degenerate_mask is not None else torch.zeros(self.nheads, dtype=torch.bool))
        self.register_buffer("degenerate", mask)
        # SCALE NORMALISATION -- see the class docstring. s_h is a plaintext
        # constant, so z * (1/s_h) is a ciphertext-times-plaintext multiply: free,
        # and it adds no ct-ct depth.
        sc = (torch.ones(self.nheads) if scales is None
              else torch.as_tensor(np.asarray(scales, dtype=np.float64), dtype=torch.float32))
        if sc.shape != (self.nheads,):
            raise ValueError("scales must be (nheads,)")
        self.register_buffer("scales", sc)
        self.register_buffer("inv_scales", 1.0 / sc)
        if trainable:
            self.coeffs = nn.Parameter(c)
        else:
            self.register_buffer("coeffs", c)

    @property
    def ct_ct_depth(self) -> int:
        """Depth of the circuit that evaluates ALL heads together.

        If the degenerate heads are packed into their own ciphertext slots they
        cost depth 0, but a single SIMD-packed evaluation pays the max. We report
        the max, which is the conservative number.
        """
        return self.schedule.ct_ct_depth

    @property
    def ct_ct_mults(self) -> int:
        return self.schedule.ct_ct_mults

    @property
    def n_degenerate(self) -> int:
        return int(self.degenerate.sum())

    @property
    def name(self) -> str:
        return self._name

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., nheads) -> same shape.

        Identical power graph to PolyExp, so identical ct-ct depth. Two
        differences: the coefficients are per-head, and `z` is first divided by a
        per-head plaintext scale so that every head's polynomial is evaluated on
        a normalised variable in [-1, 0].
        """
        if z.shape[-1] != self.nheads:
            raise ValueError(f"expected last dim {self.nheads}, got {tuple(z.shape)}")
        c = self.coeffs.to(z.dtype)                       # (H, deg+1)
        z = z * self.inv_scales.to(z.dtype)               # ct x pt, no ct-ct depth
        powers: dict[int, torch.Tensor] = {1: z}
        for target, lo, hi in self.schedule.steps:
            powers[target] = powers[lo] * powers[hi]      # one ct-ct multiply each
        out = torch.zeros_like(z) + c[:, 0]
        for k in range(1, self.degree + 1):
            out = out + c[:, k] * powers[k]
        return out

    def coeffs_in_z_basis(self) -> list[list[float]]:
        """The same polynomials written in raw z, i.e. c_k / s^k.

        Useful for a write-up, dangerous for training: this is the basis whose
        coefficients span 1e-45 .. 1 across heads.
        """
        c = self.coeffs.detach().cpu().double()
        s = self.scales.detach().cpu().double()
        k = torch.arange(self.degree + 1, dtype=torch.float64)
        return (c / s[:, None] ** k[None, :]).tolist()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": "per_head_scale_normalised",
            "degree": self.degree,
            "nheads": self.nheads,
            "ct_ct_depth": self.ct_ct_depth,
            "ct_ct_mults": self.ct_ct_mults,
            "n_degenerate_heads": self.n_degenerate,
            "intervals": self.intervals,
            "scales": [float(v) for v in self.scales.detach().cpu()],
            "evaluation": "P(z) = sum_k c_k * (z/s)^k ; 1/s is a plaintext constant",
            "coeffs_normalised_lowest_first": self.coeffs.detach().cpu().tolist(),
            "coeffs_z_basis_lowest_first": self.coeffs_in_z_basis(),
            "trainable": isinstance(self.coeffs, nn.Parameter),
        }

    def __repr__(self) -> str:
        return (f"PerHeadPolyExp(degree={self.degree}, nheads={self.nheads}, "
                f"ct_ct_depth={self.ct_ct_depth}, degenerate={self.n_degenerate})")


# fp32 underflows exp(z) to exactly 0 below about -103.97 (log of the smallest
# subnormal). A head whose z never rises above that has a == 0 for every token it
# will ever see, so the EXACT answer is the constant-zero polynomial: degree 0,
# ct-ct depth 0, error 0. This is not an approximation or a shortcut.
FP32_EXP_UNDERFLOW = -103.9
# ...and the same threshold for bf16/fp16 safety margin when accumulating in fp32.


def fit_per_head(intervals, degree: int, method: str = "chebyshev",
                 weight: str = "uniform", pin_zero: bool = False,
                 zero_degenerate: bool = True,
                 underflow_z: float = FP32_EXP_UNDERFLOW,
                 trainable: bool = False, name: str | None = None,
                 normalise: bool = True) -> PerHeadPolyExp:
    """Fit one degree-`degree` polynomial per head, on that head's own interval,
    in a per-head SCALE-NORMALISED variable.

    intervals: sequence of (xmin, xmax) per head, xmin < xmax <= 0.

    WHY NORMALISE
    -------------
    Written in raw z, the per-head coefficients of the real checkpoint span
    1e-45 to 1 -- some of them are fp32 subnormals. Two consequences:
      * `z**4` at z = -1.8e5 is 1e21, which is fine in fp32 but becomes inf the
        moment anything grows;
      * no single learning rate can train coefficients whose scales differ by
        45 orders of magnitude. Part 10's first distillation run diverged in four
        steps for exactly this reason.
    So we substitute t = z / s_h with s_h = |xmin_h| and fit
    `t -> exp(s_h * t)` on t in [-1, xmax/s_h]. The polynomial is the same
    function; only the parameterisation changes. Every coefficient is then O(1)
    or exactly 0, and `1/s_h` is a plaintext constant, so under FHE it is a
    ciphertext-times-plaintext multiply: no extra ct-ct depth.

    zero_degenerate: heads whose xmax is below `underflow_z` get the EXACT
        constant-zero polynomial (see FP32_EXP_UNDERFLOW) -- not an approximation.
    pin_zero: applied only to heads whose interval actually reaches 0. Pinning
        P(0)=1 on a head that never evaluates z=0 would waste a coefficient.
        Implemented with Chebyshev-Lobatto nodes, which include the endpoints,
        so it is exact and cannot fail to converge.
    """
    rows, degen, ivs, scales = [], [], [], []
    for lo, hi in intervals:
        lo, hi = float(lo), float(hi)
        if not (lo < hi <= 0.0):
            raise ValueError(f"bad head interval ({lo}, {hi}); need xmin < xmax <= 0")
        ivs.append((lo, hi))
        s = abs(lo) if normalise else 1.0
        if s <= 0:
            raise ValueError(f"bad scale from interval ({lo}, {hi})")
        scales.append(s)
        if zero_degenerate and hi <= underflow_z:
            rows.append(np.zeros(degree + 1))
            degen.append(True)
            continue
        degen.append(False)
        tlo, thi = lo / s, hi / s                       # == (-1, hi/s) when normalised
        pin = bool(pin_zero and thi == 0.0)
        rows.append(fit_general(lambda t, _s=s: np.exp(_s * np.asarray(t, dtype=np.float64)),
                                degree, tlo, thi, method=method, weight=weight,
                                pin_at_xmax=pin))
    return PerHeadPolyExp(np.stack(rows), scales=scales, intervals=ivs, trainable=trainable,
                          name=name or f"perhead_poly{degree}", degenerate_mask=degen)
