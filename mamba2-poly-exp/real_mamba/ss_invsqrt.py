"""1/sqrt(v) exactly as fhe/seglut_invsqrt/bench.cpp computes it, simulated in PLAINTEXT.

SegLUT (real_mamba/seglut.py) is ROSETTA's table as the paper describes it: 2048
interpolated entries per segment, good to 20+ bits, with noise bolted on
afterwards. The OpenFHE benchmark showed that the table we can actually build is
much worse than that -- ~5.5 relative bits -- and that two Newton steps in CKKS
then bring it to ~15. This module reproduces that pipeline mechanism by
mechanism, so the paired perplexity test sees the operator we would deploy,
including the ways it breaks:

    1. segment s = sum_k [v >= tau_k]                     exact (0/289 wrong measured)
    2. table position x = G + (v - tau_s) / w_s           G guard positions per end
       read position p = round(x + e_idx)                 e_idx ~ N(0, sigma_idx):
                                                          the modulus switch to q
       entries f(knot_p) and f'(knot_p) * w_s, each stored as a phase in +-phase_frac*q
       (asin pre-distorted), read back with phase noise   e_ph ~ N(0, sigma_ph)
       and decoded through q sin(2 pi phase / q) / (2 pi), as FHEW->CKKS does
       NEGACYCLIC: p outside [0, q/2) reads entry p mod q/2 with its phase negated,
       which decodes to 2*mid - f: the value REFLECTED about the table's midpoint
    3. y0 = f(knot_p) + f'(knot_p) w_s (x - p)           the read offset is recovered
                                                          exactly (to 0.003 positions)
    4. y <- y (3 - v y^2) / 2, `newton` times             CKKS, 3 levels per step
    5. y (1 + e_ckks), e_ckks ~ N(0, 2^-ckks_bits)        CKKS's own noise floor

Both noise scales are sqrt(n / 18) for FHEW dimension n: 8.5 at STD128 (n = 1305),
1.3 at TOY (n = 32); the benchmark measured 6.3-9.0 (index) and 7.5-9.7 (phase).

Steps 1-4 alone reproduce the benchmark's table and first Newton step (STD128
5.7 / 10.1 rel bits against 5.5 / 9.2 measured; TOY 8.35 / 15.3 against 8.4 /
15.4) but not the second: the benchmark stops at 15.2 (TOY 22.6) where the
noiseless iteration would give 17.6 (28). That ceiling is CKKS, so step 5 adds it;
`ckks_bits` = 15.3 puts the STD128 two-step result back on 15.2.

What a plaintext SegLUT cannot show and this can: an input past the guard band
reads a reflected entry (12x too large at v = 2 tau_S), the Newton steps then
diverge from it, and the output is garbage rather than slightly off. `n_wrap`
counts those reads, `n_wrap_lo` the ones below the table.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _encode(vals, q, phase_frac):
    """Table values -> integer phases, bench.cpp MakeTable. Returns (phases, mid, halfw, Dmax)."""
    lo, hi = vals.min(), vals.max()
    mid, halfw = 0.5 * (hi + lo), max(0.5 * (hi - lo), 1e-300)
    dmax = q * math.sin(2 * math.pi * phase_frac) / (2 * math.pi)
    D = (vals - mid) / halfw * dmax
    ph = np.rint(q * np.arcsin(2 * math.pi * D / q) / (2 * math.pi))
    ph[ph == 0] = 1          # OpenFHE recognises a negacyclic table only if no entry is 0
    return ph, mid, halfw, dmax


class SchemeSwitchInvSqrt(nn.Module):
    """One 1/sqrt table set (n_seg segments) for one norm instance; input v > 0, any shape."""

    ct_ct_depth = 0     # the table itself; Newton adds 3 levels per step in CKKS

    def __init__(self, tau, q=4096, guard=0.05, phase_frac=0.2, index_noise=8.5,
                 phase_noise=8.5, newton=2, ckks_bits=15.3, seed=0, name=None):
        super().__init__()
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        if tau.size < 2 or np.any(np.diff(tau) <= 0) or tau[0] <= 0:
            raise ValueError("tau must be strictly increasing and positive")
        self.n_seg, self.q, self.half = tau.size - 1, int(q), int(q) // 2
        self.G = math.floor(guard * self.half)
        self.index_noise, self.phase_noise = float(index_noise), float(phase_noise)
        self.newton, self.seed = int(newton), seed
        self.ckks_bits = ckks_bits
        self._name = name or "ss_invsqrt"
        w = (tau[1:] - tau[:-1]) / (self.half - 2 * self.G)

        fph, sph, dec = [], [], []
        x = np.arange(self.half, dtype=np.float64)
        for s in range(self.n_seg):
            # bench.cpp: the low guard would reach v <= 0 (a segment spans ~60x), so knots
            # stop at 0.8 tau_s; index noise moves a few positions, never that far
            vx = np.maximum(tau[s] + (x - self.G) * w[s], 0.8 * tau[s])
            pf, mf, hf, dmax = _encode(vx ** -0.5, self.q, phase_frac)
            ps, ms, hs, _ = _encode(-0.5 * vx ** -1.5 * w[s], self.q, phase_frac)
            fph.append(pf); sph.append(ps); dec.append((mf, hf, ms, hs))
        self.dmax = dmax
        f64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
        self.register_buffer("tau", f64(tau))
        self.register_buffer("w", f64(w))
        self.register_buffer("f_phase", f64(fph))          # (S, q/2)
        self.register_buffer("s_phase", f64(sph))
        self.register_buffer("dec", f64(dec))              # (S, 4): f mid, halfw, slope mid, halfw
        self._gen = None
        self.track = False
        self.reset_counters()

    @property
    def name(self):
        return self._name

    def reset_counters(self):
        self.n_seen = self.n_oob = self.n_neg = self.n_gt1 = self.n_wrap = self.n_wrap_lo = 0

    def reseed(self, seed=None):
        self.seed = self.seed if seed is None else seed
        self._gen = None

    def _randn(self, like, std):
        if self._gen is None or self._gen.device != like.device:
            self._gen = torch.Generator(device=like.device)
            self._gen.manual_seed(int(self.seed))
        return torch.randn(like.shape, generator=self._gen, device=like.device,
                           dtype=like.dtype) * std

    def _read(self, phases, p, seg, mid, halfw):
        """Entry at read position p, negacyclic (phase negated past q/2), noisy, decoded."""
        q, half = self.q, self.half
        m = torch.remainder(p, q)
        neg = m >= half
        idx = torch.where(neg, m - half, m).long()
        ph = phases[seg, idx]
        ph = torch.where(neg, -ph, ph)
        if self.phase_noise:
            ph = ph + torch.round(self._randn(ph, self.phase_noise))
        D = q * torch.sin(2 * math.pi * ph / q) / (2 * math.pi)
        return mid + halfw * D / self.dmax

    def forward(self, v, newton=None):
        dtype = v.dtype
        vd = v.double()
        tau = self.tau
        seg = torch.zeros_like(vd, dtype=torch.long)
        for k in range(1, self.n_seg):
            seg = seg + (vd >= tau[k]).long()
        x = self.G + (vd - tau[seg]) / self.w[seg]
        e = self._randn(x, self.index_noise) if self.index_noise else 0.0
        p = torch.round(x + e)
        d = self.dec[seg]                                  # (..., 4)
        fv = self._read(self.f_phase, p, seg, d[..., 0], d[..., 1])
        sv = self._read(self.s_phase, p, seg, d[..., 2], d[..., 3])
        y = fv + sv * (x - p)
        for _ in range(self.newton if newton is None else newton):
            y = y * (3.0 - vd * y * y) / 2.0
        if self.ckks_bits:
            y = y * (1.0 + self._randn(y, 2.0 ** -float(self.ckks_bits)))
        if self.track:
            self.n_seen += vd.numel()
            self.n_oob += int(((vd < tau[0]) | (vd > tau[-1])).sum())
            self.n_wrap += int(((p < 0) | (p >= self.half)).sum())
            self.n_wrap_lo += int((p < 0).sum())       # v below the table: small v
            self.n_neg += int((y <= 0).sum())
        return y.to(dtype)

    @torch.no_grad()
    def precision(self, per_segment=512, seed=0, newton=None):
        """Same contract as SegLUT.precision, plus `rel_bits` (bench.cpp's headline).

        Uniform samples inside each segment, as bench.cpp draws them. Noise stays ON:
        here the noise is the table, there is no noiseless version to report.
        """
        saved_gen, saved_track = self._gen, self.track
        self._gen, self.track = None, False
        old_seed, self.seed = self.seed, seed
        try:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            u = torch.rand(per_segment, self.n_seg, generator=gen, dtype=torch.float64)
            t = self.tau.cpu()
            xs = (t[:-1] + u * (t[1:] - t[:-1])).reshape(-1).to(self.tau.device)
            y = self.forward(xs, newton=newton)
        finally:
            self._gen, self.track, self.seed = saved_gen, saved_track, old_seed
        exact = xs.rsqrt()
        err, rel = (y - exact).abs(), ((y - exact) / exact)
        return {"bits": np.array([float(-torch.log2(err.square().mean().sqrt()))]),
                "rel_bits": np.array([float(-torch.log2(rel.square().mean().sqrt()))]),
                "max_abs": np.array([float(err.max())]),
                "max_rel": np.array([float(rel.abs().max())])}

    def __repr__(self):
        return (f"SchemeSwitchInvSqrt(n_seg={self.n_seg}, q={self.q}, guard={self.G}, "
                f"index_noise={self.index_noise}, phase_noise={self.phase_noise}, "
                f"newton={self.newton}, ckks_bits={self.ckks_bits})")


__all__ = ["SchemeSwitchInvSqrt"]
