"""The Mamba-2 selective state transition, written out in plain PyTorch.

WHY THIS FILE EXISTS
--------------------
In the real implementation (see ../PART0_SOURCE_INSPECTION.md) the transition is
buried inside a Triton kernel, and it is not even computed one timestep at a
time -- the kernel computes cumulative sums and exponentiates *those*. That is
great for speed and terrible for understanding.

This file computes the same thing the slow, obvious way, so that every
intermediate quantity has a name you can print.

THE FIVE QUANTITIES YOU CARE ABOUT
----------------------------------
    u        the input to the layer                       (batch, seqlen, d_model)
    delta    an input-dependent timestep, always > 0      (batch, seqlen, nheads)
    A        a learned decay rate, always < 0             (nheads,)
    z        = A * delta, always <= 0                     (batch, seqlen, nheads)
    a        = exp(z), always in (0, 1]                   (batch, seqlen, nheads)

and then the recurrence, per head h:

    h_t = a_t * h_{t-1} + b_t

`a_t` is the only thing standing between "remember everything" (a = 1) and
"forget everything" (a = 0). It is computed from the input, so under FHE it is
computed from *encrypted* numbers, so it is itself encrypted.

FAITHFULNESS TO THE REAL MODEL
------------------------------
The shapes and the order of operations here mirror `Mamba2.step()` at
third_party/mamba/mamba_ssm/modules/mamba2.py:307-320, which is the one place
upstream that writes the transition in exactly this form. Specifically:

    A  = -exp(A_log)                      mamba2.py:307     one scalar per head
    dt = softplus(dt_raw + dt_bias)       mamba2.py:313
    dA = exp(dt * A)                      mamba2.py:314     <-- our target
    h  = h * dA[:, :, None, None] + dBx   mamba2.py:317
    y  = einsum("bhpn,bn->bhp", h, C)     mamba2.py:318
    y  = y + D[:, None] * x               mamba2.py:319

We do NOT touch softplus, SiLU or RMSNorm. They are out of scope on purpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BabyConfig:
    """Tiny but structurally real Mamba-2 dimensions.

    The names are upstream's names. The values are chosen so that you can print
    every tensor without scrolling.
    """

    batch: int = 2
    seqlen: int = 8
    d_model: int = 4            # upstream 130M: 768
    expand: int = 2             # upstream: 2
    headdim: int = 2            # upstream: 64
    d_state: int = 2            # upstream: 128
    ngroups: int = 1            # upstream: 1

    # dt_bias is initialised from a log-uniform sample in [dt_min, dt_max],
    # same as mamba2.py:118-124.
    dt_min: float = 0.001
    dt_max: float = 0.1
    # A is initialised uniform in this range and then negated, same as
    # mamba2.py:132-135 (A_init_range).
    A_init_range: tuple[float, float] = (1.0, 16.0)

    # PEDAGOGICAL KNOB, NOT PART OF THE REAL MODEL.
    # Real Mamba-2 has no such factor. With freshly-initialised weights, dt_bias
    # is small, so z = A*delta sits close to 0 and every polynomial looks fine.
    # Setting delta_scale > 1 emulates a model whose delta has grown during
    # training, pushing z further negative, so you can watch the polynomial fail.
    # Always report the value you used.
    delta_scale: float = 1.0

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def nheads(self) -> int:
        assert self.d_inner % self.headdim == 0
        return self.d_inner // self.headdim

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(d_inner=self.d_inner, nheads=self.nheads)
        return d


class BabyMamba2Transition(nn.Module):
    """One Mamba-2 mixer, minus every optimisation, plus a swappable exp.

    The `transition` argument is a callable `z -> a`. Pass `ExactExp()` to get
    the real model; pass `PolyExp4(...)` to get the FHE-friendly one. That single
    swap is the entire research question of this project.
    """

    def __init__(self, cfg: BabyConfig, transition: nn.Module | None = None, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)

        # --- the projection that produces everything at once -------------------
        # Upstream packs [z, x, B, C, dt] into ONE matmul (mamba2.py:96). We keep
        # that layout so the slicing code looks like the real thing.
        d_in_proj = 2 * cfg.d_inner + 2 * cfg.ngroups * cfg.d_state + cfg.nheads
        self.in_proj = nn.Linear(cfg.d_model, d_in_proj, bias=False)
        with torch.no_grad():
            self.in_proj.weight.copy_(torch.randn(d_in_proj, cfg.d_model, generator=g) * 0.5)

        # --- dt_bias: exactly upstream's initialisation ------------------------
        # Sample dt log-uniformly in [dt_min, dt_max], then store its
        # softplus-inverse, so that softplus(dt_bias) == that sample.
        dt = torch.exp(
            torch.rand(cfg.nheads, generator=g) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))       # softplus^-1, mamba2.py:126
        self.dt_bias = nn.Parameter(inv_dt)

        # --- A, stored in log form so that A is negative by construction -------
        A = torch.empty(cfg.nheads).uniform_(*cfg.A_init_range, generator=g)
        self.A_log = nn.Parameter(torch.log(A))          # mamba2.py:134

        # --- D: the "skip" / residual-within-the-SSM term ----------------------
        self.D = nn.Parameter(torch.ones(cfg.nheads))

        self.transition = transition if transition is not None else ExactExp()

    # ------------------------------------------------------------------ helpers
    def A(self) -> torch.Tensor:
        """A = -exp(A_log), shape (nheads,). Always strictly negative.

        NOTE: this `exp` is applied to a *parameter*, not to data. Under FHE the
        weights are plaintext, so this exp is free. It is NOT the exp we are
        replacing -- do not get confused by seeing `exp` here.
        """
        return -torch.exp(self.A_log.float())

    # ------------------------------------------------------------------ forward
    def forward(self, u: torch.Tensor, return_all: bool = True) -> dict:
        """Run the layer and hand back every intermediate.

        u: (batch, seqlen, d_model)
        Returns a dict; see the keys at the bottom of this function.
        """
        cfg = self.cfg
        batch, seqlen, _ = u.shape

        # 1. One matmul, then slice it the way upstream does (mamba2.py:211-214).
        #    We drop the `z0/x0` MLP slices because d_intermediate == 0 in the
        #    130M checkpoint, so they are empty there too.
        zxbcdt = self.in_proj(u)
        gate_z, x, B, C, dt_raw = torch.split(
            zxbcdt,
            [cfg.d_inner, cfg.d_inner, cfg.ngroups * cfg.d_state,
             cfg.ngroups * cfg.d_state, cfg.nheads],
            dim=-1,
        )
        # NOTE: upstream runs a depthwise causal conv1d + SiLU over [x, B, C]
        # before this point. We skip the conv here -- it changes *what* x/B/C
        # are, not *how* the transition works, and skipping it keeps this file
        # readable. The real-model path (../real_mamba/) does include it.

        # 2. delta: strictly positive timestep, one per (token, head).
        #    softplus, not exp -- and we are NOT replacing softplus in this project.
        delta = F.softplus(dt_raw + self.dt_bias)                  # (B, L, H)
        if cfg.delta_scale != 1.0:
            delta = delta * cfg.delta_scale        # exploration knob, see BabyConfig

        # 3. A: one negative scalar per head.
        A = self.A()                                               # (H,)

        # 4. z = A * delta.  Negative times positive => z <= 0, always.
        #    This is the input to the nonlinearity we want to make cheap.
        z = delta * A                                              # (B, L, H)

        # 5. a = exp(z) -- or a polynomial pretending to be exp(z).
        #    THIS IS THE ONE LINE THE WHOLE PROJECT IS ABOUT.
        a = self.transition(z)                                     # (B, L, H)

        # 6. Reshape x into heads: (B, L, d_inner) -> (B, L, H, P)
        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        # B and C are shared across heads within a group (ngroups=1 => all heads).
        B = B.reshape(batch, seqlen, cfg.ngroups, cfg.d_state)
        C = C.reshape(batch, seqlen, cfg.ngroups, cfg.d_state)
        # Broadcast the single group out to every head, so the einsums below are
        # plain and head-wise. (Upstream lets the kernel handle the grouping.)
        Bh = B.expand(batch, seqlen, cfg.nheads, cfg.d_state) if cfg.ngroups == 1 \
            else B.repeat_interleave(cfg.nheads // cfg.ngroups, dim=2)
        Ch = C.expand(batch, seqlen, cfg.nheads, cfg.d_state) if cfg.ngroups == 1 \
            else C.repeat_interleave(cfg.nheads // cfg.ngroups, dim=2)

        # 7. The input term b_t. Note delta multiplies BOTH the decay (via z) and
        #    the input -- that is what "selective" means: one scalar decides both
        #    how much to forget and how much to write.
        #    b_t[h, p, n] = delta[h] * B[h, n] * x[h, p]     (mamba2.py:316)
        b = torch.einsum("blh,blhn,blhp->blhpn", delta, Bh, x)     # (B, L, H, P, N)

        # 8. The recurrence, one timestep at a time, no cleverness.
        h = torch.zeros(batch, cfg.nheads, cfg.headdim, cfg.d_state, dtype=b.dtype)
        states, ys = [], []
        for t in range(seqlen):
            # a_t is a scalar per (batch, head). It multiplies the ENTIRE state
            # of that head -- all headdim*d_state numbers -- hence the [:, :, None, None].
            h = a[:, t][:, :, None, None] * h + b[:, t]            # mamba2.py:317
            states.append(h)
            # read the state out through C
            y_t = torch.einsum("bhpn,bhn->bhp", h, Ch[:, t])       # mamba2.py:318
            y_t = y_t + self.D[None, :, None] * x[:, t]            # mamba2.py:319
            ys.append(y_t)

        H = torch.stack(states, dim=1)                             # (B, L, H, P, N)
        y = torch.stack(ys, dim=1)                                 # (B, L, H, P)

        out = {"z": z, "a": a, "h": H, "y": y}
        if return_all:
            out.update(u=u, delta=delta, A=A, b=b, x=x, B=B, C=C, gate_z=gate_z)
        return out


class ExactExp(nn.Module):
    """a = exp(z). The real thing. Expensive under FHE."""

    degree = None
    ct_ct_depth = None          # not a polynomial; depth is not defined
    name = "exact"

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(z)

    def __repr__(self) -> str:
        return "ExactExp()"


def reference_recurrence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_{t-1} + b_t, spelled out. Used by the tests.

    a: (batch, seqlen, nheads)
    b: (batch, seqlen, nheads, headdim, d_state)
    returns h: (batch, seqlen, nheads, headdim, d_state)
    """
    batch, seqlen, nheads = a.shape
    h = torch.zeros(b.shape[0], *b.shape[2:], dtype=b.dtype)
    out = []
    for t in range(seqlen):
        h = a[:, t][:, :, None, None] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)
