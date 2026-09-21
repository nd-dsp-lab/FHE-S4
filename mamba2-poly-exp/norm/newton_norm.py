"""PATH B -- prescaled Newton inverse square root. A few CKKS levels.

    v'   = v / s_layer                       s_layer a PLAINTEXT constant (Phase 1)
    y_0  = constant initial guess            no comparison, no table -- we have neither
    y_{t+1} = y_t * (1.5 - 0.5 * v' * y_t^2)
    1/sqrt(v) = y_T / sqrt(s_layer)

This one really is an approximation of RMSNorm, unlike Path A. The catch is that
Newton's method for the inverse square root converges only from a guess already
near the answer, and we cannot branch on the input to pick one. So the whole
method rests on the PRESCALE: divide by a plaintext per-layer constant so that
v' lands near 1, where a single fixed y_0 works for every token.

Phase 1 measured v spanning 6.1e6 across the corpus. The prescale removes the
between-layer part of that spread for free; the within-layer part is what the
range penalty in the trainer is meant to shrink.

DEPTH IS COUNTED, NOT ASSERTED
------------------------------
`newton_depth()` walks the expression graph and returns the real
ciphertext-ciphertext depth, the same discipline as
baby_mamba.polynomial.PowerSchedule. It does not agree with the brief's estimate
of "2 levels per Newton step" -- see the docstring there for why the first step
is free and later steps cost 3.

NO CLAMP, NO COMPARISON, NO TABLE. If a configuration needs one to converge,
that is a finding to record, not a thing to implement.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def newton_depth(t_steps: int, verbose: bool = False) -> dict:
    """Real ct-ct depth of the prescaled Newton iteration, derived from the graph.

    Depth bookkeeping, where d(.) is ct-ct depth and plaintext operands are free:

        x            d=0   (the norm's input)
        x^2          d=1   ct-ct
        v = mean(x^2)+eps  d=1   the reduction is rotations+adds, no multiply
        v' = v/s     d=1   ct x PLAINTEXT, free
        y_0          d=0   a plaintext constant

      step 1:  y_0^2 is PLAINTEXT, so v'*y_0^2 is ct x pt (free) and
               y_0*(1.5-0.5*v'*y_0^2) is pt x ct (free)      -> y_1 at d=1
      step t>1: y_t^2            d(y_t)+1
                v' * y_t^2       max(1, d(y_t)+1) + 1
                y_t * (...)      +1                          -> 3 levels per step

      finally  x * gamma/sqrt(s) * y_T   -> one more ct-ct               +1

    So the FIRST Newton step is free and each subsequent one costs 3. That is
    the opposite shape to the brief's "2 per step", and it argues for spending
    effort on the prescale (which buys accuracy at zero depth) rather than on
    extra iterations.
    """
    steps = []
    d_v = 1                       # x^2 then a free reduction
    d_y = 0                       # constant guess
    for t in range(1, t_steps + 1):
        if d_y == 0:
            note = "y^2 is plaintext, whole step is ct x pt"
            d_new = d_v
        else:
            d_sq = d_y + 1
            d_prod = max(d_v, d_sq) + 1
            d_new = d_prod + 1
            note = f"y^2 at {d_sq}, v'*y^2 at {d_prod}, y*(..) at {d_new}"
        steps.append({"step": t, "depth_before": d_y, "depth_after": d_new,
                      "levels_added": d_new - d_y, "note": note})
        d_y = d_new
    total = d_y + 1               # the final multiply into x*gamma
    if verbose:
        for s in steps:
            print(f"    newton step {s['step']}: +{s['levels_added']} "
                  f"-> depth {s['depth_after']}   ({s['note']})")
        print(f"    final scaling multiply: +1  -> TOTAL ct-ct depth {total}")
    return {"t_steps": t_steps, "per_step": steps, "total_ct_ct_depth": total,
            "depth_of_v": d_v, "depth_of_y": d_y}


class NewtonInvSqrtNorm(nn.Module):
    """RMSNorm with the inverse square root computed by prescaled Newton.

    `log_s` (the prescale) is learnable but is a PLAINTEXT constant at inference,
    so it costs nothing. `y0` likewise. Only the iteration costs levels.
    """

    def __init__(self, weight, s_init: float, t_steps: int = 2, eps: float = 1e-5,
                 y0: float | None = None, train_weight: bool = True,
                 precision_bits: int | None = None):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad_(train_weight)
        self.register_buffer("weight_orig", weight.detach().clone())
        self.log_s = nn.Parameter(torch.tensor(math.log(max(s_init, 1e-12)),
                                               dtype=torch.float32))
        # y0 = 1 is the right default once v' is near 1, since 1/sqrt(1) = 1.
        self.log_y0 = nn.Parameter(torch.tensor(math.log(y0 if y0 else 1.0),
                                                dtype=torch.float32))
        self.t_steps = t_steps
        self.eps = eps
        self.precision_bits = precision_bits
        self.exact = False
        self.record = False
        self.last_out = None
        self.last_vprime = None          # for the range penalty and reporting

    @property
    def s(self):
        return self.log_s.exp()

    @property
    def ct_ct_depth(self) -> int:
        return newton_depth(self.t_steps)["total_ct_ct_depth"]

    def forward(self, x):
        from norm.const_norm import round_to_bits
        dtype = x.dtype
        xf = x.float()
        if self.exact:
            rstd = 1.0 / torch.sqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
            out = xf * rstd * self.weight_orig.float()
        else:
            v = xf.square().mean(dim=-1, keepdim=True) + self.eps
            vp = v / self.s                                  # ct x pt, free
            self.last_vprime = vp if self.record else None
            y = self.log_y0.exp().expand_as(vp)
            for _ in range(self.t_steps):
                y = y * (1.5 - 0.5 * vp * y * y)
            rstd = y / torch.sqrt(self.s)
            out = xf * rstd * self.weight.float()
            if self.precision_bits:
                out = round_to_bits(out, self.precision_bits)
        if self.record:
            self.last_out = out
        return out.to(dtype)

    def extra_repr(self):
        return (f"t_steps={self.t_steps}, s={float(self.s.detach()):.4g}, "
                f"ct_ct_depth={self.ct_ct_depth}")


def range_penalty(modules, kind: str = "log2") -> torch.Tensor:
    """lambda * mean(log(v')^2) -- push the prescaled argument toward 1.

    This is the mechanism that makes a STATIC prescale viable against a
    data-dependent argument: it asks the network to reshape its own activations
    so that v/s stays near 1, which is where a fixed y_0 converges. Without it
    the prescale only removes the between-layer spread, not the within-layer
    spread, and Newton from a constant guess diverges on the tails.
    """
    terms = []
    for m in modules:
        vp = getattr(m, "last_vprime", None)
        if vp is None:
            continue
        lg = torch.log(vp.float().clamp_min(1e-12))   # ALLOW-CLAMP: guards log() in the TRAINING LOSS only, never in the FHE operator
        terms.append((lg ** 2).mean())
    if not terms:
        return None
    return torch.stack(terms).mean()
