"""PATH A -- replace RMSNorm with a learned CONSTANT divisor. Zero CKKS levels.

    y = x * gamma / sqrt(mean(x^2) + eps)        <- what Mamba-2 does
    y = x * gamma / c                            <- what we do instead

WHY THIS IS A REPLACEMENT, NOT AN APPROXIMATION
-----------------------------------------------
RMSNorm is *data-dependent*: the divisor is recomputed from the activations of
every token. A constant divisor is not a cheaper way of computing that number --
it is a different operator, which happens to agree with RMSNorm on tokens whose
mean-square lands near c^2 and disagrees everywhere else. Nothing is being
approximated; the function has changed, and the network has to be retrained to
live with it.

What the model has to learn to compensate: RMSNorm removes scale variation from
the residual stream before every mixer. Delete it and that variation survives
into `in_proj`, into `dt`, and therefore into `z = A*Delta` and the decay gate.
So the compensation is not local -- the surrounding projections have to learn to
tolerate an unnormalised input, and `dt_bias`/`A_log` have to re-centre the gate.
That is why this needs distillation rather than a coefficient fit.

WHY IT MATTERS MORE FOR FHE THAN FOR ORDINARY DEPLOYMENT
-------------------------------------------------------
On a GPU, RMSNorm is cheap: a reduction and a reciprocal square root, a few
microseconds. Under CKKS it needs BOTH a reciprocal and a square root, neither
of which is a polynomial, over an argument measured to span 6.1e6 on real text.
Prior estimates put a faithful version at 6-15 multiplicative levels per
instance -- and there are 49 instances in mamba2-130m (24 residual pre-norms,
24 gated norms, 1 final), making it the largest single line in the depth budget,
bigger than every other nonlinearity combined. A constant divisor costs **zero
levels**, because `gamma / c` is a single plaintext vector folded into the
existing weight.

WHAT THIS FILE DELIBERATELY DOES NOT DO
---------------------------------------
No clamp, no comparison, no table lookup, anywhere. Those are the primitives
CKKS does not give us cheaply, and reaching for one would be a finding, not a
fix. If a configuration needs one to be stable, that gets recorded rather than
implemented.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# the 20-bit precision proxy (a GUARDRAIL, not part of the operator)
# =============================================================================

def round_to_bits(x: torch.Tensor, bits: int = 20) -> torch.Tensor:
    """Round to `bits` significant bits, as a cheap stand-in for CKKS precision.

    Post-bootstrap CKKS precision is expected around 20 bits, well below fp32's
    24-bit significand, and our polynomial conditioning ceiling was measured in
    fp32. A method that survives fp32 and dies at 20 bits is a result, so every
    surviving configuration gets re-evaluated through this.

    Implemented by scaling each value to its own binade and rounding the
    mantissa, so it is relative precision (like CKKS), not absolute.
    """
    if bits <= 0 or bits >= 24:
        return x
    mant, exp = torch.frexp(x)
    scale = float(2 ** bits)
    return torch.ldexp(torch.round(mant * scale) / scale, exp)


# =============================================================================
# Path A operators
# =============================================================================

class ConstDivisorNorm(nn.Module):
    """Drop-in replacement for the plain RMSNorm modules (pre-norm and norm_f).

    `c` is stored as log_c so it stays positive under gradient descent without a
    clamp. At inference the whole operator is `x * (gamma / c)`, i.e. one
    plaintext vector multiply: **zero ciphertext levels**.
    """

    def __init__(self, weight: torch.Tensor, c_init: float, eps: float = 1e-5,
                 train_weight: bool = True, precision_bits: int | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad_(train_weight)
        # A FROZEN copy of the original gamma, so this one module can also serve
        # as the teacher. Teacher and student then share every other weight in
        # the network -- one model in memory, and the two provably differ in
        # nothing but the norm operator. Same trick as distill_poly_exp.py.
        self.register_buffer("weight_orig", weight.detach().clone())
        self.log_c = nn.Parameter(torch.tensor(math.log(max(c_init, 1e-8)),
                                               dtype=torch.float32))
        self.precision_bits = precision_bits
        self.exact = False          # flipped by norm.install_norms.exact_mode
        self.last_out = None        # set when recording, for the auxiliary loss
        self.record = False

    @property
    def c(self) -> torch.Tensor:
        return self.log_c.exp()

    @property
    def ct_ct_depth(self) -> int:
        return 0

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        if self.exact:
            rstd = 1.0 / torch.sqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
            out = xf * rstd * self.weight_orig.float()
        else:
            out = xf * (self.weight.float() / self.c)
            if self.precision_bits:
                out = round_to_bits(out, self.precision_bits)
        if self.record:
            self.last_out = out
        return out.to(dtype)

    def extra_repr(self):
        return f"c={float(self.c):.4g}, ct_ct_depth=0"


def const_gated_norm(x, weight, z, c, group_size=None, precision_bits=None):
    """Constant-divisor replacement for `rms_norm_gated_ref`.

    Keeps the EXACT SiLU on `z` on purpose: this experiment isolates the norm,
    so the polynomial SiLU-norm gate from GATES.md is deliberately not used here.
    `norm_before_gate=False`, matching Mamba-2, so the gating happens first and
    the (now constant) divisor is applied after.
    """
    dtype = x.dtype
    xf = x.float() * F.silu(z.float())
    out = xf * (weight.float() / c)
    if precision_bits:
        out = round_to_bits(out, precision_bits)
    return out.to(dtype)


class GatedConstDivisor(nn.Module):
    """Holds the learned `c` (and gamma) for one mixer's gated norm.

    The gated norm is invoked as a FUNCTION from the patched mixer forward, not
    as a module, so this object exists to own the parameters and be reachable
    from `model.parameters()`; the arithmetic happens in `const_gated_norm`.
    """

    def __init__(self, weight, c_init, group_size=None, train_weight=True,
                 precision_bits=None, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad_(train_weight)
        self.register_buffer("weight_orig", weight.detach().clone())
        self.log_c = nn.Parameter(torch.tensor(math.log(max(c_init, 1e-8)),
                                               dtype=torch.float32))
        self.group_size = group_size
        self.precision_bits = precision_bits
        self.eps = eps
        self.exact = False
        self.last_out = None
        self.record = False

    @property
    def c(self):
        return self.log_c.exp()

    @property
    def ct_ct_depth(self) -> int:
        return 0

    def forward(self, x, z):
        if self.exact:
            from real_mamba.nn_ref import rms_norm_gated_ref
            out = rms_norm_gated_ref(x, self.weight_orig, None, z=z, eps=self.eps,
                                     group_size=self.group_size,
                                     norm_before_gate=False)
        else:
            out = const_gated_norm(x, self.weight, z, self.c,
                                   group_size=self.group_size,
                                   precision_bits=self.precision_bits)
        if self.record:
            self.last_out = out
        return out

    def extra_repr(self):
        return f"c={float(self.c):.4g}, ct_ct_depth=0"
