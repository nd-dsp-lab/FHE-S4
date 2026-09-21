"""Pure-PyTorch re-implementations of the two fused ops we cannot call on CPU.

These are transcriptions of upstream reference code, not new designs. They exist
because `mamba_ssm.ops.triton.layernorm_gated.RMSNorm.forward` dispatches to a
Triton kernel, which needs CUDA; we want the instrumentation path to run on a
laptop too.

Nothing here is part of the research question -- RMSNorm and SiLU are explicitly
out of scope. These are kept bit-faithful on purpose.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def causal_conv1d_preact_ref(xBC, conv1d, d_conv):
    """The depthwise causal conv1d WITHOUT its activation.

    Split out of `causal_depthwise_conv1d_ref` so the SiLU that follows can be
    swapped for a polynomial independently, and so its input distribution can be
    measured before anything is fitted.
    """
    y = conv1d(xBC.transpose(1, 2)).transpose(1, 2)
    return y[:, : -(d_conv - 1)] if d_conv > 1 else y


def rms_norm_gated_ref(x, weight, bias=None, z=None, eps=1e-5, group_size=None,
                       norm_before_gate=False, upcast=True,
                       silu_gate=None, gate_collector=None, layer_idx=-1):
    """Verbatim port of `rms_norm_ref`,
    third_party/mamba/mamba_ssm/ops/triton/layernorm_gated.py:18-39.

    norm_before_gate=False (what Mamba2 uses) means:  norm(x * silu(z)).
    """
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * (silu_gate(z) if silu_gate is not None else F.silu(z))
    # The inverse square root is the remaining hard one: it needs BOTH a
    # reciprocal and a square root, neither of which is a polynomial. Not
    # replaced yet -- but its argument is recorded here so we can see what range
    # an approximation would have to cover. That range is what decides whether
    # Newton/Goldschmidt is affordable.
    if group_size is None:
        ms = (x.square()).mean(dim=-1, keepdim=True) + eps
        if gate_collector is not None:
            gate_collector(layer_idx, "rmsnorm_meansq", ms)
        rstd = 1 / torch.sqrt(ms)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        ms = (x_group.square()).mean(dim=-1, keepdim=True) + eps
        if gate_collector is not None:
            gate_collector(layer_idx, "rmsnorm_meansq", ms)
        rstd = 1 / torch.sqrt(ms)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out = out * F.silu(z)
    return out.to(dtype)


def rms_norm_ref(x, weight, eps=1e-5):
    """Plain RMSNorm, as used for the block norms and `norm_f`."""
    dtype = x.dtype
    xf = x.float()
    rstd = 1 / torch.sqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * weight.float()).to(dtype)


def causal_depthwise_conv1d_ref(xBC, conv1d, d_conv, activation="silu"):
    """Depthwise causal conv1d + SiLU, matching the non-`causal_conv1d` branch at
    third_party/mamba/mamba_ssm/modules/mamba2.py:231-235.

    xBC: (batch, seqlen, conv_dim) -> same shape.
    """
    assert activation in ("silu", "swish")
    y = conv1d(xBC.transpose(1, 2)).transpose(1, 2)
    y = y[:, : -(d_conv - 1)] if d_conv > 1 else y     # drop the right padding
    return F.silu(y)
