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


def rms_norm_gated_ref(x, weight, bias=None, z=None, eps=1e-5, group_size=None,
                       norm_before_gate=False, upcast=True):
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
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
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
