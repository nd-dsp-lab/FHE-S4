"""A pure-PyTorch Mamba-2 SSD written in PER-STEP PRODUCT form.

WHY THIS FILE HAS TO EXIST
--------------------------
Read ../PART0_SOURCE_INSPECTION.md section 0.2 first. Short version:

The real Mamba-2 kernel never computes `a_t = exp(A*dt_t)`. It computes
`cumsum(A*dt)` and exponentiates *differences of prefix sums*, because

    exp(z_{j+1} + ... + z_i) == exp(z_{j+1}) * ... * exp(z_i)

lets it replace a serial product with one cumulative sum. For a POLYNOMIAL that
identity is false:

    P(z_{j+1} + ... + z_i) != P(z_{j+1}) * ... * P(z_i)

So we cannot just swap `exp` for `P` inside the existing algebra -- that would
approximate the decay over a whole prefix with one polynomial call, which would
need `P` to be accurate on an interval that grows with sequence length. Useless.

This file therefore rewrites the same SSD using only:
  * per-step decay factors  a_t = transition(z_t),  and
  * PRODUCTS of them.

Every `cumsum` in `ssd_minimal.py` becomes a `cumprod` here, and every
`exp(segsum(...))` becomes a `segprod(...)`. When `transition = exp` the two are
algebraically identical, and `tests/test_reference_ssd.py` checks that
numerically against `ssd_minimal_discrete`'s formulation. When `transition = P`
they differ -- and the version here is the one that corresponds to what an FHE
implementation would actually have to compute.

WHAT IS *NOT* CHANGED
---------------------
softplus, SiLU, RMSNorm, the conv1d, the D skip, the chunk decomposition, the
einsum structure, and `A = -exp(A_log)` (a plaintext *parameter* exp). Only the
map `z -> a` is swappable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


# =============================================================================
# products in place of cumulative sums
# =============================================================================

def segprod(a: torch.Tensor) -> torch.Tensor:
    """M[..., i, j] = prod_{k=j+1}^{i} a_k   for i >= j,   0 for i < j.

    The product analogue of `segsum` in third_party/mamba/mamba_ssm/modules/
    ssd_minimal.py:23-32. Structurally identical: same masks, same axis, with
    `cumsum`+fill-0 replaced by `cumprod`+fill-1 and the final `-inf` fill (which
    exp maps to 0) replaced by a direct fill of 0.

    a: (..., T)   ->   (..., T, T)

    Numerics: for |a| < 1 this underflows toward 0, which IS the right answer
    (the influence of a distant token really is negligible). We never form a
    ratio of two products, which is what would be unstable. For |a| > 1 -- which
    only a polynomial can produce -- it grows, and we want to see that.
    """
    T = a.size(-1)
    dev = a.device
    x = repeat(a, "... d -> ... d e", e=T)
    keep = torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~keep, 1.0)            # neutral element for a product
    x = torch.cumprod(x, dim=-2)
    causal = torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool), diagonal=0)
    return x.masked_fill(~causal, 0.0)


def suffix_prod_exclusive(a: torch.Tensor) -> torch.Tensor:
    """r[..., i] = prod_{k=i+1}^{T-1} a_k, with r[..., T-1] = 1.

    The analogue of `exp(A_cumsum[..., -1:] - A_cumsum)` (ssd_minimal.py:59).
    Computed as a reversed cumprod rather than as a ratio of two products, so it
    stays well-behaved when the products underflow.
    """
    rc = torch.cumprod(a.flip(-1), dim=-1).flip(-1)      # rc[i] = prod_{k>=i} a_k
    ones = torch.ones_like(a[..., :1])
    return torch.cat([rc[..., 1:], ones], dim=-1)


# =============================================================================
# the scan
# =============================================================================

def ssd_product_form(x, dt, A, Bm, Cm, transition, D=None, chunk_size=64,
                     initial_states=None, return_a=False):
    """Mamba-2 SSD with a swappable per-step transition.

    Arguments (same names and shapes as `mamba_chunk_scan_combined`):
        x    : (batch, seqlen, nheads, headdim)     post-conv, post-SiLU
        dt   : (batch, seqlen, nheads)              post-softplus, > 0
        A    : (nheads,)                            < 0
        Bm   : (batch, seqlen, ngroups, d_state)
        Cm   : (batch, seqlen, ngroups, d_state)
        transition : callable z -> a, e.g. ExactExp() or PolyExp4(...)
        D    : (nheads,) or None                    the skip term
        chunk_size : block length Q; must divide seqlen after padding
        initial_states : (batch, nheads, headdim, d_state) or None

    Returns y : (batch, seqlen, nheads, headdim), and `a` if return_a.
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, d_state = Bm.shape
    assert nheads % ngroups == 0
    Q = chunk_size

    # -- pad the sequence up to a whole number of chunks ---------------------
    pad = (-seqlen) % Q
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))            # dt = 0 -> z = 0 -> a = P(0)
        Bm = F.pad(Bm, (0, 0, 0, 0, 0, pad))
        Cm = F.pad(Cm, (0, 0, 0, 0, 0, pad))
    L = seqlen + pad
    nc = L // Q

    # ---------------------------------------------------------------------
    # THE ONE LINE THIS PROJECT IS ABOUT
    # ---------------------------------------------------------------------
    z = dt * A                                     # (B, L, H), <= 0
    a = transition(z)                              # (B, L, H)
    # ---------------------------------------------------------------------

    # weight the input by dt, exactly as upstream does when it calls
    # ssd_minimal_discrete(x*dt.unsqueeze(-1), ...)   (ssd_minimal.py:103)
    X = x * dt.unsqueeze(-1)                       # (B, L, H, P)

    # broadcast the groups out to heads so every einsum below is head-wise
    if ngroups == 1:
        Bh = Bm.expand(batch, L, nheads, d_state)
        Ch = Cm.expand(batch, L, nheads, d_state)
    else:
        Bh = repeat(Bm, "b l g n -> b l (g r) n", r=nheads // ngroups)
        Ch = repeat(Cm, "b l g n -> b l (g r) n", r=nheads // ngroups)

    # -- split into chunks --------------------------------------------------
    Xc = rearrange(X, "b (c l) h p -> b c l h p", l=Q)
    Bc = rearrange(Bh, "b (c l) h n -> b c l h n", l=Q)
    Cc = rearrange(Ch, "b (c l) h n -> b c l h n", l=Q)
    ac = rearrange(a, "b (c l) h -> b h c l", l=Q)                 # (B, H, C, Q)

    # 1. intra-chunk (the block-diagonal part).
    #    Lmat[i, j] = prod_{k=j+1..i} a_k  -- the decay from token j to token i.
    Lmat = segprod(ac)                                             # (B, H, C, Q, Q)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cc, Bc, Lmat, Xc)

    # 2. each chunk's contribution to the state at its own right edge.
    decay_states = suffix_prod_exclusive(ac)                       # (B, H, C, Q)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_states, Xc)

    # 3. pass states between chunks. The chunk-level recurrence has decay
    #    tot[c] = prod of every a in chunk c.
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    else:
        initial_states = initial_states.unsqueeze(1)
    states = torch.cat([initial_states, states], dim=1)            # (B, C+1, H, P, N)
    tot = ac.prod(dim=-1)                                          # (B, H, C)
    tot = F.pad(tot, (1, 0), value=1.0)                            # neutral prefix
    decay_chunk = segprod(tot)                                     # (B, H, C+1, C+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. read the carried-in state out through C, decayed to each position.
    state_decay_out = torch.cumprod(ac, dim=-1)                    # (B, H, C, Q)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc, states, state_decay_out)

    y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    # 5. the D "skip" term uses the raw x, NOT x*dt
    #    (third_party/mamba/mamba_ssm/ops/triton/ssd_chunk_scan.py:1883-1886)
    if D is not None:
        y = y + x * (D.unsqueeze(-1) if D.dim() == 1 else D)

    if pad:
        y = y[:, :seqlen]
        a = a[:, :seqlen]
        z = z[:, :seqlen]
    if return_a:
        return y, z, a, final_state
    return y


def ssd_sequential(x, dt, A, Bm, Cm, transition, D=None, initial_states=None):
    """The same thing as a literal for-loop. Slow; used only to verify the above.

    This is `h_t = a_t * h_{t-1} + b_t` with nothing hidden, i.e. the shape of
    mamba2.py:317. If `ssd_product_form` and this disagree, the chunked algebra
    is wrong.
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, d_state = Bm.shape
    z = dt * A
    a = transition(z)
    if ngroups == 1:
        Bh = Bm.expand(batch, seqlen, nheads, d_state)
        Ch = Cm.expand(batch, seqlen, nheads, d_state)
    else:
        Bh = repeat(Bm, "b l g n -> b l (g r) n", r=nheads // ngroups)
        Ch = repeat(Cm, "b l g n -> b l (g r) n", r=nheads // ngroups)

    h = (torch.zeros(batch, nheads, headdim, d_state, dtype=x.dtype, device=x.device)
         if initial_states is None else initial_states.clone())
    ys = []
    for t in range(seqlen):
        b_t = torch.einsum("bh,bhn,bhp->bhpn", dt[:, t], Bh[:, t], x[:, t])
        h = a[:, t][:, :, None, None] * h + b_t
        y_t = torch.einsum("bhpn,bhn->bhp", h, Ch[:, t])
        if D is not None:
            y_t = y_t + (D.unsqueeze(-1) if D.dim() == 1 else D) * x[:, t]
        ys.append(y_t)
    return torch.stack(ys, dim=1)
