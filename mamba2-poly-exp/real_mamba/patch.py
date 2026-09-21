"""Part 5 -- patch ONLY the Mamba-2 selective transition, nothing else.

WHAT THIS DOES
--------------
Replaces `Mamba2.forward` with a pure-PyTorch forward that recomputes the mixer
**from that layer's own pretrained parameters**, routing the state scan through
`real_mamba/reference_ssd.py` so that `a_t = transition(A * dt_t)` is an explicit,
swappable line.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
* It does NOT monkey-patch `torch.exp`. Every other `exp` in the model keeps
  working, including `A = -exp(A_log)` -- which is an exp of a *weight*, i.e.
  plaintext under FHE, i.e. free, i.e. not our problem.
* It does NOT touch SiLU, RMSNorm, softplus, the conv1d, the D skip, the
  residual stream, the embedding or the LM head.
* It does NOT change any parameter value. With `--transition exact` the patched
  model must be numerically equivalent to the unpatched one; that is asserted in
  `real_mamba/tests/test_parity.py`, not assumed.

The cost of the patch is speed: we give up the fused Triton kernel. That is the
price of being able to see `z` and `a` at all (see PART0_SOURCE_INSPECTION.md
section 0.4 -- `use_mem_eff_path=False` is NOT enough; the "unfused" path is also
Triton, and even softplus lives inside the kernel).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange

from baby_mamba.transition import ExactExp
from real_mamba.nn_ref import causal_depthwise_conv1d_ref, rms_norm_gated_ref
from real_mamba.reference_ssd import ssd_product_form


# =============================================================================
# the reference mixer forward
# =============================================================================

def mamba2_reference_forward(mixer, u, transition, chunk_size=None,
                             collector=None, seq_idx=None, **unused):
    """One Mamba-2 mixer, in plain PyTorch, with a swappable z -> a map.

    Mirrors the `use_mem_eff_path=False` branch of
    third_party/mamba/mamba_ssm/modules/mamba2.py:209-261, step for step.

    `collector`, if given, receives (layer_idx, z, a) for Parts 6 and 8.
    """
    batch, seqlen, _ = u.shape

    # --- 1. the single packed input projection      (mamba2.py:181) ----------
    zxbcdt = mixer.in_proj(u)

    # --- 2. A = -exp(A_log), in fp32                (mamba2.py:182) ----------
    # NOTE: this exp is applied to a PARAMETER. Under FHE the weights are
    # plaintext, so it is free. We leave it exactly as upstream has it.
    A = -torch.exp(mixer.A_log.float())                            # (nheads,)

    # --- 3. split [z0, x0, z, xBC, dt]              (mamba2.py:210-214) ------
    d_ssm = getattr(mixer, "d_ssm", mixer.d_inner)
    nheads, ngroups, d_state = mixer.nheads, mixer.ngroups, mixer.d_state
    d_mlp = (zxbcdt.shape[-1] - 2 * d_ssm - 2 * ngroups * d_state - nheads) // 2
    z0, x0, gate, xBC, dt = torch.split(
        zxbcdt, [d_mlp, d_mlp, d_ssm, d_ssm + 2 * ngroups * d_state, nheads], dim=-1)

    # --- 4. depthwise causal conv1d + SiLU          (mamba2.py:231-235) ------
    # Untouched by this project. We only re-express it in pure torch because
    # `causal_conv1d_fn` is a CUDA extension.
    xBC = causal_depthwise_conv1d_ref(xBC, mixer.conv1d, mixer.d_conv,
                                      getattr(mixer, "activation", "silu"))
    x, Bm, Cm = torch.split(xBC, [d_ssm, ngroups * d_state, ngroups * d_state], dim=-1)

    # --- 5. dt = softplus(dt + dt_bias)             (mamba2.py:254 dt_softplus) -
    # Upstream does this INSIDE the Triton kernel (ssd_chunk_state.py:73-77), so
    # it is invisible from Python on the default path. Here it is explicit.
    # softplus is NOT part of what we replace.
    dt = F.softplus(dt.float() + mixer.dt_bias.float())            # (B, L, nheads)
    dt_limit = tuple(getattr(mixer, "dt_limit", (0.0, float("inf"))))
    if dt_limit != (0.0, float("inf")):
        # Upstream clamps dt when dt_limit is set (ssd_chunk_state.py:80). The
        # pretrained 130M checkpoint uses the default, so this branch never runs
        # for it -- but if you load a checkpoint that does set dt_limit, we must
        # reproduce it or we are not evaluating the same model.
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])             # ALLOW-CLAMP: fidelity to upstream, not a stability crutch

    # --- 6. the scan, with the swappable transition --------------------------
    D = mixer.D.float()
    if getattr(mixer, "D_has_hdim", False):
        D = rearrange(D, "(h p) -> h p", p=mixer.headdim)
    y, z_vals, a_vals, _ = ssd_product_form(
        rearrange(x.float(), "b l (h p) -> b l h p", p=mixer.headdim),
        dt,
        A,
        rearrange(Bm.float(), "b l (g n) -> b l g n", g=ngroups),
        rearrange(Cm.float(), "b l (g n) -> b l g n", g=ngroups),
        transition,
        D=D,
        chunk_size=chunk_size or mixer.chunk_size,
        return_a=True,
    )
    if collector is not None:
        collector(getattr(mixer, "layer_idx", -1), z_vals, a_vals)

    y = rearrange(y, "b l h p -> b l (h p)").to(u.dtype)

    # --- 7. gated RMSNorm, then out_proj            (mamba2.py:258-267) ------
    if getattr(mixer, "rmsnorm", True):
        y = rms_norm_gated_ref(y, mixer.norm.weight, getattr(mixer.norm, "bias", None),
                               z=gate, eps=mixer.norm.eps,
                               group_size=getattr(mixer.norm, "group_size", None),
                               norm_before_gate=getattr(mixer, "norm_before_gate", False))
    else:
        y = y * F.silu(gate)
    if d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)
    return mixer.out_proj(y)


# =============================================================================
# the patch itself
# =============================================================================

@dataclass
class PatchHandle:
    """What `patch_transition` installed, so it can be undone."""
    mixers: list = field(default_factory=list)
    originals: dict = field(default_factory=dict)
    transition: object = None
    n_patched: int = 0

    model: object = None

    def restore(self):
        if self.model is not None and hasattr(self.model, "fhe_transitions"):
            del self.model.fhe_transitions
        for mixer in self.mixers:
            orig = self.originals.get(id(mixer))
            if orig is None:
                with contextlib.suppress(AttributeError):
                    del mixer.forward
            else:
                mixer.forward = orig
            mixer._transition = None
        self.mixers, self.originals, self.n_patched = [], {}, 0


def patch_transition(model, transition=None, chunk_size=None, collector=None,
                     layers=None, verbose=False) -> PatchHandle:
    """Install the reference forward on every Mamba-2 mixer (or a subset).

    transition: either one module shared by every layer, or a
                {layer_idx: module} dict (needed for per-head coefficients,
                which differ per layer -- see real_mamba/transitions.py).
    layers:     an iterable of layer indices, or None for all of them. Patching a
                subset is useful for bisection ("does layer 0 alone explain the
                damage?").

    The transition modules are registered on the model as `model.fhe_transitions`
    so that (a) `model.parameters()` sees trainable polynomial coefficients for
    Part 9 MODE B, and (b) `model.to(device)` moves their coefficient buffers.
    """
    import torch.nn as nn

    from real_mamba.model import iter_mixers

    transition = transition if transition is not None else ExactExp()
    handle = PatchHandle(transition=transition)
    wanted = None if layers is None else set(layers)

    # register so parameters/buffers follow the model around
    as_dict = transition if isinstance(transition, dict) else None
    holder = nn.ModuleDict(
        {str(k): v for k, v in as_dict.items()} if as_dict is not None
        else ({"shared": transition} if isinstance(transition, nn.Module) else {})
    )
    ref = next(model.parameters())
    model.fhe_transitions = holder.to(device=ref.device)
    handle.model = model

    for idx, mixer in iter_mixers(model):
        if wanted is not None and idx not in wanted:
            continue
        mine = as_dict[idx] if as_dict is not None else transition
        handle.originals[id(mixer)] = mixer.__dict__.get("forward")
        handle.mixers.append(mixer)
        mixer._transition = mine
        mixer.layer_idx = getattr(mixer, "layer_idx", idx)

        def bound(u, _m=mixer, inference_params=None, **kw):
            if inference_params is not None:
                raise NotImplementedError(
                    "The reference transition path does not implement cached decoding. "
                    "Evaluate with full forward passes (eval_lm.py), not generate()."
                )
            return mamba2_reference_forward(_m, u, _m._transition,
                                            chunk_size=chunk_size, collector=collector, **kw)

        mixer.forward = bound
        handle.n_patched += 1

    if handle.n_patched == 0:
        raise RuntimeError("patch_transition found no Mamba-2 mixers to patch")
    if verbose:
        shown = (f"per-layer, e.g. L0 = {as_dict[min(as_dict)]}" if as_dict is not None
                 else transition)
        print(f"[patch] reference transition installed on {handle.n_patched} mixer(s); "
              f"transition = {shown}")
    return handle


@contextlib.contextmanager
def patched(model, transition=None, **kw):
    """`with patched(model, PolyExp4(...)): ...`"""
    handle = patch_transition(model, transition, **kw)
    try:
        yield handle
    finally:
        handle.restore()


def set_transition(handle: PatchHandle, transition):
    """Swap the transition in place without re-patching.

    Accepts a single module or a {layer_idx: module} dict, same as
    `patch_transition`.
    """
    import torch.nn as nn

    handle.transition = transition
    as_dict = transition if isinstance(transition, dict) else None
    for mixer in handle.mixers:
        mixer._transition = as_dict[mixer.layer_idx] if as_dict is not None else transition
    if handle.model is not None:
        holder = nn.ModuleDict(
            {str(k): v for k, v in as_dict.items()} if as_dict is not None
            else ({"shared": transition} if isinstance(transition, nn.Module) else {})
        )
        ref = next(handle.model.parameters())
        handle.model.fhe_transitions = holder.to(device=ref.device)
