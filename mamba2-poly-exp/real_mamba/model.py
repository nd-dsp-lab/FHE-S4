"""Load the pretrained state-spaces/mamba2-130m checkpoint.

TWO BACKENDS, ONE CHECKPOINT
----------------------------
  official  `MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m")` from the
            installed `mamba-ssm` package. This is the path the brief asks for and
            the one to use on the GPU box. It needs CUDA + triton (+ optionally
            causal_conv1d) because `mamba_ssm` imports Triton kernels at import
            time.
  local     The same official `pytorch_model.bin`, loaded into a minimal module
            tree defined below with byte-identical parameter names. Runs on CPU,
            MPS or CUDA with no Triton. This is a FALLBACK so the project is
            runnable on a laptop -- not a reimplementation of the research.

`backend="auto"` tries official and silently falls back to local.

Whichever backend you use, the forward pass we actually study is
`real_mamba/patch.py`, which recomputes the mixer from the layer's own
parameters. So the two backends differ only in who constructs the nn.Modules.
`real_mamba/tests/test_parity.py` checks they agree.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from real_mamba.nn_ref import rms_norm_ref

DEFAULT_MODEL = "state-spaces/mamba2-130m"


def recommended_dtype(device: str = "cuda", verbose: bool = True):
    """The best dtype this GPU can actually run, not the one it claims to support.

    MEASURED on a Quadro RTX 6000 (sm_75) at CRC, 2026-09-19:

        mamba_chunk_scan_combined      float16  OK
                                       float32  IndexError: map::at  (Triton
                                                cannot compile the kernel)
                                       bfloat16 IndexError: map::at

    and `torch.cuda.is_bf16_supported()` returns **True** on that card anyway --
    it reports driver-level support, not tensor-core support. Turing has fp16
    tensor cores and no bf16 ones, so trusting that flag picks a dtype the
    official backend cannot compile and the hardware would emulate slowly.

    Rule: bf16 only from Ampere (sm_80) up; fp16 on anything older; fp32 on CPU.
    The `local` backend is pure PyTorch and runs fp32 happily on any GPU -- this
    only constrains `--backend official`.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return torch.float32
    major = torch.cuda.get_device_properties(0).major
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    if verbose and major < 8:
        name = torch.cuda.get_device_properties(0).name
        print(f"[dtype] {name} is sm_{major}x: using float16, not bfloat16 "
              f"(no bf16 tensor cores; the official Triton SSD kernel will not "
              f"compile in bf16 or fp32 on this card)")
    return dtype


# =============================================================================
# config
# =============================================================================

@dataclass
class LiteConfig:
    """The subset of `MambaConfig` that a Mamba2-only checkpoint needs."""
    d_model: int = 768
    n_layer: int = 24
    d_intermediate: int = 0
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16
    rms_norm: bool = True
    residual_in_fp32: bool = True
    norm_epsilon: float = 1e-5
    # Mamba2 defaults, third_party/mamba/mamba_ssm/modules/mamba2.py:38-61
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 256
    dt_limit: tuple[float, float] = (0.0, float("inf"))
    ssm_cfg: dict = field(default_factory=dict)

    @property
    def padded_vocab_size(self) -> int:
        v = self.vocab_size
        if v % self.pad_vocab_size_multiple != 0:
            v += self.pad_vocab_size_multiple - v % self.pad_vocab_size_multiple
        return v

    @classmethod
    def from_hf(cls, name: str = DEFAULT_MODEL) -> "LiteConfig":
        from huggingface_hub import hf_hub_download
        raw = json.loads(open(hf_hub_download(name, "config.json")).read())
        ssm = dict(raw.get("ssm_cfg") or {})
        layer = ssm.pop("layer", "Mamba1")
        if layer != "Mamba2":
            raise ValueError(
                f"{name} is a {layer} checkpoint. This project is Mamba-2 only "
                f"(the transition it studies does not exist in the same form in Mamba-1/3)."
            )
        known = {f for f in cls.__dataclass_fields__ if f != "ssm_cfg"}
        kw = {k: v for k, v in raw.items() if k in known}
        kw.update({k: v for k, v in ssm.items() if k in known})
        return cls(ssm_cfg=ssm, **kw)


# =============================================================================
# the local fallback module tree
# =============================================================================

class Mamba2Lite(nn.Module):
    """Parameter-compatible stand-in for `mamba_ssm.modules.mamba2.Mamba2`.

    It holds EXACTLY the upstream parameter names and shapes, and exposes the
    attributes that `real_mamba/patch.py` reads. Its own `forward` is the same
    reference forward the patch installs, so `--backend local` and
    `--backend official` compute the same thing.
    """

    def __init__(self, cfg: LiteConfig, layer_idx: int = 0, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.d_model = cfg.d_model
        self.d_state = cfg.d_state
        self.d_conv = cfg.d_conv
        self.expand = cfg.expand
        self.d_inner = cfg.expand * cfg.d_model
        self.d_ssm = self.d_inner                      # d_intermediate == 0
        self.headdim = cfg.headdim
        self.ngroups = cfg.ngroups
        self.nheads = self.d_ssm // self.headdim
        self.chunk_size = cfg.chunk_size
        self.dt_limit = tuple(cfg.dt_limit)
        self.activation = "silu"
        self.rmsnorm = True
        self.norm_before_gate = False
        self.D_has_hdim = False
        self.use_mem_eff_path = False
        self.process_group = None

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **fk)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=self.d_conv,
                                groups=conv_dim, padding=self.d_conv - 1, bias=True, **fk)

        self.act = nn.SiLU()
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads, **fk))
        self.A_log = nn.Parameter(torch.zeros(self.nheads, **fk))
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))

        # Mirrors RMSNormGated(d_ssm, eps=1e-5, norm_before_gate=False,
        # group_size=d_ssm // ngroups) -- mamba2.py:143-146
        self.norm = _RMSNormGatedLite(self.d_ssm, eps=1e-5,
                                      group_size=self.d_ssm // self.ngroups,
                                      norm_before_gate=False, **fk)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **fk)

    def forward(self, u, inference_params=None, **kw):
        from real_mamba.patch import mamba2_reference_forward
        from baby_mamba.transition import ExactExp
        return mamba2_reference_forward(self, u, getattr(self, "_transition", None) or ExactExp())


class _RMSNormGatedLite(nn.Module):
    """Parameter-compatible stand-in for the Triton `RMSNorm` (gated) module."""

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=False,
                 device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x, z=None):
        from real_mamba.nn_ref import rms_norm_gated_ref
        return rms_norm_gated_ref(x, self.weight, self.bias, z=z, eps=self.eps,
                                  group_size=self.group_size,
                                  norm_before_gate=self.norm_before_gate)


class _RMSNormLite(nn.Module):
    def __init__(self, d, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x):
        return rms_norm_ref(x, self.weight, self.eps)


class _BlockLite(nn.Module):
    """Mirrors `mamba_ssm.modules.block.Block` with fused_add_norm=False.

    Add -> RMSNorm -> mixer, returning (mixer_out, residual). Same convention as
    third_party/mamba/mamba_ssm/modules/block.py:51-55.
    """

    def __init__(self, cfg: LiteConfig, layer_idx: int, device=None, dtype=None):
        super().__init__()
        self.residual_in_fp32 = cfg.residual_in_fp32
        self.norm = _RMSNormLite(cfg.d_model, eps=cfg.norm_epsilon, device=device, dtype=dtype)
        self.mixer = Mamba2Lite(cfg, layer_idx=layer_idx, device=device, dtype=dtype)
        self.mlp = None

    def forward(self, hidden_states, residual=None, **kw):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states, **kw)
        return hidden_states, residual


class MixerModelLite(nn.Module):
    def __init__(self, cfg: LiteConfig, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Embedding(cfg.padded_vocab_size, cfg.d_model,
                                      device=device, dtype=dtype)
        self.layers = nn.ModuleList([_BlockLite(cfg, i, device=device, dtype=dtype)
                                     for i in range(cfg.n_layer)])
        self.norm_f = _RMSNormLite(cfg.d_model, eps=cfg.norm_epsilon,
                                   device=device, dtype=dtype)

    def forward(self, input_ids, **kw):
        h = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, residual, **kw)
        residual = (h + residual) if residual is not None else h
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))


class MambaLMHeadModelLite(nn.Module):
    """Mirrors `MambaLMHeadModel` closely enough to share a state dict."""

    def __init__(self, cfg: LiteConfig, device=None, dtype=None):
        super().__init__()
        self.config = cfg
        self.backbone = MixerModelLite(cfg, device=device, dtype=dtype)
        self.lm_head = nn.Linear(cfg.d_model, cfg.padded_vocab_size, bias=False,
                                 device=device, dtype=dtype)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids, **kw):
        h = self.backbone(input_ids, **kw)
        logits = self.lm_head(h)
        return _Output(logits)


@dataclass
class _Output:
    """Matches the namedtuple `MambaLMHeadModel` returns (`.logits`)."""
    logits: torch.Tensor


# =============================================================================
# loading
# =============================================================================

def _load_state_dict(name: str):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(name, "pytorch_model.bin")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_local(name: str = DEFAULT_MODEL, device="cpu", dtype=torch.float32):
    cfg = LiteConfig.from_hf(name)
    model = MambaLMHeadModelLite(cfg, device="cpu", dtype=dtype)
    sd = _load_state_dict(name)
    sd = {k: v.to(dtype if v.is_floating_point() else v.dtype) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # lm_head.weight is tied to the embedding, so seeing it twice is expected.
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/module mismatch.\n  missing: {missing}\n  unexpected: {unexpected}\n"
            "The local backend mirrors upstream parameter names exactly; a mismatch means "
            "upstream changed. Use --backend official, and please update real_mamba/model.py."
        )
    model.tie_weights()
    return model.to(device).eval(), cfg


def load_official(name: str = DEFAULT_MODEL, device="cuda", dtype=torch.float32):
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel.from_pretrained(name, device=device, dtype=dtype)
    cfg = LiteConfig.from_hf(name)
    return model.eval(), cfg


def load_model(name: str = DEFAULT_MODEL, backend: str = "auto",
               device: str = "cpu", dtype=torch.float32, verbose: bool = True):
    """Returns (model, cfg, backend_used)."""
    if backend not in ("auto", "official", "local"):
        raise ValueError("backend must be auto | official | local")
    if backend in ("auto", "official"):
        try:
            model, cfg = load_official(name, device=device, dtype=dtype)
            if verbose:
                print(f"[model] loaded {name} via the official mamba_ssm package")
            return model, cfg, "official"
        except Exception as e:                                       # noqa: BLE001
            if backend == "official":
                raise
            if verbose:
                print(f"[model] official mamba_ssm unavailable ({type(e).__name__}: {e}); "
                      f"falling back to --backend local")
    model, cfg = load_local(name, device=device, dtype=dtype)
    if verbose:
        print(f"[model] loaded {name} into the local pure-PyTorch module tree "
              f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {dtype})")
    return model, cfg, "local"


def iter_mixers(model):
    """Yield (layer_idx, mixer) for every Mamba2 mixer in the model.

    Works for both backends by duck-typing on the attributes the transition
    needs, rather than on the class name.
    """
    layers = model.backbone.layers
    for i, block in enumerate(layers):
        mixer = getattr(block, "mixer", None)
        if mixer is None:
            continue
        if all(hasattr(mixer, k) for k in ("A_log", "dt_bias", "in_proj", "conv1d", "out_proj")):
            yield i, mixer


def count_parameters(model) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total
