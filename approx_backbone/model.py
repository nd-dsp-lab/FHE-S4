"""Standalone approximate S4D backbone for adding-problem inference.

This package is intended as a handoff artifact for collaborators who will add
the FHE/OpenFHE logic later. It is pure PyTorch/Numpy and does not import
TenSEAL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_GELU_MODE = "poly6"
DEFAULT_GLU_MODE = "linear_gate"
DEFAULT_CHECKPOINT = "s4d_best_gelu_poly6_glu_linear_gate.pt"

GLU_GATE_DOMAIN = (-142.0, 142.0)
GELU_DOMAIN = (-5.0, 15.0)

POLY4_GATE_CHEB = np.array([0.5, 1.03580642, 0.0, -0.273067994, 0.0], dtype=np.float64)
POLY6_GATE_CHEB = np.array(
    [0.5, 0.927977957, 0.0, -0.35367563, 0.0, 0.168670784, 0.0],
    dtype=np.float64,
)
POLY4_GELU_CHEB = np.array(
    [5.977405989737, 7.988856571172, 1.530634414283, -0.667904507889, 0.246138181485],
    dtype=np.float64,
)
POLY6_GELU_CHEB = np.array(
    [
        6.04831644882,
        8.094460228971,
        1.371512430522,
        -0.740775295685,
        0.134773988002,
        0.126019270103,
        -0.239569101196,
    ],
    dtype=np.float64,
)


def chebyshev_eval_torch(
    x: torch.Tensor,
    coeffs: torch.Tensor,
    domain: tuple[float, float],
) -> torch.Tensor:
    lo, hi = domain
    t = (2.0 * x - (hi + lo)) / (hi - lo)
    if coeffs.numel() == 1:
        return coeffs[0] * torch.ones_like(t)

    t0 = torch.ones_like(t)
    result = coeffs[0] * t0
    t1 = t
    result = result + coeffs[1] * t1
    for i in range(2, coeffs.numel()):
        ti = 2.0 * t * t1 - t0
        result = result + coeffs[i] * ti
        t0, t1 = t1, ti
    return result


class SelectableGELU(nn.Module):
    def __init__(self, gelu_mode: str = DEFAULT_GELU_MODE):
        super().__init__()
        self.gelu_mode = gelu_mode
        self.gelu_domain = GELU_DOMAIN
        self.register_buffer("poly4_coeffs", torch.tensor(POLY4_GELU_CHEB, dtype=torch.float32))
        self.register_buffer("poly6_coeffs", torch.tensor(POLY6_GELU_CHEB, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gelu_mode == "exact":
            return F.gelu(x)
        if self.gelu_mode == "poly4":
            return chebyshev_eval_torch(x, self.poly4_coeffs, GELU_DOMAIN)
        if self.gelu_mode == "poly6":
            return chebyshev_eval_torch(x, self.poly6_coeffs, GELU_DOMAIN)
        raise ValueError(f"Unsupported gelu_mode: {self.gelu_mode}")


class SelectableGLU(nn.Module):
    def __init__(self, glu_mode: str = DEFAULT_GLU_MODE, split_dim: int = -2):
        super().__init__()
        self.glu_mode = glu_mode
        self.split_dim = split_dim
        self.register_buffer("poly4_coeffs", torch.tensor(POLY4_GATE_CHEB, dtype=torch.float32))
        self.register_buffer("poly6_coeffs", torch.tensor(POLY6_GATE_CHEB, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=self.split_dim)
        if self.glu_mode == "exact":
            gate = torch.sigmoid(b)
        elif self.glu_mode == "poly4_gate":
            gate = chebyshev_eval_torch(b, self.poly4_coeffs, GLU_GATE_DOMAIN)
        elif self.glu_mode == "poly6_gate":
            gate = chebyshev_eval_torch(b, self.poly6_coeffs, GLU_GATE_DOMAIN)
        elif self.glu_mode == "linear_gate":
            gate = torch.clamp(0.25 * b + 0.5, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported glu_mode: {self.glu_mode}")
        return a * gate


class S4DKernel(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.A_imag = nn.Parameter(torch.pi * torch.arange(d_state).float().repeat(d_model, 1))
        self.B = nn.Parameter(torch.ones(d_model, d_state, dtype=torch.cfloat))
        self.C = nn.Parameter(torch.randn(d_model, d_state, dtype=torch.cfloat))

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt_broad = dt.unsqueeze(-1)
        A_bar = torch.exp(A * dt_broad)
        B_bar = (1 / A) * (A_bar - 1.0) * self.B
        t = torch.arange(L, device=A.device).unsqueeze(0).unsqueeze(0)
        A_bar_pow = torch.pow(A_bar.unsqueeze(-1), t)
        K = (self.C.unsqueeze(-1) * A_bar_pow * B_bar.unsqueeze(-1)).sum(dim=1)
        return K.real


class ApproxMiniS4D(nn.Module):
    """Approximate nonlinear S4D block for handoff to the FHE team."""

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 64,
        L: int = 1000,
        dropout: float = 0.0,
        gelu_mode: str = DEFAULT_GELU_MODE,
        glu_mode: str = DEFAULT_GLU_MODE,
    ):
        super().__init__()
        self.d_model = d_model
        self.L = L
        self.gelu_mode = gelu_mode
        self.glu_mode = glu_mode

        self.D = nn.Parameter(torch.randn(d_model))
        self.kernel_gen = S4DKernel(d_model, d_state)
        self.activation = SelectableGELU(gelu_mode=gelu_mode)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            SelectableGLU(glu_mode=glu_mode, split_dim=-2),
        )
        self.decoder = nn.Linear(d_model, 1)

    def forward_features(self, u: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose intermediate tensors to simplify future FHE insertion."""
        K = self.kernel_gen(self.L)
        k_conv = K.unsqueeze(1)
        conv = torch.nn.functional.conv1d(u, k_conv, padding=self.L - 1, groups=self.d_model)
        conv = conv[:, :, : self.L]
        skip = conv + u * self.D.unsqueeze(-1)
        post_gelu = self.dropout(self.activation(skip))
        pre_gate = self.output_linear[0](post_gelu)
        gated = self.output_linear[1](pre_gate)
        pooled = gated.mean(dim=-1)
        return {
            "kernel": K,
            "conv": conv,
            "skip": skip,
            "post_gelu": post_gelu,
            "pre_gate": pre_gate,
            "gated": gated,
            "pooled": pooled,
        }

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(u)
        return self.decoder(feats["pooled"])


class ApproxAddingModel(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 64,
        seq_len: int = 1000,
        dropout: float = 0.0,
        gelu_mode: str = DEFAULT_GELU_MODE,
        glu_mode: str = DEFAULT_GLU_MODE,
    ):
        super().__init__()
        self.encoder = nn.Linear(2, d_model)
        self.s4d = ApproxMiniS4D(
            d_model=d_model,
            d_state=d_state,
            L=seq_len,
            dropout=dropout,
            gelu_mode=gelu_mode,
            glu_mode=glu_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.transpose(-1, -2)
        return self.s4d(x)


def adding_problem_sample(seq_len: int, rng: torch.Generator) -> tuple[torch.Tensor, float]:
    values = torch.rand(seq_len, generator=rng)
    indices = torch.randperm(seq_len, generator=rng)[:2]
    mask = torch.zeros(seq_len)
    mask[indices[0]] = 1.0
    mask[indices[1]] = 1.0
    target = float(values[indices[0]] + values[indices[1]])
    x = torch.stack([values, mask], dim=-1)
    return x, target


class AddingDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, seed: int = 44):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.rng = torch.Generator().manual_seed(seed)
        self._data = None

    def _generate(self) -> None:
        if self._data is not None:
            return
        xs, ys = [], []
        for _ in range(self.num_samples):
            x, y = adding_problem_sample(self.seq_len, self.rng)
            xs.append(x)
            ys.append(y)
        self._data = (torch.stack(xs), torch.tensor(ys, dtype=torch.float32))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._generate()
        return self._data[0][idx], self._data[1][idx]


def resolve_checkpoint(path_str: str | None = None) -> Path:
    candidates = []
    if path_str:
        candidates.append(Path(path_str))
    candidates.extend(
        [
            Path(DEFAULT_CHECKPOINT),
            Path("s4d") / DEFAULT_CHECKPOINT,
            Path("s4d_best_linear_gate.pt"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find checkpoint. Tried: {candidates}")


def load_approx_backbone(
    checkpoint_path: str | Path | None = None,
    seq_len: int = 1000,
    device: torch.device | None = None,
) -> ApproxAddingModel:
    if device is None:
        device = torch.device("cpu")
    ckpt = resolve_checkpoint(str(checkpoint_path) if checkpoint_path is not None else None)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    d_model = int(state["encoder.weight"].shape[0])
    d_state = int(state["s4d.kernel_gen.log_A_real"].shape[1])
    model = ApproxAddingModel(
        d_model=d_model,
        d_state=d_state,
        seq_len=seq_len,
        dropout=0.0,
        gelu_mode=DEFAULT_GELU_MODE,
        glu_mode=DEFAULT_GLU_MODE,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate_adding_model(
    model: ApproxAddingModel,
    loader: DataLoader,
    device: torch.device,
    tolerance: float = 0.04,
) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(-1)
            out = model(x)
            total_mse += F.mse_loss(out, y, reduction="sum").item()
            correct += ((out - y).abs().squeeze(-1) < tolerance).sum().item()
            n += x.size(0)
    return total_mse / max(n, 1), correct / max(n, 1)


def measure_pre_gate_range(
    model: ApproxAddingModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    stats = {"min": float("inf"), "max": float("-inf")}

    def hook(_module, _inputs, output):
        stats["min"] = min(stats["min"], float(output.min().item()))
        stats["max"] = max(stats["max"], float(output.max().item()))

    handle = model.s4d.output_linear[0].register_forward_hook(hook)
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
    handle.remove()
    return stats

