import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


class MiniS4D(nn.Module):
    def __init__(self, d_model: int = 64, d_state: int = 64, L: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.L = L
        self.D = nn.Parameter(torch.randn(d_model))
        self.kernel_gen = S4DKernel(d_model, d_state)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        K = self.kernel_gen(self.L)
        k_conv = K.unsqueeze(1)
        y = torch.nn.functional.conv1d(u, k_conv, padding=self.L - 1, groups=self.d_model)
        y = y[:, :, : self.L]
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        return self.decoder(y.mean(dim=-1))


class AddingModel(nn.Module):
    def __init__(self, d_model: int = 64, d_state: int = 64, seq_len: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Linear(2, d_model)
        self.s4d = MiniS4D(d_model=d_model, d_state=d_state, L=seq_len, dropout=dropout)

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
    def __init__(self, num_samples: int, seq_len: int, seed: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.seed = seed
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


def resolve_checkpoint(path_str: str) -> Path:
    candidates = [
        Path(path_str),
        Path("s4d") / Path(path_str).name,
        Path("s4d_adding_best.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find checkpoint. Tried: {candidates}")


def load_adding_model(checkpoint_path: Path, seq_len: int, device: torch.device) -> AddingModel:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if "encoder.weight" not in state:
        raise ValueError("Checkpoint does not appear to be a full AddingModel state_dict.")
    d_model = int(state["encoder.weight"].shape[0])
    d_state = int(state["s4d.kernel_gen.log_A_real"].shape[1])
    model = AddingModel(d_model=d_model, d_state=d_state, seq_len=seq_len, dropout=0.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_range_tracker() -> dict[str, float]:
    return {"min": float("inf"), "max": float("-inf")}


def update_range(stats: dict[str, float], tensor: torch.Tensor) -> None:
    stats["min"] = min(stats["min"], float(tensor.min().item()))
    stats["max"] = max(stats["max"], float(tensor.max().item()))


def collect_hook_ranges(
    model: AddingModel,
    seq_len: int,
    num_batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    tracked = {
        "s4d_input_u": make_range_tracker(),
        "after_gelu": make_range_tracker(),
        "output_linear_input": make_range_tracker(),
        "conv1d_pre_glu": make_range_tracker(),
    }

    def s4d_input_hook(_module, inputs):
        update_range(tracked["s4d_input_u"], inputs[0].detach())

    def gelu_out_hook(_module, _inputs, output):
        update_range(tracked["after_gelu"], output.detach())

    def output_linear_input_hook(_module, inputs):
        update_range(tracked["output_linear_input"], inputs[0].detach())

    def conv1d_out_hook(_module, _inputs, output):
        update_range(tracked["conv1d_pre_glu"], output.detach())

    handles = [
        model.s4d.register_forward_pre_hook(s4d_input_hook),
        model.s4d.activation.register_forward_hook(gelu_out_hook),
        model.s4d.output_linear.register_forward_pre_hook(output_linear_input_hook),
        model.s4d.output_linear[0].register_forward_hook(conv1d_out_hook),
    ]

    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(num_batches):
            x, _ = adding_problem_batch(batch_size=batch_size, seq_len=seq_len, rng=rng)
            _ = model(x.to(device))

    for handle in handles:
        handle.remove()
    return tracked


def adding_problem_batch(batch_size: int, seq_len: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for _ in range(batch_size):
        x, y = adding_problem_sample(seq_len, rng)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


def forward_with_optional_norm(
    model: AddingModel,
    x: torch.Tensor,
    norm_mode: str,
    range_stats: dict[str, float] | None = None,
) -> torch.Tensor:
    x = model.encoder(x)
    u = x.transpose(-1, -2)
    s4d = model.s4d
    K = s4d.kernel_gen(s4d.L)
    k_conv = K.unsqueeze(1)
    y = torch.nn.functional.conv1d(u, k_conv, padding=s4d.L - 1, groups=s4d.d_model)
    y = y[:, :, : s4d.L]
    y = y + u * s4d.D.unsqueeze(-1)
    y = s4d.dropout(s4d.activation(y))

    if norm_mode == "layernorm":
        y = F.layer_norm(y.transpose(1, 2), (s4d.d_model,)).transpose(1, 2)
    elif norm_mode == "l2norm":
        y = F.normalize(y, p=2.0, dim=1)

    if range_stats is not None:
        update_range(range_stats, y.detach())

    y = s4d.output_linear(y)
    return s4d.decoder(y.mean(dim=-1))


def evaluate_with_optional_norm(
    model: AddingModel,
    loader: DataLoader,
    device: torch.device,
    tolerance: float,
    norm_mode: str,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    total_mse = 0.0
    correct = 0
    n = 0
    range_stats = make_range_tracker()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(-1)
            out = forward_with_optional_norm(model, x, norm_mode=norm_mode, range_stats=range_stats)
            total_mse += F.mse_loss(out, y, reduction="sum").item()
            correct += ((out - y).abs().squeeze(-1) < tolerance).sum().item()
            n += x.size(0)

    return total_mse / max(n, 1), correct / max(n, 1), range_stats


def range_within_bounds(stats: dict[str, float], lo: float = -4.0, hi: float = 4.0) -> bool:
    return stats["min"] >= lo and stats["max"] <= hi


def print_range(label: str, stats: dict[str, float]) -> None:
    print(f"{label:<24} [{stats['min']:.6f}, {stats['max']:.6f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose activation ranges for S4D adding-problem checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="s4d/s4d_adding_best.pt")
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--num_probe_batches", type=int, default=100)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--test_seed", type=int, default=44)
    parser.add_argument("--tolerance", type=float, default=0.04)
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model = load_adding_model(checkpoint_path, seq_len=args.seq_len, device=device)

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")

    hook_ranges = collect_hook_ranges(
        model=model,
        seq_len=args.seq_len,
        num_batches=args.num_probe_batches,
        batch_size=args.probe_batch_size,
        seed=args.test_seed,
        device=device,
    )

    print("\nactivation ranges over probe batches:")
    print_range("raw S4D input u", hook_ranges["s4d_input_u"])
    print_range("after GELU", hook_ranges["after_gelu"])
    print_range("output_linear input", hook_ranges["output_linear_input"])
    print_range("Conv1d before GLU", hook_ranges["conv1d_pre_glu"])

    test_ds = AddingDataset(args.test_samples, args.seq_len, seed=args.test_seed)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)

    baseline_mse, baseline_acc, baseline_range = evaluate_with_optional_norm(
        model=model,
        loader=test_loader,
        device=device,
        tolerance=args.tolerance,
        norm_mode="none",
    )
    layernorm_mse, layernorm_acc, layernorm_range = evaluate_with_optional_norm(
        model=model,
        loader=test_loader,
        device=device,
        tolerance=args.tolerance,
        norm_mode="layernorm",
    )
    l2norm_mse, l2norm_acc, l2norm_range = evaluate_with_optional_norm(
        model=model,
        loader=test_loader,
        device=device,
        tolerance=args.tolerance,
        norm_mode="l2norm",
    )

    print("\nfull test set results (1000 samples, seed=44):")
    print(
        f"{'variant':<18} {'range':<28} {'test_mse':>12} {'test_acc':>12} {'candidate':>14}"
    )
    print("-" * 90)

    def candidate_label(rng_stats: dict[str, float], acc: float) -> str:
        if range_within_bounds(rng_stats) and acc >= 0.95:
            return "YES"
        return "NO"

    variants = [
        ("baseline", baseline_range, baseline_mse, baseline_acc),
        ("layernorm", layernorm_range, layernorm_mse, layernorm_acc),
        ("l2norm", l2norm_range, l2norm_mse, l2norm_acc),
    ]
    for name, rng_stats, mse, acc in variants:
        rng_text = f"[{rng_stats['min']:.4f}, {rng_stats['max']:.4f}]"
        print(
            f"{name:<18} {rng_text:<28} {mse:>12.6f} {acc:>12.4%} {candidate_label(rng_stats, acc):>14}"
        )

    print("\nFHE-friendly normalization candidate rule:")
    print("- resulting pre-output_linear range must stay within [-4, 4]")
    print("- test accuracy must stay above 95%")
    print("- fixed linear or constant-scale normalization is attractive for CKKS because it adds zero multiplicative depth")


if __name__ == "__main__":
    main()
