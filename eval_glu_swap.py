import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Dataset


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_minimax_chebyshev(
    fn,
    degree: int,
    interval: tuple[float, float],
    fit_points: int = 8001,
    maxiter: int = 400,
) -> Chebyshev:
    lo, hi = interval
    x_fit = np.linspace(lo, hi, fit_points, dtype=np.float64)
    y_fit = fn(x_fit)
    init = Chebyshev.fit(x_fit, y_fit, deg=degree, domain=[lo, hi])

    x_objective = np.linspace(lo, hi, fit_points, dtype=np.float64)
    y_objective = fn(x_objective)

    def objective(coeffs: np.ndarray) -> float:
        cheb = Chebyshev(coeffs, domain=[lo, hi])
        err = cheb(x_objective) - y_objective
        return float(np.max(np.abs(err)))

    result = minimize(
        objective,
        x0=np.asarray(init.coef, dtype=np.float64),
        method="Powell",
        options={"maxiter": maxiter, "xtol": 1e-8, "ftol": 1e-8, "disp": False},
    )
    return Chebyshev(np.asarray(result.x, dtype=np.float64), domain=[lo, hi])


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


def cheb_eval_torch(x: torch.Tensor, coeffs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
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


class ApproxGLU(nn.Module):
    def __init__(self, mode: str, domain: tuple[float, float], cheb_coeffs: np.ndarray | None = None):
        super().__init__()
        self.mode = mode
        self.lo = float(domain[0])
        self.hi = float(domain[1])
        if cheb_coeffs is not None:
            self.register_buffer("cheb_coeffs", torch.tensor(cheb_coeffs, dtype=torch.float32))
        else:
            self.cheb_coeffs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        if self.mode == "exact":
            gate = torch.sigmoid(b)
        elif self.mode in {"poly4", "poly6"}:
            if self.cheb_coeffs is None:
                raise RuntimeError("Polynomial coefficients are required for polynomial GLU approximation.")
            gate = cheb_eval_torch(b, self.cheb_coeffs, self.lo, self.hi)
        elif self.mode == "linear":
            gate = torch.clamp(0.25 * b + 0.5, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return a * gate


def make_swapped_model(
    checkpoint_path: Path,
    seq_len: int,
    device: torch.device,
    mode: str,
    domain: tuple[float, float],
    cheb_coeffs: np.ndarray | None = None,
) -> AddingModel:
    model = load_adding_model(checkpoint_path, seq_len=seq_len, device=device)
    conv = model.s4d.output_linear[0]
    model.s4d.output_linear = nn.Sequential(conv, ApproxGLU(mode=mode, domain=domain, cheb_coeffs=cheb_coeffs))
    model.eval()
    return model


def evaluate(model: AddingModel, loader: DataLoader, device: torch.device, tolerance: float) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(-1)
            out = model(x)
            total_mse += torch.nn.functional.mse_loss(out, y, reduction="sum").item()
            correct += ((out - y).abs().squeeze(-1) < tolerance).sum().item()
            n += x.size(0)
    return total_mse / max(n, 1), correct / max(n, 1)


def measure_pre_glu_range(
    checkpoint_path: Path,
    seq_len: int,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model = load_adding_model(checkpoint_path, seq_len=seq_len, device=device)
    stats = {"min": float("inf"), "max": float("-inf")}

    def hook(_module, _inputs, output):
        stats["min"] = min(stats["min"], float(output.min().item()))
        stats["max"] = max(stats["max"], float(output.max().item()))

    handle = model.s4d.output_linear[0].register_forward_hook(hook)
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
    handle.remove()
    return stats["min"], stats["max"]


def pointwise_gate_errors(
    domain: tuple[float, float],
    num_points: int,
    poly4: Chebyshev,
    poly6: Chebyshev,
) -> list[dict[str, float | str]]:
    lo, hi = domain
    x = np.linspace(lo, hi, num_points, dtype=np.float64)
    y_true = sigmoid_np(x)
    variants = {
        "poly4": poly4(x),
        "poly6": poly6(x),
        "linear": np.clip(0.25 * x + 0.5, 0.0, 1.0),
    }
    rows = []
    for name, y_hat in variants.items():
        err = y_hat - y_true
        rows.append(
            {
                "name": name,
                "max_abs_error": float(np.max(np.abs(err))),
                "rms_error": float(np.sqrt(np.mean(np.square(err)))),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Drop-in GLU replacement test for the adding-problem checkpoint. "
            "Because nn.GLU computes a * sigmoid(b), this script preserves the "
            "GLU architecture and approximates the sigmoid gate on the second half "
            "of the Conv1d output."
        )
    )
    parser.add_argument("--checkpoint", type=str, default="s4d/s4d_adding_best.pt")
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--tolerance", type=float, default=0.04)
    parser.add_argument("--fit_lo", type=float, default=-142.0)
    parser.add_argument("--fit_hi", type=float, default=142.0)
    parser.add_argument("--fit_points", type=int, default=8001)
    parser.add_argument("--eval_points", type=int, default=20000)
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    fit_domain = (args.fit_lo, args.fit_hi)

    test_ds = AddingDataset(args.test_samples, args.seq_len, seed=args.seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"checkpoint: {checkpoint_path}")
    print(f"fit domain for gate approximations: [{args.fit_lo}, {args.fit_hi}]")

    observed_lo, observed_hi = measure_pre_glu_range(
        checkpoint_path=checkpoint_path,
        seq_len=args.seq_len,
        loader=test_loader,
        device=device,
    )
    print(f"observed Conv1d-to-GLU range on test set: [{observed_lo:.6f}, {observed_hi:.6f}]")

    poly4 = fit_minimax_chebyshev(
        sigmoid_np,
        degree=4,
        interval=fit_domain,
        fit_points=args.fit_points,
    )
    poly6 = fit_minimax_chebyshev(
        sigmoid_np,
        degree=6,
        interval=fit_domain,
        fit_points=args.fit_points,
    )

    gate_error_rows = pointwise_gate_errors(
        domain=fit_domain,
        num_points=args.eval_points,
        poly4=poly4,
        poly6=poly6,
    )
    print("\npointwise gate-approximation error on [-142, 142] vs sigmoid:")
    header = f"{'approximation':<12} {'max_abs_error':>16} {'rms_error':>16}"
    print(header)
    print("-" * len(header))
    for row in gate_error_rows:
        print(f"{row['name']:<12} {row['max_abs_error']:>16.8e} {row['rms_error']:>16.8e}")

    print("\nfitted gate approximation coefficients (Chebyshev basis over fit domain):")
    print("poly4:", np.array2string(np.asarray(poly4.coef), precision=8, separator=", "))
    print("poly6:", np.array2string(np.asarray(poly6.coef), precision=8, separator=", "))

    variants = [
        ("exact_glu", "exact", None),
        ("poly4_gate", "poly4", np.asarray(poly4.coef, dtype=np.float64)),
        ("poly6_gate", "poly6", np.asarray(poly6.coef, dtype=np.float64)),
        ("linear_gate", "linear", None),
    ]

    results = []
    baseline_mse = None
    for label, mode, coeffs in variants:
        model = make_swapped_model(
            checkpoint_path=checkpoint_path,
            seq_len=args.seq_len,
            device=device,
            mode=mode,
            domain=fit_domain,
            cheb_coeffs=coeffs,
        )
        mse, acc = evaluate(model, test_loader, device=device, tolerance=args.tolerance)
        if baseline_mse is None:
            baseline_mse = mse
        results.append(
            {
                "variant": label,
                "test_mse": mse,
                "test_acc": acc,
                "mse_delta_vs_baseline": mse - baseline_mse,
            }
        )
        print(f"finished {label}: mse={mse:.6f}, acc={acc:.4%}")

    print("\nadding problem swap-test results:")
    out_header = f"{'variant':<16} {'test_mse':>12} {'test_acc':>12} {'mse_delta':>14}"
    print(out_header)
    print("-" * len(out_header))
    for row in results:
        print(
            f"{row['variant']:<16} "
            f"{row['test_mse']:>12.6f} "
            f"{row['test_acc']:>12.4%} "
            f"{row['mse_delta_vs_baseline']:>14.6f}"
        )


if __name__ == "__main__":
    main()
