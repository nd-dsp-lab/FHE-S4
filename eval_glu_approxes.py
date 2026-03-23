import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import minimize


def silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def poly_eval_np(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    return Polynomial(coeffs)(x)


def fit_minimax_poly(
    fn,
    degree: int,
    interval: tuple[float, float],
    fit_points: int = 4001,
    maxiter: int = 400,
) -> np.ndarray:
    lo, hi = interval
    x_fit = np.linspace(lo, hi, fit_points, dtype=np.float64)
    y_fit = fn(x_fit)

    init = Chebyshev.fit(x_fit, y_fit, deg=degree, domain=[lo, hi]).convert(kind=Polynomial)
    x_objective = np.linspace(lo, hi, fit_points, dtype=np.float64)
    y_objective = fn(x_objective)

    def objective(coeffs: np.ndarray) -> float:
        err = poly_eval_np(coeffs, x_objective) - y_objective
        return float(np.max(np.abs(err)))

    result = minimize(
        objective,
        x0=np.asarray(init.coef, dtype=np.float64),
        method="Powell",
        options={"maxiter": maxiter, "xtol": 1e-8, "ftol": 1e-8, "disp": False},
    )
    return np.asarray(result.x, dtype=np.float64)


def fit_linear_least_squares(
    fn,
    interval: tuple[float, float],
    fit_points: int = 4001,
) -> np.ndarray:
    lo, hi = interval
    x = np.linspace(lo, hi, fit_points, dtype=np.float64)
    y = fn(x)
    slope, intercept = np.polyfit(x, y, deg=1)
    return np.array([intercept, slope], dtype=np.float64)


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


def adding_problem_batch(batch_size: int, seq_len: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.rand(batch_size, seq_len, generator=rng)
    masks = torch.zeros(batch_size, seq_len)
    targets = torch.zeros(batch_size)
    for b in range(batch_size):
        indices = torch.randperm(seq_len, generator=rng)[:2]
        masks[b, indices[0]] = 1.0
        masks[b, indices[1]] = 1.0
        targets[b] = values[b, indices[0]] + values[b, indices[1]]
    x = torch.stack([values, masks], dim=-1)
    return x, targets


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


def collect_output_linear_input_stats(
    checkpoint_path: Path,
    seq_len: int,
    num_batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    model = load_adding_model(checkpoint_path, seq_len=seq_len, device=device)
    stats = {"min": float("inf"), "max": float("-inf")}

    def hook(_module, inputs):
        x = inputs[0].detach()
        stats["min"] = min(stats["min"], float(x.min().item()))
        stats["max"] = max(stats["max"], float(x.max().item()))

    handle = model.s4d.output_linear.register_forward_pre_hook(hook)
    rng = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for _ in range(num_batches):
            x, _ = adding_problem_batch(batch_size=batch_size, seq_len=seq_len, rng=rng)
            _ = model(x.to(device))

    handle.remove()

    num_bins = 256
    hist_counts = np.zeros(num_bins, dtype=np.float64)
    hist_edges = np.linspace(stats["min"], stats["max"], num_bins + 1, dtype=np.float64)

    def hist_hook(_module, inputs):
        x = inputs[0].detach().cpu().numpy().reshape(-1)
        counts, _ = np.histogram(x, bins=hist_edges)
        hist_counts[:] += counts

    handle = model.s4d.output_linear.register_forward_pre_hook(hist_hook)
    rng = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for _ in range(num_batches):
            x, _ = adding_problem_batch(batch_size=batch_size, seq_len=seq_len, rng=rng)
            _ = model(x.to(device))

    handle.remove()
    return {
        "min": stats["min"],
        "max": stats["max"],
        "hist_counts": hist_counts,
        "hist_edges": hist_edges,
    }


def weighted_rms_from_hist(
    fn,
    hist_counts: np.ndarray,
    hist_edges: np.ndarray,
) -> float:
    total = float(np.sum(hist_counts))
    if total == 0:
        return float("nan")
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    weights = hist_counts / total
    err = fn(centers) - silu_np(centers)
    return float(np.sqrt(np.sum(weights * np.square(err))))


def format_coeffs(coeffs: np.ndarray) -> str:
    return "[" + ", ".join(f"{c:.8g}" for c in coeffs) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GLU/SiLU approximations for S4D.")
    parser.add_argument("--checkpoint", type=str, default="s4d/s4d_adding_best.pt")
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--num_probe_batches", type=int, default=100)
    parser.add_argument("--probe_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--eval_lo", type=float, default=-0.5)
    parser.add_argument("--eval_hi", type=float, default=15.0)
    parser.add_argument("--eval_points", type=int, default=10000)
    parser.add_argument("--plot_lo", type=float, default=-1.0)
    parser.add_argument("--plot_hi", type=float, default=15.0)
    parser.add_argument("--plot_points", type=int, default=2000)
    parser.add_argument("--plot_path", type=str, default="glu_approx_comparison.png")
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    print(f"checkpoint: {checkpoint_path}")
    stats = collect_output_linear_input_stats(
        checkpoint_path=checkpoint_path,
        seq_len=args.seq_len,
        num_batches=args.num_probe_batches,
        batch_size=args.probe_batch_size,
        seed=args.seed,
        device=device,
    )
    print(
        "empirical output_linear input range over "
        f"{args.num_probe_batches} batches: "
        f"[{stats['min']:.6f}, {stats['max']:.6f}]"
    )
    in_assumed_range = stats["min"] >= args.eval_lo and stats["max"] <= args.eval_hi
    print(f"within assumed evaluation range [{args.eval_lo}, {args.eval_hi}]: {in_assumed_range}")
    print(
        "note: this domain is highly asymmetric and SiLU is nearly linear for large "
        "positive x, so direct minimax fits behave differently than on symmetric [-4, 4]."
    )

    sigmoid_poly3 = fit_minimax_poly(sigmoid_np, degree=3, interval=(args.eval_lo, args.eval_hi))
    silu_poly4 = fit_minimax_poly(silu_np, degree=4, interval=(args.eval_lo, args.eval_hi))
    silu_poly6 = fit_minimax_poly(silu_np, degree=6, interval=(args.eval_lo, args.eval_hi))
    center = 0.5 * (args.eval_lo + args.eval_hi)
    scale = 0.5 * (args.eval_hi - args.eval_lo)
    silu_poly6_centered = fit_minimax_poly(
        lambda t: silu_np(center + scale * t),
        degree=6,
        interval=(-1.0, 1.0),
    )
    linear_fit_hi = fit_linear_least_squares(silu_np, interval=(4.0, args.eval_hi))

    x_eval = np.linspace(args.eval_lo, args.eval_hi, args.eval_points, dtype=np.float64)
    y_true_eval = silu_np(x_eval)

    def deg3_sigmoid_then_x(x: np.ndarray) -> np.ndarray:
        return x * poly_eval_np(sigmoid_poly3, x)

    def deg4_silu_direct(x: np.ndarray) -> np.ndarray:
        return poly_eval_np(silu_poly4, x)

    def deg6_silu_direct(x: np.ndarray) -> np.ndarray:
        return poly_eval_np(silu_poly6, x)

    def linear_gate(x: np.ndarray) -> np.ndarray:
        return x * np.clip(0.25 * x + 0.5, 0.0, 1.0)

    def deg6_centered(x: np.ndarray) -> np.ndarray:
        t = (x - center) / scale
        return poly_eval_np(silu_poly6_centered, t)

    def piecewise_poly4_linear(x: np.ndarray) -> np.ndarray:
        y = deg4_silu_direct(x).copy()
        hi_mask = x > 4.0
        y[hi_mask] = poly_eval_np(linear_fit_hi, x[hi_mask])
        return y

    approximations = {
        "deg3_sigmoid_then_x": {
            "fn": deg3_sigmoid_then_x,
            "note": "",
        },
        "deg4_silu_direct": {
            "fn": deg4_silu_direct,
            "note": "",
        },
        "deg6_silu_direct": {
            "fn": deg6_silu_direct,
            "note": "",
        },
        "linear_gate": {
            "fn": linear_gate,
            "note": "",
        },
        "deg6_centered_direct": {
            "fn": deg6_centered,
            "note": "fit in centered variable t=(x-7.25)/7.75",
        },
        "piecewise_poly4_linear": {
            "fn": piecewise_poly4_linear,
            "note": "branch costs ~15 bootstrapping levels in CKKS, likely impractical.",
        },
    }

    interval_domains = [
        ("[-0.5,4]", (-0.5, 4.0)),
        ("[4,10]", (4.0, 10.0)),
        ("[10,15]", (10.0, 15.0)),
    ]

    rows = []
    for name, approx in approximations.items():
        fn = approx["fn"]
        y_hat = fn(x_eval)
        err = y_hat - y_true_eval
        interval_stats = {}
        for interval_name, (lo, hi) in interval_domains:
            mask = (x_eval >= lo) & (x_eval <= hi)
            interval_err = err[mask]
            interval_stats[interval_name] = {
                "max_abs_error": float(np.max(np.abs(interval_err))),
                "rms_error": float(np.sqrt(np.mean(np.square(interval_err)))),
            }
        rows.append(
            {
                "name": name,
                "max_abs_error": float(np.max(np.abs(err))),
                "rms_error": float(np.sqrt(np.mean(np.square(err)))),
                "interval_stats": interval_stats,
                "note": approx["note"],
            }
        )

    hist_counts = stats["hist_counts"]
    hist_edges = stats["hist_edges"]
    weighted_poly4 = weighted_rms_from_hist(deg4_silu_direct, hist_counts, hist_edges)
    weighted_centered6 = weighted_rms_from_hist(deg6_centered, hist_counts, hist_edges)

    print("\napproximation coefficients:")
    print(f"deg3 sigmoid coeffs: {format_coeffs(sigmoid_poly3)}")
    print(f"deg4 SiLU coeffs:    {format_coeffs(silu_poly4)}")
    print(f"deg6 SiLU coeffs:    {format_coeffs(silu_poly6)}")
    print(f"deg6 centered coeffs in t-domain: {format_coeffs(silu_poly6_centered)}")
    print(f"best linear fit on [4, 15]: y ~= {linear_fit_hi[1]:.8f} * x + {linear_fit_hi[0]:.8f}")

    print(f"\nsummary on uniform grid over [{args.eval_lo}, {args.eval_hi}]:")
    header = f"{'approximation':<24} {'max_abs_error':>16} {'rms_error':>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['name']:<24} {row['max_abs_error']:>16.8e} {row['rms_error']:>16.8e}")
        if row["note"]:
            print(f"  note: {row['note']}")

    print("\ninterval-wise errors:")
    interval_header = (
        f"{'approximation':<24} "
        f"{'[-0.5,4] max/rms':>24} "
        f"{'[4,10] max/rms':>24} "
        f"{'[10,15] max/rms':>24}"
    )
    print(interval_header)
    print("-" * len(interval_header))
    for row in rows:
        s1 = row["interval_stats"]["[-0.5,4]"]
        s2 = row["interval_stats"]["[4,10]"]
        s3 = row["interval_stats"]["[10,15]"]
        c1 = f"{s1['max_abs_error']:.3e}/{s1['rms_error']:.3e}"
        c2 = f"{s2['max_abs_error']:.3e}/{s2['rms_error']:.3e}"
        c3 = f"{s3['max_abs_error']:.3e}/{s3['rms_error']:.3e}"
        print(
            f"{row['name']:<24} "
            f"{c1:>24} "
            f"{c2:>24} "
            f"{c3:>24}"
        )

    print("\nweighted RMS by empirical activation histogram:")
    print(f"deg4_silu_direct:    {weighted_poly4:.8e}")
    print(f"deg6_centered_direct:{weighted_centered6:.8e}")

    x_plot = np.linspace(args.plot_lo, args.plot_hi, args.plot_points, dtype=np.float64)
    y_true_plot = silu_np(x_plot)
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_true_plot, label="true SiLU", linewidth=3, color="black")
    for row in rows:
        plt.plot(x_plot, approximations[row["name"]]["fn"](x_plot), label=row["name"], linewidth=2)
    plt.axvline(args.eval_lo, color="gray", linestyle="--", linewidth=1)
    plt.axvline(args.eval_hi, color="gray", linestyle="--", linewidth=1)
    plt.axvspan(float(stats["min"]), float(stats["max"]), color="gold", alpha=0.15, label="empirical activation range")
    plt.title("SiLU / GLU Approximation Comparison")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=200)
    print(f"\nsaved figure: {args.plot_path}")


if __name__ == "__main__":
    main()
