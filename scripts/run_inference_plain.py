import argparse
import sys
import time
from pathlib import Path

import torch

# Allow running as: python scripts/run_inference_plain.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    # Common package-style layout.
    from s4d.model import MiniS4D  # type: ignore
except Exception:
    # Fallback for this repository layout.
    from train_mini_s4d import MiniS4D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny plaintext S4D inference demo.")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length.")
    parser.add_argument("--d-model", type=int, default=4, help="Model width.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")
    batch_size = 1

    torch.manual_seed(args.seed)

    model = MiniS4D(d_model=args.d_model, L=args.seq_len, dropout=0.0).to(device)
    model.eval()

    x = torch.randn(batch_size, args.d_model, args.seq_len, device=device)

    with torch.no_grad():
        t0 = time.perf_counter()
        y = model(x)
        t1 = time.perf_counter()

    print(f"input_shape: {tuple(x.shape)}")
    print(f"output_shape: {tuple(y.shape)}")
    print(f"forward_time_s: {t1 - t0:.6f}")


if __name__ == "__main__":
    main()
