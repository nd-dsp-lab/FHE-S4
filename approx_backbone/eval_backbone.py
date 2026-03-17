import argparse

import torch
from torch.utils.data import DataLoader

from approx_backbone import (
    AddingDataset,
    evaluate_adding_model,
    load_approx_backbone,
    measure_pre_gate_range,
    resolve_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick evaluation script for the approximate S4D backbone.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to evaluate.")
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--tolerance", type=float, default=0.04)
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = resolve_checkpoint(args.checkpoint)
    model = load_approx_backbone(ckpt, seq_len=args.seq_len, device=device)

    test_ds = AddingDataset(args.test_samples, args.seq_len, seed=args.seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    mse, acc = evaluate_adding_model(model, test_loader, device=device, tolerance=args.tolerance)
    pre_gate = measure_pre_gate_range(model, test_loader, device=device)

    print(f"checkpoint: {ckpt}")
    print(f"test_mse: {mse:.6f}")
    print(f"test_acc: {acc:.4%}")
    print(f"pre_gate_range: [{pre_gate['min']:.6f}, {pre_gate['max']:.6f}]")


if __name__ == "__main__":
    main()
