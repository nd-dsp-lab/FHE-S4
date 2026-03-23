import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adding-problem models across GLU gate modes.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["exact", "poly4_gate", "poly6_gate", "linear_gate"],
        choices=["exact", "poly4_gate", "poly6_gate", "linear_gate"],
        help="GLU modes to train.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Forwarded to adding_problem.py as --device.",
    )
    parser.add_argument("--results_json", type=str, default="glu_retrain_results.json")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing per-mode checkpoints if they exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregated = {"runs": []}

    for mode in args.modes:
        ckpt_path = f"s4d_best_{mode}.pt"
        results_path = f"glu_retrain_{mode}.json"
        cmd = [
            sys.executable,
            "adding_problem.py",
            "--glu_mode",
            mode,
            "--epochs",
            str(args.epochs),
            "--seq_len",
            str(args.seq_len),
            "--d_model",
            str(args.d_model),
            "--d_state",
            str(args.d_state),
            "--train_samples",
            str(args.train_samples),
            "--val_samples",
            str(args.val_samples),
            "--test_samples",
            str(args.test_samples),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--dropout",
            str(args.dropout),
            "--tolerance",
            str(args.tolerance),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--best_ckpt",
            ckpt_path,
            "--save_results",
            results_path,
        ]
        if args.resume and Path(ckpt_path).exists():
            cmd.extend(["--ckpt", ckpt_path])

        print(f"\n=== Training mode: {mode} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        with open(results_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)
        aggregated["runs"].append(run_info)

    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, sort_keys=True)

    print(f"\nSaved aggregated results to {args.results_json}")
    print("\nSummary:")
    header = f"{'mode':<12} {'test_mse':>12} {'test_acc':>12} {'pre_gate_range':>28}"
    print(header)
    print("-" * len(header))
    for run in aggregated["runs"]:
        rng = run.get("pre_gate_range", {})
        rng_text = f"[{rng.get('min', float('nan')):.4f}, {rng.get('max', float('nan')):.4f}]"
        print(
            f"{run['glu_mode']:<12} "
            f"{run['test_mse']:>12.6f} "
            f"{run['test_acc']:>12.4%} "
            f"{rng_text:>28}"
        )


if __name__ == "__main__":
    main()
