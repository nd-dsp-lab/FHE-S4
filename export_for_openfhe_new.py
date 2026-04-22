import argparse
import json
import time

import numpy as np
import torch

from run_inference_fhe_shell_new import (
    extract_channel_kernel_coeffs,
    pack_per_channel,
    toeplitz_plain,
)

try:
    from approx_backbone.model import ApproxMiniS4D  # type: ignore
except Exception:
    from train_mini_s4d import MiniS4D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Toeplitz inputs, coefficients, and expected output for C++ testing."
    )
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--toeplitz_K",
        type=int,
        default=16,
        help="Number of kernel taps to use in Toeplitz op (<= seq_len).",
    )
    parser.add_argument("--out", type=str, default="forward_pass_data.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Model and inputs
    batch_size = 1
    device = torch.device("cpu")
    seq_len = args.seq_len
    d_model = args.d_model

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_np = np.random.randn(seq_len, d_model).astype(np.float64)
    x_torch = torch.from_numpy(x_np.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)

    model = ApproxMiniS4D(d_model=d_model, L=seq_len, dropout=0.0).to(device)
    model.eval()

    # Extract GELU coefficients
    gelu_module = model.activation
    cheb_coeffs = gelu_module.poly6_coeffs.detach().cpu().numpy()
    gelu_domain = gelu_module.gelu_domain

    # Extract kernel and pack
    K = min(seq_len, args.toeplitz_K)
    coeffs_by_channel = extract_channel_kernel_coeffs(model, seq_len, d_model, K)
    chans_plain = pack_per_channel(x_np)

    # Phase 1: Toeplitz + Skip Connection
    y_conv_chans = [
        toeplitz_plain(chans_plain[c], coeffs_by_channel[c]) for c in range(d_model)
    ]
    
    D = model.D.detach().cpu().numpy().astype(np.float64)
    y_skip_chans = [
        y_conv_chans[c] + chans_plain[c] * D[c] for c in range(d_model)
    ]
    
    # Phase 2: Compute remaining non-linear components in PyTorch
    y_skip_matrix = np.stack(y_skip_chans, axis=1) # (L, d_model)
    with torch.no_grad():
        y_torch = torch.from_numpy(y_skip_matrix.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)
        y_hybrid = model.dropout(model.activation(y_torch))
        y_hybrid = model.output_linear(y_hybrid) # (1, d_model, L)
        
        y_act_np = y_hybrid.detach().cpu().numpy()[0].astype(np.float64) # (d_model, L)
        y_act_chans = [y_act_np[c] for c in range(d_model)]

        y_hybrid_out = model.decoder(y_hybrid.mean(dim=-1))
        out_expected = float(y_hybrid_out.detach().cpu().numpy()[0, 0])

    decoder_weight = model.decoder.weight.detach().cpu().numpy()[0].astype(np.float64).tolist()
    decoder_bias = float(model.decoder.bias.detach().cpu().numpy()[0])

    conv_layer = model.output_linear[0]
    # Conv1d weight shape is (out_channels, in_channels, kernel_size=1).
    # Export as a 2D matrix so C++ can apply channel mixing explicitly.
    conv_weight = conv_layer.weight.detach().cpu().numpy()[:, :, 0].astype(np.float64).tolist()
    conv_bias = conv_layer.bias.detach().cpu().numpy().astype(np.float64).tolist()

    # Save to JSON
    export_data = {
        "gelu_cheb": cheb_coeffs.tolist(),
        "gelu_domain": list(gelu_domain),
        "seq_len": seq_len,
        "d_model": d_model,
        "toeplitz_K": K,
        "conv_weight": conv_weight,
        "conv_bias": conv_bias,
        "decoder_weight": decoder_weight,
        "decoder_bias": decoder_bias,
        "out_expected": out_expected,
        "channels": []
    }

    for c in range(d_model):
        export_data["channels"].append({
            "x": chans_plain[c].tolist(),
            "coeffs": coeffs_by_channel[c].tolist(),
            "D": float(D[c]),
            "y_skip_expected": y_skip_chans[c].tolist(),
            "y_act": y_act_chans[c].tolist()
        })

    with open(args.out, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported data to {args.out}")


if __name__ == "__main__":
    main()
