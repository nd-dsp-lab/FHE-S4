#!/usr/bin/env python
"""Part 9 -- can lightweight fine-tuning recover what the polynomial cost?

    # 100k-token smoke test, the cheapest thing that can fail
    python finetune_poly_exp.py --transition poly4 --interval-mode per-head --pin-zero \
        --mode A --token-budget 100000

    # then
    python finetune_poly_exp.py ... --mode B --token-budget 1000000
    python finetune_poly_exp.py ... --mode C --token-budget 5000000

THREE PROGRESSIVE MODES (see real_mamba/train_utils.py for the details)
    A   polynomial FROZEN; only dt_bias / A_log / D trainable   (~1.7k params)
    B   A + the polynomial coefficients themselves
    C   B + LoRA on in_proj and out_proj
    full  everything -- an escape hatch, not a default

Nothing else is modified: SiLU, RMSNorm, softplus, the conv1d and the scan are
untouched, and `torch.exp` is never patched globally.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from real_mamba.data import get_blocks
from real_mamba.eval_lm import evaluate, peak_memory_gb, reset_peak_memory
from real_mamba.model import DEFAULT_MODEL, iter_mixers, load_model, recommended_dtype
from real_mamba.patch import patch_transition, set_transition
from real_mamba.train_utils import (
    RunLogger,
    poly_health,
    probe_transition_stats,
    save_json,
    select_trainable,
    set_seed,
    token_budget_batches,
)
from real_mamba.transitions import (
    add_interval_mode_args,
    add_transition_args,
    depth_estimate_any,
    describe_any,
    transition_from_args,
)
from baby_mamba.transition import ExactExp


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="auto",
                    choices=("auto", "float32", "bfloat16", "float16"),
                    help="parameter dtype. 'auto' picks what the GPU can actually run "
                         "(bf16 only on sm_80+; fp16 on Turing, where bf16 and fp32 "
                         "cannot compile the official kernel). The SSM scan always runs "
                         "in fp32 internally, matching what the Triton kernel does for dA.")
    ap.add_argument("--train-data", default="wikitext103")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-data", default="wikitext2")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--eval-blocks", type=int, default=48)
    ap.add_argument("--chunk-size", type=int, default=128)

    ap.add_argument("--mode", default="A", choices=("A", "B", "C", "full"))
    ap.add_argument("--token-budget", type=int, default=100_000,
                    help="total training tokens. Start at 100k, then 1M, then 5M.")
    ap.add_argument("--micro-batch", type=int, default=1, help="sequences per forward")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="micro-batches per optimiser step")
    ap.add_argument("--lr", type=float, default=None,
                    help="default depends on --mode: 1e-2 for A, 3e-3 for B/C, 1e-5 for full")
    ap.add_argument("--poly-lr", type=float, default=None,
                    help="separate lr for the polynomial coefficients (default: --lr / 10)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--log-every", type=int, default=10, help="optimiser steps per log line")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="optimiser steps between mid-training evals (0 = only start/end)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--save-weights", action="store_true",
                    help="also write the trained tensors (only the trainable ones)")
    add_transition_args(ap)
    add_interval_mode_args(ap)
    return ap


# Chosen empirically. MODE A can take a large lr because it only touches 1,728
# parameters that all sit close to the transition. MODE B/C are lower because the
# polynomial coefficients are free to leave the decay-like region, and our first
# attempt at 3e-3 diverged in four steps.
DEFAULT_LR = {"A": 1e-2, "B": 1e-3, "C": 1e-3, "full": 1e-5}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.transition == "exact":
        # This is the CONTROL run, and it matters. Fine-tuning on wikitext-103 and
        # evaluating on wikitext-2 still moves perplexity for reasons that have
        # nothing to do with the polynomial (learning-rate warmup on dt_bias,
        # domain drift, etc). Without this control you cannot tell "the model
        # adapted to the polynomial" from "any 1.7k parameters improve on this
        # eval set". Run the identical command with --transition exact and compare.
        if args.mode == "B":
            raise SystemExit(
                "--transition exact has no polynomial coefficients, so --mode B is "
                "undefined. Use --mode A (or C) for the exact control run."
            )
        print("[note] --transition exact: this is the CONTROL run. Compare its ending "
              "perplexity against the polynomial run's, not against the exact BASELINE.")
    # MODE B/C normally train the polynomial coefficients too -- but the exact-exp
    # CONTROL run has no coefficients, and auto-enabling the flag for it made
    # make_transition() raise. The control must still be allowed to run MODE C,
    # because a poly MODE C result is uninterpretable without a LoRA control.
    if args.mode in ("B", "C") and not args.trainable_poly and args.transition != "exact":
        print("[note] --mode B/C needs trainable coefficients; enabling --trainable-poly")
        args.trainable_poly = True
    if args.lr is None:
        args.lr = DEFAULT_LR[args.mode]
    if args.poly_lr is None:
        args.poly_lr = args.lr / 10.0

    tag = (f"{args.transition}_{args.interval_mode}"
           f"{'_pin0' if args.pin_zero else ''}_mode{args.mode}_{args.token_budget}tok")
    args.outdir = args.outdir or Path("runs/part9_finetune") / tag
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    args.dtype = str(dtype).replace("torch.", "")          # record what we actually used
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)

    eval_blocks, _, eval_info = get_blocks(args.eval_data, args.eval_split, args.seq_len)
    eval_blocks = eval_blocks[: args.eval_blocks]
    train_blocks, _, train_info = get_blocks(args.train_data, args.train_split, args.seq_len)

    # ------------------------------------------------------------------ baselines
    print("\n--- baseline 1: the untouched model (exact exp) ---")
    handle = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)
    exact_res = evaluate(model, eval_blocks, device=args.device,
                         vocab_size=cfg.vocab_size, progress=True)
    print(f"    exact:  loss {exact_res['loss']:.4f}  ppl {exact_res['perplexity']:.4f}")

    transition = transition_from_args(args)
    desc_before = describe_any(transition)
    set_transition(handle, transition)

    print(f"\n--- baseline 2: {args.transition} before any training ---")
    start_res = evaluate(model, eval_blocks, device=args.device,
                         vocab_size=cfg.vocab_size, progress=True)
    print(f"    start:  loss {start_res['loss']:.4f}  ppl {start_res['perplexity']:.4f}")

    # ------------------------------------------------------------------ training
    info = select_trainable(model, args.mode, args.lora_r, args.lora_alpha,
                            skip_poly=(args.transition == "exact"))
    poly_ids = {id(p) for p in (getattr(model, "fhe_transitions", None) or
                                torch.nn.ModuleDict()).parameters()}
    decay, poly = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (poly if id(p) in poly_ids else decay).append(p)
    param_groups = [{"params": decay, "lr": args.lr, "weight_decay": args.weight_decay}]
    if poly:
        param_groups.append({"params": poly, "lr": args.poly_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    total_steps = max(1, args.token_budget // tokens_per_step)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    logger = RunLogger(args.outdir)
    reset_peak_memory(args.device)

    print(f"\n--- training: MODE {args.mode}, {args.token_budget:,} tokens, "
          f"{total_steps} optimiser steps of {tokens_per_step:,} tokens "
          f"(micro_batch {args.micro_batch} x grad_accum {args.grad_accum}) ---")
    epochs = args.token_budget / max(train_info["n_tokens"], 1)
    print(f"    train corpus {train_info['n_tokens']:,} tokens -> {epochs:.2f} epochs")
    if epochs > 1.5:
        print("    NOTE: more than one epoch. Any gain may be partly memorisation; "
              "use --train-data wikitext103 or a smaller budget.")

    model.train()
    t0 = time.time()
    step, micro, seen = 0, 0, 0
    running = 0.0
    batches = token_budget_batches(train_blocks, args.token_budget, args.micro_batch,
                                   seed=args.seed)
    opt.zero_grad(set_to_none=True)
    for batch, seen in batches:
        ids = batch.to(args.device)
        logits = model(ids).logits[:, :-1].float()[..., : cfg.vocab_size]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1))
        (loss / args.grad_accum).backward()
        lv = float(loss.detach())
        if not math.isfinite(lv):
            raise SystemExit(
                f"\n[DIVERGED] training loss became {lv} at step {step}, "
                f"tokens {seen:,}.\n"
                "This is a real result, not a crash to work around. The usual causes:\n"
                "  * --lr too high for A_log (A = -exp(A_log), so A moves exponentially);\n"
                "  * --mode B with a polynomial whose coefficients can leave the\n"
                "    decay-like region -- lower --poly-lr;\n"
                "  * --interval-mode global, where the polynomial already diverges\n"
                "    before training starts (see Part 7).\n"
                "Report it; do not clamp it."
            )
        running += lv
        micro += 1
        if micro % args.grad_accum == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == 1:
                avg = running / (args.grad_accum * max(args.log_every if step > 1 else 1, 1))
                logger.log(step=step, tokens_seen=seen, loss=avg, lm_loss=avg,
                           lr=sched.get_last_lr()[0], grad_norm=float(gnorm),
                           peak_gpu_gb=peak_memory_gb(args.device))
                print(f"    step {step:5d}/{total_steps}  tokens {seen:>9,}  "
                      f"loss {avg:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"|g| {float(gnorm):.3f}  ({time.time() - t0:.0f}s)")
                running = 0.0
            if args.eval_every and step % args.eval_every == 0:
                model.eval()
                mid = evaluate(model, eval_blocks[:8], device=args.device,
                               vocab_size=cfg.vocab_size)
                print(f"      [mid-eval] ppl {mid['perplexity']:.4f}")
                model.train()
    train_seconds = time.time() - t0
    logger.close()

    # ------------------------------------------------------------------ final eval
    print("\n--- after training ---")
    model.eval()
    end_res = evaluate(model, eval_blocks, device=args.device,
                       vocab_size=cfg.vocab_size, progress=True)
    desc_after = describe_any(transition)
    health = poly_health(model)
    trans_stats = probe_transition_stats(model, handle, eval_blocks, args.device)

    metrics = {
        "starting_gap_vs_exact": start_res["perplexity"] - exact_res["perplexity"],
        "exact_perplexity": exact_res["perplexity"],
        "exact_loss": exact_res["loss"],
        "starting_perplexity": start_res["perplexity"],
        "starting_loss": start_res["loss"],
        "ending_perplexity": end_res["perplexity"],
        "ending_loss": end_res["loss"],
        "recovered_fraction": (
            (start_res["perplexity"] - end_res["perplexity"]) /
            (start_res["perplexity"] - exact_res["perplexity"])
            if math.isfinite(start_res["perplexity"])
            and abs(start_res["perplexity"] - exact_res["perplexity"]) > 1e-9 else float("nan")),
        "delta_perplexity_vs_exact": end_res["perplexity"] - exact_res["perplexity"],
        "training_seconds": train_seconds,
        "training_hours": train_seconds / 3600.0,
        "peak_gpu_memory_gb": peak_memory_gb(args.device),
        "optimiser_steps": step,
        "tokens_seen": seen,
        "trainable": info,
        "fhe_depth_estimate": depth_estimate_any(transition),
        **trans_stats,
    }

    print()
    print("=" * 78)
    print(f"PART 9 -- MODE {args.mode}, {args.transition}, "
          f"{args.interval_mode} interval, {seen:,} tokens")
    print("=" * 78)
    print(f"  exact-model perplexity  : {exact_res['perplexity']:.4f}")
    print(f"  starting perplexity     : {start_res['perplexity']:.4f}")
    print(f"  ending perplexity       : {end_res['perplexity']:.4f}")
    print(f"  delta vs exact          : {metrics['delta_perplexity_vs_exact']:+.4f}")
    gap = start_res["perplexity"] - exact_res["perplexity"]
    rf = metrics["recovered_fraction"]
    if not math.isfinite(rf) or abs(gap) < 0.05:
        print(f"  fraction of gap recovered: n/a -- the starting gap was only "
              f"{gap:+.4f} perplexity, so this ratio is meaningless.")
        print(f"      Compare the ENDING perplexity ({end_res['perplexity']:.4f}) against an")
        print(f"      exact-exp control trained identically:")
        print(f"        python {__import__('sys').argv[0]} --transition exact --mode "
              f"{'A' if args.mode == 'B' else args.mode} --token-budget {args.token_budget} "
              f"--train-data {args.train_data} --eval-blocks {args.eval_blocks}")
    elif rf > 1.0:
        print(f"  fraction of gap recovered: {rf:.3f}  -- OVER 1.0, i.e. the tuned "
              f"polynomial model beat the untuned exact model.")
        print("      Do NOT read that as 'the polynomial is better than exp'. Fine-tuning on")
        print(f"      {args.train_data} while evaluating on {args.eval_data} improves perplexity")
        print("      for reasons unrelated to the transition. Run the same command with")
        print("      --transition exact --mode A to get the control, and compare endings.")
    else:
        print(f"  fraction of gap recovered: {rf:.3f}")
    print(f"  wall-clock training time : {train_seconds:.1f}s ({train_seconds / 3600:.4f} h)")
    print(f"  peak GPU memory          : {metrics['peak_gpu_memory_gb']:.3f} GB"
          if math.isfinite(metrics["peak_gpu_memory_gb"]) else
          "  peak GPU memory          : n/a (CPU run)")
    print(f"  trainable parameters     : {info['n_trainable']:,} "
          f"({100 * info['trainable_fraction']:.4f}% of {info['n_total']:,})")
    print(f"  FHE ct-ct depth          : {metrics['fhe_depth_estimate']}")

    save_json(args.outdir / "config.json",
              {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
              | {"backend": backend, "eval_info": eval_info, "train_info": train_info})
    save_json(args.outdir / "metrics.json", metrics)
    save_json(args.outdir / "polynomial_coefficients.json",
              {"before_training": desc_before, "after_training": desc_after,
               "health_after_training": health})
    if args.save_weights:
        torch.save({k: v.detach().cpu() for k, v in model.named_parameters()
                    if v.requires_grad}, args.outdir / "trained_tensors.pt")
    handle.restore()
    print(f"\nwrote {args.outdir}/ (config.json, metrics.json, "
          f"polynomial_coefficients.json, training_log.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
