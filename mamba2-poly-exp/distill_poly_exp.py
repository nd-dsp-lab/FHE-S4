#!/usr/bin/env python
"""Part 10 (optional) -- teacher/student distillation for the polynomial model.

    python distill_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
        --mode A --token-budget 1000000 --lambda-kd 1.0 --temperature 2.0

Teacher  : the untouched pretrained Mamba2-130M (exact exp), FROZEN, always
           under torch.no_grad().
Student  : the SAME checkpoint with the polynomial transition.

    L = lambda_lm * cross_entropy(student, labels)
      + lambda_kd * T^2 * KL(student_logits/T || teacher_logits/T)
      [ + lambda_transition * MSE(a_student, a_teacher) ]

Only output-logit distillation plus the optional transition matching. Hidden-state
matching is deliberately NOT implemented -- the brief says get this working first,
and it is also the variant most likely to fight the polynomial rather than help it.

MEMORY NOTE FOR ONE 24 GB GPU
-----------------------------
Teacher and student are the SAME 129M weights except for the transition, so we
do NOT hold two copies. One model is loaded; the teacher pass runs it with the
exact-exp transition under no_grad, the student pass with the polynomial. This
halves memory and removes any chance of the two differing in anything but the
transition. The cost is one extra forward pass per micro-batch.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from baby_mamba.transition import ExactExp
from real_mamba.data import get_blocks
from real_mamba.eval_lm import evaluate, peak_memory_gb, reset_peak_memory
from real_mamba.model import DEFAULT_MODEL, load_model, recommended_dtype
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


class TransitionRecorder:
    """Captures `a` per layer so the optional transition-matching loss can use it."""

    def __init__(self):
        self.a = {}
        self.enabled = True

    def __call__(self, layer_idx, z, a):
        if self.enabled:
            self.a[layer_idx] = a

    def clear(self):
        self.a = {}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="auto",
                    choices=("auto", "float32", "bfloat16", "float16"),
                    help="'auto' picks what the GPU can actually run -- see "
                         "real_mamba.model.recommended_dtype")
    ap.add_argument("--train-data", default="wikitext103")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-data", default="wikitext2")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--eval-blocks", type=int, default=48)
    ap.add_argument("--chunk-size", type=int, default=128)

    ap.add_argument("--mode", default="A", choices=("A", "B", "C", "full"))
    ap.add_argument("--token-budget", type=int, default=1_000_000)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--poly-lr", type=float, default=None)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)

    ap.add_argument("--lambda-lm", type=float, default=1.0)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--lambda-transition", type=float, default=0.0,
                    help="weight on MSE(a_student, a_teacher); 0 disables it")
    ap.add_argument("--temperature", type=float, default=2.0)

    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=None)
    add_transition_args(ap)
    add_interval_mode_args(ap)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.transition == "exact":
        raise SystemExit("distillation needs a student that differs from the teacher; "
                         "pick --transition poly2/poly3/poly4")
    if args.mode in ("B", "C") and not args.trainable_poly:
        print("[note] --mode B/C needs trainable coefficients; enabling --trainable-poly")
        args.trainable_poly = True
    if args.poly_lr is None:
        args.poly_lr = args.lr / 10.0

    tag = (f"{args.transition}_{args.interval_mode}{'_pin0' if args.pin_zero else ''}"
           f"_mode{args.mode}_kd{args.lambda_kd:g}_T{args.temperature:g}"
           f"_tr{args.lambda_transition:g}_{args.token_budget}tok")
    args.outdir = args.outdir or Path("runs/part10_distill") / tag
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    args.dtype = str(dtype).replace("torch.", "")
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    eval_blocks, _, eval_info = get_blocks(args.eval_data, args.eval_split, args.seq_len)
    eval_blocks = eval_blocks[: args.eval_blocks]
    train_blocks, _, train_info = get_blocks(args.train_data, args.train_split, args.seq_len)

    rec = TransitionRecorder()
    handle = patch_transition(model, ExactExp(), chunk_size=args.chunk_size, collector=rec)
    exact_t = ExactExp()
    student_t = transition_from_args(args)
    desc_before = describe_any(student_t)

    print("\n--- teacher (exact exp), frozen ---")
    rec.enabled = False
    exact_res = evaluate(model, eval_blocks, device=args.device,
                         vocab_size=cfg.vocab_size, progress=True)
    print(f"    teacher: loss {exact_res['loss']:.4f}  ppl {exact_res['perplexity']:.4f}")

    set_transition(handle, student_t)
    print(f"\n--- student ({args.transition}) before distillation ---")
    start_res = evaluate(model, eval_blocks, device=args.device,
                         vocab_size=cfg.vocab_size, progress=True)
    print(f"    student: loss {start_res['loss']:.4f}  ppl {start_res['perplexity']:.4f}")

    info = select_trainable(model, args.mode, args.lora_r, args.lora_alpha)
    poly_ids = {id(p) for p in (getattr(model, "fhe_transitions", None) or
                                torch.nn.ModuleDict()).parameters()}
    rest, poly = [], []
    for p in model.parameters():
        if p.requires_grad:
            (poly if id(p) in poly_ids else rest).append(p)
    groups = [{"params": rest, "lr": args.lr, "weight_decay": 0.0}]
    if poly:
        groups.append({"params": poly, "lr": args.poly_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))

    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    total_steps = max(1, args.token_budget // tokens_per_step)
    warmup = max(1, int(args.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * min((s - warmup) / max(1, total_steps - warmup), 1.0))))

    logger = RunLogger(args.outdir)
    reset_peak_memory(args.device)
    T = args.temperature
    match_transition = args.lambda_transition > 0

    print(f"\n--- distilling: MODE {args.mode}, {args.token_budget:,} tokens, "
          f"{total_steps} steps | lambda_lm={args.lambda_lm} lambda_kd={args.lambda_kd} "
          f"T={T} lambda_transition={args.lambda_transition} ---")
    model.train()
    t0 = time.time()
    step, micro = 0, 0
    acc = {"loss": 0.0, "lm": 0.0, "kd": 0.0, "tr": 0.0}
    opt.zero_grad(set_to_none=True)

    for batch, seen in token_budget_batches(train_blocks, args.token_budget,
                                            args.micro_batch, seed=args.seed):
        ids = batch.to(args.device)

        # ---- teacher pass: exact exp, no grad, frozen ----------------------
        set_transition(handle, exact_t)
        rec.enabled = match_transition
        rec.clear()
        with torch.no_grad():
            teacher_logits = model(ids).logits[:, :-1].float()[..., : cfg.vocab_size]
        teacher_a = {k: v.detach() for k, v in rec.a.items()} if match_transition else {}

        # ---- student pass: polynomial, with grad --------------------------
        set_transition(handle, student_t)
        rec.clear()
        student_logits = model(ids).logits[:, :-1].float()[..., : cfg.vocab_size]
        student_a = rec.a if match_transition else {}
        rec.enabled = False

        targets = ids[:, 1:]
        lm = F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]),
                             targets.reshape(-1))
        # KL(teacher || student) in the standard distillation direction, scaled by
        # T^2 so the gradient magnitude does not shrink as T grows.
        kd = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                      F.log_softmax(teacher_logits / T, dim=-1),
                      reduction="batchmean", log_target=True) * (T * T)
        tr = student_logits.new_zeros(())
        if match_transition and teacher_a:
            terms = [F.mse_loss(student_a[k], teacher_a[k]) for k in teacher_a
                     if k in student_a]
            if terms:
                tr = torch.stack(terms).mean()

        loss = args.lambda_lm * lm + args.lambda_kd * kd + args.lambda_transition * tr
        (loss / args.grad_accum).backward()
        lv = float(loss.detach())
        if not math.isfinite(lv):
            raise SystemExit(
                f"\n[DIVERGED] distillation loss became {lv} at step {step}.\n"
                "Lower --lr and/or --poly-lr. With --mode B the polynomial coefficients\n"
                "are free to leave the decay-like region; per-head coefficients are\n"
                "scale-normalised so one lr is meaningful, but 3e-3 is still aggressive.\n"
                "Report it; do not clamp it."
            )
        for k, v in (("loss", loss), ("lm", lm), ("kd", kd), ("tr", tr)):
            acc[k] += float(v.detach())
        micro += 1

        if micro % args.grad_accum == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == 1:
                d = args.grad_accum * (args.log_every if step > 1 else 1)
                logger.log(step=step, tokens_seen=seen, loss=acc["loss"] / d,
                           lm_loss=acc["lm"] / d, kd_loss=acc["kd"] / d,
                           transition_loss=acc["tr"] / d, lr=sched.get_last_lr()[0],
                           grad_norm=float(gnorm), peak_gpu_gb=peak_memory_gb(args.device))
                print(f"    step {step:5d}/{total_steps}  tokens {seen:>9,}  "
                      f"loss {acc['loss'] / d:.4f}  lm {acc['lm'] / d:.4f}  "
                      f"kd {acc['kd'] / d:.4f}  tr {acc['tr'] / d:.6f}  "
                      f"({time.time() - t0:.0f}s)")
                acc = {k: 0.0 for k in acc}
    train_seconds = time.time() - t0
    logger.close()

    print("\n--- after distillation ---")
    model.eval()
    set_transition(handle, student_t)
    end_res = evaluate(model, eval_blocks, device=args.device,
                       vocab_size=cfg.vocab_size, progress=True)
    trans_stats = probe_transition_stats(model, handle, eval_blocks, args.device)

    metrics = {
        "teacher_perplexity": exact_res["perplexity"],
        "starting_perplexity": start_res["perplexity"],
        "ending_perplexity": end_res["perplexity"],
        "teacher_loss": exact_res["loss"],
        "starting_loss": start_res["loss"],
        "ending_loss": end_res["loss"],
        "delta_perplexity_vs_teacher": end_res["perplexity"] - exact_res["perplexity"],
        "recovered_fraction": (
            (start_res["perplexity"] - end_res["perplexity"]) /
            (start_res["perplexity"] - exact_res["perplexity"])
            if math.isfinite(start_res["perplexity"])
            and abs(start_res["perplexity"] - exact_res["perplexity"]) > 1e-9 else float("nan")),
        "training_seconds": train_seconds,
        "training_hours": train_seconds / 3600.0,
        "peak_gpu_memory_gb": peak_memory_gb(args.device),
        "optimiser_steps": step,
        "trainable": info,
        "fhe_depth_estimate": depth_estimate_any(student_t),
        "lambda_lm": args.lambda_lm, "lambda_kd": args.lambda_kd,
        "lambda_transition": args.lambda_transition, "temperature": T,
        **trans_stats,
    }

    print()
    print("=" * 78)
    print(f"PART 10 -- DISTILLATION, MODE {args.mode}, {args.transition}")
    print("=" * 78)
    print(f"  teacher (exact) perplexity : {exact_res['perplexity']:.4f}")
    print(f"  student before             : {start_res['perplexity']:.4f}")
    print(f"  student after              : {end_res['perplexity']:.4f}")
    print(f"  delta vs teacher           : {metrics['delta_perplexity_vs_teacher']:+.4f}")
    print(f"  training time              : {train_seconds:.1f}s")
    print(f"  peak GPU memory            : {metrics['peak_gpu_memory_gb']:.3f} GB"
          if math.isfinite(metrics["peak_gpu_memory_gb"]) else
          "  peak GPU memory            : n/a (CPU run)")

    save_json(args.outdir / "config.json",
              {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
              | {"backend": backend, "eval_info": eval_info, "train_info": train_info})
    save_json(args.outdir / "metrics.json", metrics)
    save_json(args.outdir / "polynomial_coefficients.json",
              {"before": desc_before, "after": describe_any(student_t),
               "health_after": poly_health(model)})
    handle.restore()
    print(f"\nwrote {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
