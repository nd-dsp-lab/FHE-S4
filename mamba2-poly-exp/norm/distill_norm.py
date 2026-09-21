#!/usr/bin/env python
"""PHASE 2/3 -- distil the norm replacement back to teacher quality.

    python norm/distill_norm.py --stage A3 --token-budget 100000

The teacher is the untouched pretrained model. The student is the SAME weights
with the norm operator replaced. They share every parameter except the norm, so
only one model is in memory and the two provably differ in nothing else --
verified: teacher mode reproduces the untouched model bit-for-bit.

    L = lambda_lm * CE(student, labels)
      + lambda_kd * T^2 * KL(student/T || teacher/T)
      + lambda_aux * mean_over_sites  MSE(student_norm_out, teacher_norm_out)
                                      / mean(teacher_norm_out^2)

The auxiliary term is optional and is reported both with and without, as the
brief asks. It is a *local* signal at exactly the operator being replaced, which
the logit-level KL only sees after 24 layers of mixing. It is normalised by the
teacher's own scale so one lambda is meaningful across all 49 sites, whose
magnitudes differ by orders of magnitude.

Evaluated at BOTH sequence lengths every stage. A method that passes 512 and
fails 2048 is a drift result and is reported as such, never averaged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baby_mamba.transition import ExactExp
from norm.install_norms import (
    STAGES,
    exact_mode,
    install_path_a,
    install_path_b,
    norm_kwargs_for_mixer,
    norm_outputs,
    recording,
)
from real_mamba.data import get_blocks
from real_mamba.eval_lm import peak_memory_gb, reset_peak_memory
from real_mamba.model import DEFAULT_MODEL, load_model, recommended_dtype
from real_mamba.patch import mamba2_reference_forward, patch_transition
from real_mamba.train_utils import RunLogger, save_json, set_seed, token_budget_batches


def bind_forwards(handle, mixers, chunk_size):
    """Install the reference forward with this stage's gated-norm override."""
    for mixer in mixers:
        kw = norm_kwargs_for_mixer(handle, mixer.layer_idx)

        def bound(u, _m=mixer, _kw=kw, inference_params=None, **extra):
            return mamba2_reference_forward(_m, u, _m._transition,
                                             chunk_size=chunk_size, **_kw, **extra)
        mixer.forward = bound


@torch.no_grad()
def evaluate_ppl(model, blocks, device, vocab_size, batch_size=1):
    tot, ntok = 0.0, 0
    for i in range(0, blocks.shape[0], batch_size):
        ids = blocks[i:i + batch_size].to(device)
        lg = model(ids).logits[:, :-1].float()[..., :vocab_size]
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                     ids[:, 1:].reshape(-1), reduction="sum"))
        ntok += ids[:, 1:].numel()
    m = tot / ntok
    return (math.exp(m) if m < 700 else float("inf")), m


def aux_loss(student_outs, teacher_outs):
    """Scale-free MSE at the replaced sites."""
    terms = []
    for k, t in teacher_outs.items():
        s = student_outs.get(k)
        if s is None:
            continue
        tf = t.float()
        terms.append(F.mse_loss(s.float(), tf) / (tf.square().mean() + 1e-8))
    if not terms:
        return None
    return torch.stack(terms).mean()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=("auto", "float32", "bfloat16", "float16"))
    ap.add_argument("--path", default="A", choices=("A", "B"),
                    help="A = learned constant divisor (0 levels). "
                         "B = prescaled Newton inverse sqrt (2/5/8 levels for t=1/2/3).")
    ap.add_argument("--stage", default="A3", choices=sorted(STAGES))
    ap.add_argument("--t-steps", type=int, default=2,
                    help="Path B only: Newton iterations. MEASURED convergence basin is "
                         "v/s in [0.25, 2.0]; outside it MORE steps diverge faster.")
    ap.add_argument("--lambda-range", type=float, default=0.0,
                    help="Path B only: weight on mean(log(v/s)^2), pulling the prescaled "
                         "argument into the convergence basin.")
    ap.add_argument("--norm-stats", type=Path, default=Path("norm/norm_stats.json"))
    ap.add_argument("--stats-length", default="512",
                    help="which sequence length's Phase 1 medians initialise c")
    ap.add_argument("--train-data", default="wikitext2")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-data", default="wikitext2")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--train-seq-len", type=int, default=512)
    ap.add_argument("--eval-lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--eval-blocks", type=int, default=40)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--token-budget", type=int, default=100_000)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--c-lr", type=float, default=1e-2,
                    help="separate lr for the log_c scalars, which start far from optimal")
    ap.add_argument("--lambda-lm", type=float, default=1.0)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--lambda-aux", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--precision-bits", type=int, default=None,
                    help="GUARDRAIL: re-evaluate with norm-path activations rounded "
                         "to this many significant bits (20 ~ post-bootstrap CKKS)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.norm_stats.exists():
        raise SystemExit(f"need {args.norm_stats}; run norm/collect_norm_stats.py first")
    tag = (f"{args.stage}_kd{args.lambda_kd:g}_T{args.temperature:g}"
           f"_aux{args.lambda_aux:g}_{args.token_budget}tok")
    args.outdir = args.outdir or Path("runs/norm_pathA") / tag
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    stats_all = json.loads(args.norm_stats.read_text())
    stats = stats_all["by_length"][args.stats_length]

    ph = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)
    if args.path == "A":
        handle = install_path_a(model, stats, stage=args.stage, precision_bits=None)
    else:
        handle = install_path_b(model, stats, stage=args.stage.replace("A", "B"),
                                t_steps=args.t_steps, precision_bits=None)
    bind_forwards(handle, ph.mixers, args.chunk_size)

    eval_sets = {}
    for L in args.eval_lengths:
        b, _, _ = get_blocks(args.eval_data, args.eval_split, L, verbose=False)
        eval_sets[L] = b[: args.eval_blocks]
    train_blocks, _, tinfo = get_blocks(args.train_data, args.train_split,
                                        args.train_seq_len, verbose=False)

    # ---------------- baselines ------------------------------------------
    print(f"\n--- teacher (untouched) and student-before, stage {args.stage} ---")
    base, start = {}, {}
    model.eval()
    for L, blocks in eval_sets.items():
        with exact_mode(handle):
            base[L] = evaluate_ppl(model, blocks, args.device, cfg.vocab_size)[0]
        start[L] = evaluate_ppl(model, blocks, args.device, cfg.vocab_size)[0]
        print(f"  L={L:>5}  teacher {base[L]:9.4f}   student before {start[L]:9.4f} "
              f"({start[L] - base[L]:+.4f})")

    # ---------------- optimiser ------------------------------------------
    c_params = [p for n, p in model.named_parameters()
                if p.requires_grad and n.endswith("log_c")]
    c_ids = {id(p) for p in c_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in c_ids]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in c_params + other:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [{"params": c_params, "lr": args.c_lr, "weight_decay": 0.0},
         {"params": other, "lr": args.lr, "weight_decay": 0.0}], betas=(0.9, 0.95))
    n_train = sum(p.numel() for p in c_params + other)
    print(f"[train] {len(c_params)} log_c scalars + {sum(p.numel() for p in other):,} "
          f"other = {n_train:,} trainable of {sum(p.numel() for p in model.parameters()):,}")

    tps = args.micro_batch * args.grad_accum * args.train_seq_len
    total_steps = max(1, args.token_budget // tps)
    warm = max(1, int(args.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, total_steps - warm), 1.0))))
    logger = RunLogger(args.outdir)
    reset_peak_memory(args.device)
    T = args.temperature

    print(f"\n--- distilling: {args.token_budget:,} tokens, {total_steps} steps, "
          f"lm={args.lambda_lm} kd={args.lambda_kd} T={T} aux={args.lambda_aux} ---")
    model.train()
    t0, step, micro = time.time(), 0, 0
    acc = {"loss": 0.0, "lm": 0.0, "kd": 0.0, "aux": 0.0}
    opt.zero_grad(set_to_none=True)
    use_aux = args.lambda_aux > 0

    for batch, seen in token_budget_batches(train_blocks, args.token_budget,
                                            args.micro_batch, seed=args.seed):
        ids = batch.to(args.device)
        with torch.no_grad(), exact_mode(handle, record=use_aux):
            tl = model(ids).logits[:, :-1].float()[..., : cfg.vocab_size]
            t_outs = {k: v.detach() for k, v in norm_outputs(handle).items()} if use_aux else {}
        need_rec = use_aux or (args.path == "B" and args.lambda_range > 0)
        ctx = recording(handle) if need_rec else torch.enable_grad()
        with ctx:
            sl = model(ids).logits[:, :-1].float()[..., : cfg.vocab_size]
            s_outs = norm_outputs(handle) if use_aux else {}

        tgt = ids[:, 1:]
        lm = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), tgt.reshape(-1))
        kd = F.kl_div(F.log_softmax(sl / T, -1), F.log_softmax(tl / T, -1),
                      reduction="batchmean", log_target=True) * (T * T)
        ax = aux_loss(s_outs, t_outs) if use_aux else None
        loss = args.lambda_lm * lm + args.lambda_kd * kd
        if ax is not None:
            loss = loss + args.lambda_aux * ax
        if args.path == "B" and args.lambda_range > 0:
            from norm.newton_norm import range_penalty
            rp = range_penalty(list(handle.replacements.values()))
            if rp is not None:
                loss = loss + args.lambda_range * rp
        lv = float(loss.detach())
        if not math.isfinite(lv):
            raise SystemExit(f"\n[DIVERGED] loss={lv} at step {step}. Lower --lr/--c-lr. "
                             f"Report it; do not clamp it.")
        (loss / args.grad_accum).backward()
        for k, v in (("loss", loss), ("lm", lm), ("kd", kd),
                     ("aux", ax if ax is not None else torch.zeros(()))):
            acc[k] += float(v.detach())
        micro += 1
        if micro % args.grad_accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(c_params + other, args.grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % args.log_every == 0 or step == 1:
                d = args.grad_accum * (args.log_every if step > 1 else 1)
                logger.log(step=step, tokens_seen=seen, loss=acc["loss"] / d,
                           lm_loss=acc["lm"] / d, kd_loss=acc["kd"] / d,
                           transition_loss=acc["aux"] / d, lr=sched.get_last_lr()[0],
                           grad_norm=float(gn), peak_gpu_gb=peak_memory_gb(args.device))
                print(f"    step {step:5d}/{total_steps} tok {seen:>9,}  "
                      f"loss {acc['loss'] / d:.4f}  lm {acc['lm'] / d:.4f}  "
                      f"kd {acc['kd'] / d:.4f}  aux {acc['aux'] / d:.5f}  "
                      f"({time.time() - t0:.0f}s)")
                acc = {k: 0.0 for k in acc}
    train_s = time.time() - t0
    logger.close()

    # ---------------- final evaluation, both lengths, then 20-bit ---------
    print("\n--- after distillation ---")
    model.eval()
    end, end20 = {}, {}
    for L, blocks in eval_sets.items():
        end[L] = evaluate_ppl(model, blocks, args.device, cfg.vocab_size)[0]
        print(f"  L={L:>5}  teacher {base[L]:9.4f}  before {start[L]:9.4f}  "
              f"after {end[L]:9.4f}  (delta vs teacher {end[L] - base[L]:+.4f})")
    if args.precision_bits:
        for m in handle.replacements.values():
            m.precision_bits = args.precision_bits
        for L, blocks in eval_sets.items():
            end20[L] = evaluate_ppl(model, blocks, args.device, cfg.vocab_size)[0]
            print(f"  L={L:>5}  at {args.precision_bits}-bit norm precision: "
                  f"{end20[L]:9.4f}  (delta vs fp32 student {end20[L] - end[L]:+.4f})")
        for m in handle.replacements.values():
            m.precision_bits = None

    cs = {k: float(m.c.detach()) for k, m in handle.replacements.items()}
    metrics = {
        "stage": args.stage, "teacher_ppl": base, "student_before_ppl": start,
        "student_after_ppl": end, "student_after_ppl_20bit": end20,
        "delta_vs_teacher": {str(L): end[L] - base[L] for L in end},
        "path": args.path,
        "levels_per_instance": (0 if args.path == "A" else
                                next(iter(handle.replacements.values())).ct_ct_depth
                                if handle.replacements else None),
        "t_steps": args.t_steps if args.path == "B" else None,
        "lambda_range": args.lambda_range if args.path == "B" else None,
        "n_instances_replaced": len(handle.replacements),
        "n_trainable": n_train, "training_seconds": train_s,
        "training_hours": train_s / 3600, "optimiser_steps": step,
        "peak_gpu_memory_gb": peak_memory_gb(args.device),
        "learned_c": cs,
        "lambda_lm": args.lambda_lm, "lambda_kd": args.lambda_kd,
        "lambda_aux": args.lambda_aux, "temperature": T,
    }
    save_json(args.outdir / "metrics.json", metrics)
    save_json(args.outdir / "config.json",
              {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
              | {"backend": backend, "train_info": tinfo})
    handle.restore(); ph.restore()
    print(f"\nwrote {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
