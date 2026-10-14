#!/usr/bin/env python
"""Absorb RMSNorm into the next linear layer (THE-X style), and fine-tune the
WHOLE model to live with it.

    python norm/absorb_ft.py --stage A1 --token-budget 1000000            # absorbed
    python norm/absorb_ft.py --stage A1 --token-budget 1000000 --control  # its control

WHY THIS EXISTS. THE-X (Chen et al. 2022, sec. 3.1.3) replaces LayerNorm with a
fixed per-feature scale, `x * gamma`, which folds into the next linear layer and
costs zero ciphertext levels. Path A (norm/distill_norm.py) is that recipe, and
on a 24-layer LM it is not accurate enough: +39 ppl after 1M tokens with only the
constants trainable, because within one layer v = mean(x^2) varies ~30x from token
to token and one constant cannot track it. THE-X also allows fine-tuning the model
together with the replacement; Path A deliberately froze everything else. This
script is that missing experiment.

WHAT IS TRAINED. Everything except A_log, dt_bias (training them once pushed the
decay factor above 1 on 2-4% of tokens) and the token embedding (tied to lm_head,
and held by the client, which embeds its own input).

    loss = lm * CE + kd * T^2 KL(student/T || teacher/T) + vspread * mean_layers Var(log v)

`vspread` pushes the model to keep each layer's v constant across tokens. If it
does, a constant divisor stops being an approximation and becomes exact.

THE TEACHER is a separate, frozen copy of the original model. The one-weight-set
trick distill_norm.py uses (teacher = same weights in exact mode) is only valid
while the backbone is frozen.

THE CONTROL (--control) trains the unmodified model with the identical recipe.
Fine-tuning on wikitext lowers wikitext perplexity by itself; only the
absorbed-vs-control difference says whether absorption is affordable. Compare the
two runs with norm/compare_absorb.py.

ABSORPTION IS CHECKED, NOT CLAIMED. After training, each constant is folded into
in_proj (W <- W diag(gamma / c)), the norm becomes a pass-through, and the logits
are compared with the unfolded model.

Scope: stage A1 (the 24 residual pre-norms). norm_f moves to the client (lm_head is
the input embedding, which the client already holds). The gated norm is next.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baby_mamba.transition import ExactExp
from norm.install_norms import NormHandle, _unfuse_add_norm, find_norm_sites, install_path_a
from real_mamba.data import get_blocks
from real_mamba.eval_lm import peak_memory_gb, reset_peak_memory
from real_mamba.model import DEFAULT_MODEL, load_model
from real_mamba.patch import patch_transition
from real_mamba.train_utils import RunLogger, save_json, set_seed, token_budget_batches

FROZEN_SUFFIXES = ("A_log", "dt_bias")


class FoldedNorm(nn.Module):
    """What an absorbed norm becomes: nothing. Keeps `.weight` only because the
    unfused Block reads `self.norm.weight.dtype` to cast its input."""

    def __init__(self, like: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", torch.ones(1, dtype=like.dtype, device=like.device))

    def forward(self, x):
        return x


def unfuse(model):
    h = NormHandle(model=model)
    _unfuse_add_norm(model, h)
    return h


def pre_norms(model):
    return {l: getattr(p, a) for l, (p, a, _) in find_norm_sites(model)["norm_pre"].items()}


class VSpread:
    """Var over tokens of log(mean(x^2)) at each pre-norm's INPUT, per layer.

    Recorded through forward pre-hooks, which only fire because the model is
    unfused (fused add-norm never calls the norm module -- see install_norms.py)."""

    def __init__(self, model, eps=1e-5):
        self.eps, self.vals, self.on, self.handles = eps, {}, False, []
        for l, m in pre_norms(model).items():
            self.handles.append(m.register_forward_pre_hook(self._hook(l)))

    def _hook(self, layer):
        def f(_m, args):
            if self.on:
                x = args[0].float()
                lv = torch.log(x.square().mean(-1) + self.eps).reshape(-1)
                self.vals[layer] = lv.var() if lv.numel() > 1 else lv.sum() * 0
        return f

    def penalty(self):
        return torch.stack(list(self.vals.values())).mean() if self.vals else None

    def std_by_layer(self):
        return {l: float(v.detach().sqrt()) for l, v in sorted(self.vals.items())}


@torch.no_grad()
def block_nlls(model, blocks, device, vocab, spread: VSpread | None = None):
    """Per-block mean NLL (each block is the unit of the paired test), and the
    per-layer std of log v averaged over blocks."""
    out, sds = [], []
    if spread is not None:
        spread.on = True
    for i in range(blocks.shape[0]):
        ids = blocks[i:i + 1].to(device)
        lg = model(ids).logits[:, :-1].float()[..., :vocab]
        out.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))))
        if spread is not None:
            sds.append(spread.std_by_layer())
    if spread is not None:
        spread.on = False
    sd = ({l: float(np.mean([d[l] for d in sds])) for l in sds[0]} if sds else {})
    return np.array(out), sd


def paired(nll_a, nll_b):
    """ppl(a) - ppl(b), block by block: mean and 2 SEM."""
    d = np.exp(nll_a) - np.exp(nll_b)
    if not np.all(np.isfinite(d)):
        return {"d_ppl": float("inf"), "two_sem": float("nan"), "verdict": "BROKEN"}
    sem = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    m = float(d.mean())
    return {"d_ppl": m, "two_sem": 2 * sem,
            "verdict": "not distinguishable" if abs(m) <= 2 * sem else "SIGNIFICANT"}


@torch.no_grad()
def calibrate_c(teacher, blocks, device, eps=1e-5):
    """Per-layer median sqrt(v) at each pre-norm input: the constant that best
    matches RMSNorm on a typical token. Computed here, so the run needs no
    Phase 1 file from another machine."""
    acc = {}
    hs = []
    for l, m in pre_norms(teacher).items():
        def f(_m, args, _l=l):
            v = args[0].float().square().mean(-1).reshape(-1) + eps
            acc.setdefault(_l, []).append(v.sqrt().cpu())
        hs.append(m.register_forward_pre_hook(f))
    for i in range(blocks.shape[0]):
        teacher(blocks[i:i + 1].to(device))
    for h in hs:
        h.remove()
    return {"norm_pre": {str(l): {"sqrt_median": float(torch.cat(v).median())} for l, v in acc.items()}}


@torch.no_grad()
def fold_and_check(model, ids, vocab):
    """Fold every constant divisor into the following in_proj, swap the norm for
    a pass-through, and confirm the logits do not move."""
    before = model(ids).logits[:, :, :vocab].float()
    n = 0
    for l, (parent, attr, _) in find_norm_sites(model)["norm_pre"].items():
        m = getattr(parent, attr)
        if not hasattr(m, "log_c"):
            continue
        s = (m.weight.float() / m.c).to(parent.mixer.in_proj.weight.dtype)      # (d_model,)
        W = parent.mixer.in_proj.weight
        W.data.mul_(s[None, :])                                                 # W <- W diag(s)
        setattr(parent, attr, FoldedNorm(W))
        n += 1
    after = model(ids).logits[:, :, :vocab].float()
    rel = float((after - before).abs().max() / before.abs().max())
    return {"folded": n, "max_rel_logit_change": rel}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stage", default="A1", choices=("A1",),
                    help="A1 = the 24 residual pre-norms. norm_f goes to the client; gated norm is next")
    ap.add_argument("--control", action="store_true",
                    help="no replacement: the exact model trained with the identical recipe")
    ap.add_argument("--token-budget", type=int, default=1_000_000)
    ap.add_argument("--train-data", default="wikitext103")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--train-seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-backbone", type=float, default=2e-5,
                    help="small on purpose: a full fine-tune at 1e-3 diverged (240 -> 5833 ppl, job 1464468)")
    ap.add_argument("--lr-gamma", type=float, default=1e-3)
    ap.add_argument("--lr-c", type=float, default=1e-2)
    ap.add_argument("--lambda-lm", type=float, default=1.0)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--lambda-vspread", type=float, default=0.0,
                    help="weight on mean_layers Var_tokens(log v) at the pre-norm inputs. SCALE: "
                         "the KD term is SUMMED over the ~511 positions of a sequence (batchmean "
                         "over a batch of one, inherited from distill_norm.py) while Var(log v) is "
                         "O(0.1-0.5), so lambda 0.1 is ~1e-4 of the loss -- round 1 (job 1539519) "
                         "ran 0.1 and it changed nothing. Meaningful values are O(100-1000).")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--calib-blocks", type=int, default=16)
    ap.add_argument("--eval-data", default="wikitext2")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--eval-lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--eval-blocks", type=int, default=40)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)

    kind = "control" if args.control else f"absorb_{args.stage}"
    tag = f"{kind}_vs{args.lambda_vspread:g}_lr{args.lr_backbone:g}_{args.token_budget}tok"
    args.outdir = args.outdir or Path("runs/norm_absorb") / tag
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dev = args.device

    # fp32 + the reference scan (patch_transition): the official Triton scan only
    # compiles in fp16 on sm_75, and a fine-tune needs fp32.
    student, cfg, backend = load_model(args.model, args.backend, dev, torch.float32)
    teacher, _, _ = load_model(args.model, args.backend, dev, torch.float32)
    for m in (student, teacher):
        patch_transition(m, ExactExp(), chunk_size=args.chunk_size)
    unfuse(teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    vocab = cfg.vocab_size

    train_blocks, _, tinfo = get_blocks(args.train_data, args.train_split, args.train_seq_len, verbose=False)
    eval_sets = {L: get_blocks(args.eval_data, args.eval_split, L, verbose=False)[0][: args.eval_blocks]
                 for L in args.eval_lengths}

    # -------- install (or not, for the control) ---------------------------
    if args.control:
        handle = NormHandle(model=student)
        _unfuse_add_norm(student, handle)          # same code path as the absorbed run
        stats = None
    else:
        stats = calibrate_c(teacher, train_blocks[: args.calib_blocks], dev)
        handle = install_path_a(student, stats, stage=args.stage, precision_bits=None)
    spread = VSpread(student)
    t_spread = VSpread(teacher)

    # -------- baselines ------------------------------------------------------
    base, start = {}, {}
    res = {"teacher": {}, "before": {}, "after": {}, "vstd": {}}
    print(f"\n--- {kind}: teacher vs student before training ---")
    for L, blocks in eval_sets.items():
        tn, tsd = block_nlls(teacher, blocks, dev, vocab, t_spread)
        sn, _ = block_nlls(student, blocks, dev, vocab)
        base[L], start[L] = tn, sn
        pr = paired(sn, tn)
        res["teacher"][L] = {"ppl": float(np.exp(tn).mean())}
        res["before"][L] = pr
        res["vstd"].setdefault("teacher", {})[L] = tsd
        print(f"  L={L:>5}  teacher {np.exp(tn).mean():9.4f}   student {np.exp(sn).mean():9.4f}   "
              f"d {pr['d_ppl']:+.4f} +/- {pr['two_sem']:.4f}")

    # -------- what trains ----------------------------------------------------
    repl = getattr(student, "norm_replacements", None)
    rep_ids = {id(p) for p in repl.parameters()} if repl is not None else set()
    c_params, g_params, bb_params, frozen = [], [], [], []
    for n, p in student.named_parameters():
        if n.endswith(FROZEN_SUFFIXES) or "embedding" in n or n.startswith("lm_head"):
            p.requires_grad_(False)
            frozen.append(n)
        elif id(p) in rep_ids:
            p.requires_grad_(True)
            (c_params if n.endswith("log_c") else g_params).append(p)
        else:
            p.requires_grad_(True)
            bb_params.append(p)
    assert not any(n.endswith(FROZEN_SUFFIXES) and p.requires_grad for n, p in student.named_parameters())
    groups = [{"params": bb_params, "lr": args.lr_backbone}]
    if g_params:
        groups.append({"params": g_params, "lr": args.lr_gamma})
    if c_params:
        groups.append({"params": c_params, "lr": args.lr_c})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    trainable = c_params + g_params + bb_params
    n_tr = sum(p.numel() for p in {id(p): p for p in trainable}.values())
    print(f"[train] {n_tr:,} trainable ({len(c_params)} constants); frozen: A_log, dt_bias, embedding/lm_head")

    tps = args.micro_batch * args.grad_accum * args.train_seq_len
    total = max(1, args.token_budget // tps)
    warm = max(1, int(args.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, total - warm), 1.0))))
    logger = RunLogger(args.outdir)
    reset_peak_memory(dev)
    T = args.temperature

    # -------- train ----------------------------------------------------------
    print(f"\n--- training {args.token_budget:,} tokens, {total} steps, vspread={args.lambda_vspread} ---")
    student.train()
    t0, step, micro = time.time(), 0, 0
    acc = {"loss": 0.0, "lm": 0.0, "kd": 0.0, "vs": 0.0}
    opt.zero_grad(set_to_none=True)
    for batch, seen in token_budget_batches(train_blocks, args.token_budget, args.micro_batch, seed=args.seed):
        ids = batch.to(dev)
        with torch.no_grad():
            tl = teacher(ids).logits[:, :-1].float()[..., :vocab]
        spread.on = args.lambda_vspread > 0
        sl = student(ids).logits[:, :-1].float()[..., :vocab]
        spread.on = False
        lm = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), ids[:, 1:].reshape(-1))
        kd = F.kl_div(F.log_softmax(sl / T, -1), F.log_softmax(tl / T, -1),
                      reduction="batchmean", log_target=True) * (T * T)
        loss = args.lambda_lm * lm + args.lambda_kd * kd
        vs = spread.penalty() if args.lambda_vspread > 0 else None
        if vs is not None:
            loss = loss + args.lambda_vspread * vs
        if not math.isfinite(float(loss.detach())):
            raise SystemExit(f"\n[DIVERGED] loss={float(loss)} at step {step}. Lower --lr-backbone. "
                             "Report it; do not clamp it.")
        (loss / args.grad_accum).backward()
        for k, v in (("loss", loss), ("lm", lm), ("kd", kd), ("vs", vs if vs is not None else torch.zeros(()))):
            acc[k] += float(v.detach())
        micro += 1
        if micro % args.grad_accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % args.log_every == 0 or step == 1:
                d = args.grad_accum * (args.log_every if step > 1 else 1)
                logger.log(step=step, tokens_seen=seen, loss=acc["loss"] / d, lm_loss=acc["lm"] / d,
                           kd_loss=acc["kd"] / d, transition_loss=acc["vs"] / d,
                           lr=sched.get_last_lr()[0], grad_norm=float(gn), peak_gpu_gb=peak_memory_gb(dev))
                print(f"    step {step:5d}/{total} tok {seen:>10,}  loss {acc['loss'] / d:.4f}  "
                      f"lm {acc['lm'] / d:.4f}  kd {acc['kd'] / d:.4f}  vspread {acc['vs'] / d:.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
                acc = {k: 0.0 for k in acc}
    train_s = time.time() - t0
    logger.close()

    # -------- after: paired vs teacher, spread, absorption -------------------
    print("\n--- after training (paired, per block, vs the untouched teacher) ---")
    student.eval()
    after = {}
    for L, blocks in eval_sets.items():
        sn, ssd = block_nlls(student, blocks, dev, vocab, spread)
        after[L] = sn
        pr = paired(sn, base[L])
        res["after"][L] = pr
        res["vstd"].setdefault("student_after", {})[L] = ssd
        tsd = res["vstd"]["teacher"][L]
        med = lambda d: float(np.median(list(d.values()))) if d else float("nan")
        print(f"  L={L:>5}  ppl {np.exp(sn).mean():9.4f}   d vs teacher {pr['d_ppl']:+.4f} +/- "
              f"{pr['two_sem']:.4f}  {pr['verdict']:20s}  std(log v) median layer: "
              f"teacher {med(tsd):.3f} -> student {med(ssd):.3f}")

    fold = None
    if not args.control:
        fold = fold_and_check(student, eval_sets[args.eval_lengths[0]][:1].to(dev), vocab)
        print(f"\n[absorb] folded {fold['folded']} constants into in_proj; "
              f"max relative logit change {fold['max_rel_logit_change']:.2e}  "
              f"(the folded model has NO pre-norm left: zero levels)")

    metrics = {
        "kind": kind, "stage": args.stage, "control": args.control,
        "paired_vs_teacher": res, "fold_check": fold,
        "per_block_nll": {"teacher": {L: base[L].tolist() for L in base},
                          "before": {L: start[L].tolist() for L in start},
                          "after": {L: after[L].tolist() for L in after}},
        "n_trainable": n_tr, "frozen": frozen, "optimiser_steps": step,
        "training_hours": train_s / 3600, "peak_gpu_memory_gb": peak_memory_gb(dev),
        "c_init": stats, "learned_c": ({k: float(m.c) for k, m in repl.items()} if repl else None),
    }
    save_json(args.outdir / "metrics.json", metrics)
    save_json(args.outdir / "config.json",
              {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
              | {"backend": backend, "train_info": tinfo})
    print(f"\nwrote {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
