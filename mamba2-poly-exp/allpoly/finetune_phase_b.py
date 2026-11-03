#!/usr/bin/env python
"""Phase B: fine-tune the WHOLE model with polynomial operators installed, then
re-measure the ranges, refit, and judge the result against a control.

    python -m allpoly.finetune_phase_b --ops softplus --degree-softplus-q 5    # polynomial arm
    python -m allpoly.finetune_phase_b --control                               # its control
    python -m allpoly.compare_phase_b runs/phase_b/<poly run> runs/phase_b/<control run>

    python -m allpoly.finetune_phase_b --smoke --ops softplus                  # CPU pipeline check

THE QUESTION. Phase A found the cheapest degree that works with NO training
(softplus Q=13, 6 levels). Phase B asks whether fine-tuning lets a CHEAPER
polynomial reach the same accuracy. Cost comes from OpenFHE's depth table, not the
degree itself: Q in 3..5 costs 5 levels with the square, 6..13 costs 6 -- so for
softplus the only cheaper tier is Q <= 5, and Q=5 is the best polynomial in it.

ONE MODEL, BUILT UP (allpoly/README.md):

    stage 1   --ops softplus                                   (init: the original model)
    stage 2   --ops softplus silu_norm  --init-from <stage 1>  (stage 1's refit ranges carry over)
    stage 3   --ops softplus silu_norm silu_conv --init-from <stage 2>

  `--ops` lists EVERY polynomial in the model at that stage, not only the new one.
  The control for stage k starts from the control of stage k-1 (--control --init-from).

WHAT ONE RUN DOES, and why each step is there:
  1. Fit each polynomial in --ops to the ranges the starting model actually sees
     (Phase A's calibration for stage 1). Measure before you fit.
  2. Evaluate the starting model with those polynomials ("before": for stage 1 this
     reproduces Phase A's only:softplus row -- check it does).
  3. Fine-tune with the polynomials IN the forward pass, so the weights adapt to
     them. The polynomial coefficients are fixed buffers; gradients flow THROUGH
     the polynomial into the backbone. Trained: everything except A_log, dt_bias
     (training them once pushed the decay factor above 1 on 2-4% of tokens) and
     the embedding (tied to lm_head, held by the client). Same as norm/absorb_ft.py.
  4. RE-MEASURE the ranges on the trained model, with the polynomials active (the
     configuration that will be deployed), and REFIT. Training moves the ranges; a
     polynomial left on its old interval is how every blow-up in this repo started.
  5. Evaluate, paired per shard on wikitext-2 validation, exactly as Phase A does:
       after_refit   trained model + refitted polynomials   <- THE result
       after_trainfit  trained model + the fit it trained with  (did the refit matter?)
       after_exact   trained model with every operator exact   (what training did alone)
  6. The control (--control) runs steps 3 and 5 on the exact model with the identical
     recipe. Fine-tuning on wikitext lowers wikitext perplexity by itself; only
     poly-vs-control says what the polynomial costs (allpoly/compare_phase_b.py).

OVERFITTING. Only a small part of the model is replaced, so the fine-tune could
fit the training text around it instead of learning to live with the polynomial.
Every --heldout-every steps the run records the loss on a fixed sample of
TRAINING blocks and on held-out wikitext-2 TEST blocks; a widening gap is
overfitting. The verdict itself is on wikitext-2 validation, which training never sees.

FHE-FRIENDLINESS. Training is plaintext, so an improvement could come from
something a circuit cannot do cheaply. This script adds nothing to the forward:
no comparisons, no branches on activation values, no clamps. Off-interval inputs
are evaluated anyway and COUNTED (frac_out). The only operations in the evaluated
path are the model's own linear layers, the convolution, the scan, and the
installed Chebyshev polynomials.

LOSS. Identical to norm/absorb_ft.py, so the two tracks compare:
    loss = lm * CE + kd * T^2 KL(student/T || teacher/T)
NOTE inherited from there: KL uses reduction="batchmean" on a batch of ONE
sequence, so it is SUMMED over its ~511 positions while CE is a mean. The KD
term therefore dominates. Kept on purpose for comparability; say so when reporting.
"""
from __future__ import annotations

import argparse
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from allpoly.eval_allpoly import ALL_SITES, shard_mean_nll
from allpoly.forward import (ExactOps, PolyOps, allpoly_forward, calibrate,
                             load_calibration, save_calibration)
from allpoly.ops import Refresher
from eval.noise_floor import make_shards
from real_mamba.data import get_blocks, get_tokenizer, load_text
from real_mamba.eval_lm import peak_memory_gb, reset_peak_memory
from real_mamba.model import load_model
from real_mamba.train_utils import RunLogger, save_json, set_seed, token_budget_batches

# Operators this script can install. The three RMSNorm sites are a separate track
# (norm/), and recording ranges with them active is not implemented in PolyOps.
TRAINABLE_OPS = ("exp", "softplus", "silu_conv", "silu_norm")
FROZEN_SUFFIXES = ("A_log", "dt_bias")
# which calibration entry each operator is fitted to
RANGE_KEY = {"softplus": "softplus_in", "silu_conv": "silu_conv_in",
             "silu_norm": "silu_norm_in", "exp": "z"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def poly_kwargs(args) -> dict:
    return dict(degree_silu_conv=args.degree_silu_conv, degree_silu_norm=args.degree_silu_norm,
                degree_softplus_q=args.degree_softplus_q, degree_exp=args.degree_exp,
                margin=args.margin)


def make_ops(calib, which, args, device):
    """The operator set for a run: PolyOps for the polynomial arm, ExactOps otherwise."""
    if not which:
        return ExactOps(record=False)
    return PolyOps(calib, which, **poly_kwargs(args)).to(device)


@torch.no_grad()
def recalibrate_deployed(model, calib_old, which, args, blocks, device, chunk_size):
    """Re-measure every operator's input range on `model`, WITH the polynomials in
    `which` active (fitted to `calib_old`), plus every refresh point's max |x|.

    Why with them active: in the deployed circuit the softplus polynomial in layer
    3 changes what layer 4's operators see. Ranges measured on the exact model would
    be ranges of a model nobody runs. (The refit then moves the polynomials slightly,
    which moves downstream ranges again -- a second-order effect the margin covers;
    frac_out after the refit is the check.)
    """
    rec = make_ops(calib_old, which, args, device)
    rec.record = True
    rec.stats = defaultdict(dict)
    rec.absmax = {}
    for i in range(blocks.shape[0]):
        allpoly_forward(model, blocks[i:i + 1].to(device), rec, refresh=rec.refresh,
                        chunk_size=chunk_size)
    rec.record = False
    return rec


def range_drift(calib_old, calib_new, which, margin) -> dict:
    """How far training moved each operator's input range.

    Per op, over all layers and channels: the median and max of new width / old
    width, and the fraction of channels whose NEW range leaves the interval the old
    polynomial was FITTED on (old range widened by `margin`, as PolyOps does). Those
    channels are the ones where keeping the old fit would evaluate a polynomial
    outside its interval -- the reason for the refit.
    """
    out = {}
    for op in which:
        key = RANGE_KEY[op]
        old, new = calib_old.stats.get(key, {}), calib_new.stats.get(key, {})
        ratios, escaped, n = [], 0, 0
        for l in sorted(set(old) & set(new)):
            olo, ohi = old[l]["min"].double(), old[l]["max"].double()
            nlo, nhi = new[l]["min"].double(), new[l]["max"].double()
            w_old = (ohi - olo).abs() + 1e-12
            ratios.append(((nhi - nlo) / w_old).numpy())
            pad = margin * w_old
            escaped += int(((nlo < olo - pad) | (nhi > ohi + pad)).sum())
            n += int(olo.numel())
        if not ratios:
            continue
        r = np.concatenate(ratios)
        out[op] = {"width_ratio_median": float(np.median(r)), "width_ratio_max": float(r.max()),
                   "width_ratio_min": float(r.min()), "frac_channels_outside_old_fit": escaped / max(n, 1)}
    return out


@torch.no_grad()
def mean_nll(model, ops, blocks, device, vocab, chunk_size):
    """Token-weighted mean NLL over `blocks` (the overfitting monitor)."""
    tot, n = 0.0, 0
    for i in range(blocks.shape[0]):
        ids = blocks[i:i + 1].to(device)
        lg = allpoly_forward(model, ids, ops, chunk_size=chunk_size)[:, :-1, :vocab]
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"))
        n += ids[:, 1:].numel()
    return tot / max(n, 1)


def paired(nll_a, nll_b) -> dict:
    """ppl(a) - ppl(b), shard by shard: mean and 2 SEM (Phase A's test)."""
    d = np.exp(np.asarray(nll_a)) - np.exp(np.asarray(nll_b))
    if not np.all(np.isfinite(d)):
        return {"d_ppl": float("inf"), "two_sem": float("nan"), "verdict": "BROKEN (inf/nan)"}
    m = float(d.mean())
    sem = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"d_ppl": m, "two_sem": 2 * sem,
            "verdict": "not distinguishable" if abs(m) <= 2 * sem else "SIGNIFICANT"}


@torch.no_grad()
def evaluate(model, ops, refresh, shard_sets, device, vocab, chunk_size, name, base=None):
    """Per-shard NLL at every length, with decay and off-interval reports."""
    res = {}
    for L, shards in shard_sets.items():
        if isinstance(ops, PolyOps):
            ops.reset_counts()
        else:
            ops.reset_decay()
        ops.track_a = True
        t0 = time.time()
        nll = [shard_mean_nll(model, ops, refresh, b, device, vocab, chunk_size) for b in shards]
        r = {"per_shard_nll": [float(x) for x in nll], "ppl_mean": float(np.exp(nll).mean()),
             "decay": ops.decay_report(), "seconds": round(time.time() - t0, 1)}
        if isinstance(ops, PolyOps):
            r["operators"] = ops.report()
        if base is not None:
            r["vs_teacher"] = paired(nll, base[L])
        res[L] = r
        dr = r["decay"]
        fo = max((v.get("frac_out_max", 0.0) for v in r.get("operators", {}).values()), default=0.0)
        vt = r.get("vs_teacher")
        vs = (f"d vs teacher {vt['d_ppl']:+9.4f} +/- {vt['two_sem']:.4f}  {vt['verdict']:20s}"
              if vt else " " * 56)
        print(f"  {name:16s} L={L:>5}  ppl {r['ppl_mean']:9.4f}  {vs}  frac_out {fo:.1e}  "
              f"a>1 {dr['frac_a_gt_1']:.1e} (max {dr['a_max']:.3g})  a<0 {dr['frac_a_lt_0']:.1e}  "
              f"({r['seconds']:.0f}s)", flush=True)
    ops.track_a = False
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="local", choices=("local", "official", "auto"),
                    help="only loads the weights; every forward here is allpoly_forward")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # what is installed
    ap.add_argument("--ops", nargs="+", default=["softplus"], choices=TRAINABLE_OPS,
                    help="EVERY polynomial operator in the model at this stage")
    ap.add_argument("--control", action="store_true",
                    help="install nothing: the exact model trained with the identical recipe")
    ap.add_argument("--degree-softplus-q", type=int, default=5,
                    help="degree of Q in Delta = Q(x)^2. Q<=5 costs 5 levels, 6..13 cost 6 (Phase A: Q13)")
    ap.add_argument("--degree-silu-conv", type=int, default=7)
    ap.add_argument("--degree-silu-norm", type=int, default=13)
    ap.add_argument("--degree-exp", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.05, help="interval widening, fraction of width (Phase A: 0.05)")
    # where the run starts
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a previous Phase B run dir: start from its model.pt (and, for a polynomial "
                         "arm, its refit_calib.pt)")
    ap.add_argument("--calib-cache", type=Path, default=None,
                    help="ranges to fit to at the start. Default: <init-from>/refit_calib.pt, else "
                         "Phase A's allpoly/results/calib.pt (computed and saved if missing)")
    ap.add_argument("--calib-blocks", type=int, default=64, help="Phase A: 64")
    ap.add_argument("--calib-len", type=int, default=2048, help="Phase A: 2048")
    # training (defaults = norm/absorb_ft.py, at the lr that worked there)
    ap.add_argument("--token-budget", type=int, default=1_000_000)
    ap.add_argument("--train-data", default="wikitext103")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--train-seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-backbone", type=float, default=1e-4,
                    help="absorption round 1: 1e-4 cost 3x less than 2e-5 (PROGRESS.md); "
                         "1e-3 diverged in an earlier full fine-tune")
    ap.add_argument("--lambda-lm", type=float, default=1.0)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="gradient-norm clipping in the OPTIMISER, not the forward: changes nothing "
                         "a circuit computes")
    ap.add_argument("--chunk-size", type=int, default=128, help="scan chunk size for training")
    # overfitting monitor
    ap.add_argument("--heldout-every", type=int, default=25, help="optimiser steps between checks")
    ap.add_argument("--heldout-blocks", type=int, default=8)
    # evaluation (defaults = Phase A)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--max-blocks-per-shard", type=int, default=None)
    ap.add_argument("--eval-chunk-size", type=int, default=256)
    ap.add_argument("--eval-noise-bits", type=float, nargs="*", default=[],
                    help="also evaluate after_refit with bootstrap noise (needs 'exp' in --ops: exact "
                         "exp + noise overflows, see allpoly/README.md)")
    ap.add_argument("--no-save-model", action="store_true",
                    help="skip model.pt (~0.5 GB). Without it the next stage cannot start from this run")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU check: 2x128-token calibration, a few training steps, L=64. "
                         "Verdicts MEAN NOTHING; it passes if it ends with SMOKE TEST PASSED")
    args = ap.parse_args(argv)

    which = () if args.control else tuple(dict.fromkeys(args.ops))
    if args.eval_noise_bits and "exp" not in which:
        ap.error("--eval-noise-bits needs 'exp' in --ops (exact exp + noise overflows)")
    if args.smoke:
        args.calib_blocks, args.calib_len = 2, 128
        args.train_data = "wikitext2"
        args.token_budget, args.train_seq_len, args.grad_accum = 4 * 128, 128, 2
        args.chunk_size = args.eval_chunk_size = 32
        args.shards, args.lengths, args.max_blocks_per_shard = 2, [64], 2
        args.heldout_every, args.heldout_blocks, args.log_every = 1, 2, 1

    deg = {"softplus": f"q{args.degree_softplus_q}", "silu_conv": f"d{args.degree_silu_conv}",
           "silu_norm": f"d{args.degree_silu_norm}", "exp": f"d{args.degree_exp}"}
    kind = "control" if args.control else "poly_" + "+".join(f"{o}{deg[o]}" for o in which)
    tag = f"{kind}_lr{args.lr_backbone:g}_{args.token_budget}tok" + ("_smoke" if args.smoke else "")
    if args.init_from is not None:
        tag += f"__from_{args.init_from.name}"
    args.outdir = args.outdir or Path("runs/phase_b") / tag
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dev = args.device
    if args.smoke:
        print("=" * 88)
        print("SMOKE TEST: tiny calibration and a few training steps. This only checks that the")
        print("pipeline RUNS. Inputs fall outside the fitted ranges, so the numbers MEAN NOTHING.")
        print("=" * 88)
    print(f"[phase B] {kind}   outdir {args.outdir}")

    # -------- models ---------------------------------------------------------
    # fp32 throughout: polynomial error must not be confounded with fp16 rounding.
    student, cfg, backend = load_model(backend=args.backend, device=dev, dtype=torch.float32)
    teacher, _, _ = load_model(backend=args.backend, device=dev, dtype=torch.float32)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if args.init_from is not None:
        sd = torch.load(args.init_from / "model.pt", map_location=dev, weights_only=True)
        student.load_state_dict(sd)
        print(f"[init] student starts from {args.init_from}/model.pt; the teacher is the ORIGINAL model")
    vocab = cfg.vocab_size

    # -------- data -----------------------------------------------------------
    train_blocks, _, tinfo = get_blocks(args.train_data, args.train_split, args.train_seq_len, verbose=False)
    heldout = get_blocks("wikitext2", "test", args.train_seq_len, verbose=False)[0][: args.heldout_blocks]
    g = torch.Generator().manual_seed(args.seed + 1)
    train_probe = train_blocks[torch.randperm(train_blocks.shape[0], generator=g)[: args.heldout_blocks]]
    cblocks = get_blocks("wikitext2", "train", args.calib_len, verbose=False)[0][: args.calib_blocks]
    val_ids = get_tokenizer()(load_text("wikitext2", "validation"), return_tensors=None)["input_ids"]
    shard_sets = {}
    for L in args.lengths:
        s = make_shards(val_ids, args.shards, L)
        shard_sets[L] = [x[: args.max_blocks_per_shard] for x in s] if args.max_blocks_per_shard else s

    # -------- fit at the start ----------------------------------------------
    calib0 = None
    if which:
        cpath = args.calib_cache
        if cpath is None:
            cpath = (args.init_from / "refit_calib.pt") if args.init_from else Path("allpoly/results/calib.pt")
            if args.smoke and args.init_from is None:
                cpath = args.outdir / "calib_smoke.pt"
        if cpath.exists():
            calib0 = load_calibration(cpath)
            print(f"[calib] fitting to {cpath}")
        else:
            if args.init_from is not None:
                raise SystemExit(f"{cpath} not found: a later stage must fit to the ranges its starting "
                                 "model sees. Point --calib-cache at the previous stage's refit_calib.pt")
            print(f"[calib] {cpath} not found: calibrating the exact model on "
                  f"{args.calib_blocks} x {args.calib_len} wikitext-2 train tokens (Phase A's recipe)")
            calib0 = calibrate(student, cblocks, dev, chunk_size=args.eval_chunk_size)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            save_calibration(calib0, cpath)
        shutil.copy(cpath, args.outdir / "start_calib.pt")
    ops_train = make_ops(calib0, which, args, dev)
    ops_train.track_a = False

    # -------- baselines ------------------------------------------------------
    metrics = {"kind": kind, "control": args.control, "ops": list(which),
               "degrees": {o: deg[o] for o in which}, "eval": {}}
    print("\n--- teacher (original model, every operator exact) ---")
    t_ops = ExactOps(record=False)
    base_eval = evaluate(teacher, t_ops, None, shard_sets, dev, vocab, args.eval_chunk_size, "teacher")
    base = {L: r["per_shard_nll"] for L, r in base_eval.items()}
    metrics["eval"]["teacher"] = base_eval
    if which or args.init_from is not None:
        print("--- before training: the starting model with this run's operators ---")
        metrics["eval"]["before"] = evaluate(student, ops_train, None, shard_sets, dev, vocab,
                                             args.eval_chunk_size, "before", base)
    if which:
        metrics["operators_at_start"] = {o: {"degree": r.get("degree"), "depth_openfhe": r.get("depth_openfhe")}
                                         for o, r in ops_train.report().items()}

    # -------- what trains ----------------------------------------------------
    trainable, frozen = [], []
    for n, p in student.named_parameters():
        if n.endswith(FROZEN_SUFFIXES) or "embedding" in n or n.startswith("lm_head"):
            p.requires_grad_(False)
            frozen.append(n)
        else:
            p.requires_grad_(True)
            trainable.append(p)
    assert not any(n.endswith(FROZEN_SUFFIXES) and p.requires_grad for n, p in student.named_parameters())
    n_tr = sum(p.numel() for p in trainable)
    print(f"\n[train] {n_tr:,} trainable; frozen: A_log, dt_bias, embedding/lm_head "
          f"({len(frozen)} tensors). Polynomial coefficients are fixed buffers.")
    opt = torch.optim.AdamW([{"params": trainable, "lr": args.lr_backbone}], betas=(0.9, 0.95),
                            weight_decay=0.0)
    tps = args.micro_batch * args.grad_accum * args.train_seq_len
    total = max(1, args.token_budget // tps)
    warm = max(1, int(args.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, total - warm), 1.0))))
    logger = RunLogger(args.outdir)
    reset_peak_memory(dev)
    T = args.temperature

    monitor = []

    def check_overfit(step, seen):
        student.eval()
        tr = mean_nll(student, ops_train, train_probe, dev, vocab, args.chunk_size)
        ho = mean_nll(student, ops_train, heldout, dev, vocab, args.chunk_size)
        student.train()
        monitor.append({"step": step, "tokens_seen": seen, "train_nll": tr, "heldout_nll": ho,
                        "gap": ho - tr})
        print(f"    [overfit check] step {step:5d}  train-sample nll {tr:.4f}  held-out nll {ho:.4f}  "
              f"gap {ho - tr:+.4f}", flush=True)

    # -------- train ----------------------------------------------------------
    print(f"\n--- training {args.token_budget:,} tokens, {total} steps, lr {args.lr_backbone:g} ---")
    check_overfit(0, 0)
    student.train()
    t0, step, micro, seen = time.time(), 0, 0, 0
    acc = {"loss": 0.0, "lm": 0.0, "kd": 0.0}
    opt.zero_grad(set_to_none=True)
    for batch, seen in token_budget_batches(train_blocks, args.token_budget, args.micro_batch, seed=args.seed):
        ids = batch.to(dev)
        with torch.no_grad():
            tl = allpoly_forward(teacher, ids, t_ops, chunk_size=args.chunk_size)[:, :-1, :vocab]
        sl = allpoly_forward(student, ids, ops_train, chunk_size=args.chunk_size)[:, :-1, :vocab]
        lm = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), ids[:, 1:].reshape(-1))
        kd = F.kl_div(F.log_softmax(sl / T, -1), F.log_softmax(tl / T, -1),
                      reduction="batchmean", log_target=True) * (T * T)
        loss = args.lambda_lm * lm + args.lambda_kd * kd
        if not math.isfinite(float(loss.detach())):
            raise SystemExit(f"\n[DIVERGED] loss={float(loss)} at step {step}. Lower --lr-backbone, or "
                             "check frac_out in 'before'. Report it; do not clamp it.")
        (loss / args.grad_accum).backward()
        for k, v in (("loss", loss), ("lm", lm), ("kd", kd)):
            acc[k] += float(v.detach())
        micro += 1
        if micro % args.grad_accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % args.log_every == 0 or step == 1:
                d = args.grad_accum * (args.log_every if step > 1 else 1)
                logger.log(step=step, tokens_seen=seen, loss=acc["loss"] / d, lm_loss=acc["lm"] / d,
                           kd_loss=acc["kd"] / d, lr=sched.get_last_lr()[0], grad_norm=float(gn),
                           peak_gpu_gb=peak_memory_gb(dev))
                print(f"    step {step:5d}/{total} tok {seen:>10,}  loss {acc['loss'] / d:.4f}  "
                      f"lm {acc['lm'] / d:.4f}  kd {acc['kd'] / d:.4f}  ({time.time() - t0:.0f}s)", flush=True)
                acc = {k: 0.0 for k in acc}
            if step % args.heldout_every == 0 and step < total:
                check_overfit(step, seen)
    check_overfit(step, seen)
    train_s = time.time() - t0
    logger.close()
    student.eval()

    # -------- re-measure, refit, evaluate ------------------------------------
    print("\n--- after training (paired per shard vs the untouched teacher) ---")
    if which:
        print(f"[refit] re-measuring ranges on the trained model, polynomials active "
              f"({args.calib_blocks} x {args.calib_len} wikitext-2 train tokens)")
        calib1 = recalibrate_deployed(student, calib0, which, args, cblocks, dev, args.eval_chunk_size)
        save_calibration(calib1, args.outdir / "refit_calib.pt")
        drift = range_drift(calib0, calib1, which, args.margin)
        metrics["range_drift"] = drift
        for o, d in drift.items():
            print(f"  [drift] {o:10s} width new/old median {d['width_ratio_median']:.3f} "
                  f"(min {d['width_ratio_min']:.3f}, max {d['width_ratio_max']:.3f}); channels whose new "
                  f"range leaves the OLD FIT's interval: {d['frac_channels_outside_old_fit']:.1%}")
        ops_refit = make_ops(calib1, which, args, dev)
        metrics["eval"]["after_refit"] = evaluate(student, ops_refit, None, shard_sets, dev, vocab,
                                                  args.eval_chunk_size, "after_refit", base)
        metrics["eval"]["after_trainfit"] = evaluate(student, ops_train, None, shard_sets, dev, vocab,
                                                     args.eval_chunk_size, "after_trainfit", base)
        metrics["eval"]["after_exact"] = evaluate(student, ExactOps(record=False), None, shard_sets, dev,
                                                  vocab, args.eval_chunk_size, "after_exact", base)
        for b in args.eval_noise_bits:
            rf = Refresher(b, calib1.absmax, ALL_SITES, seed=1)
            metrics["eval"][f"after_refit+noise@{b:g}"] = evaluate(
                student, ops_refit, rf, shard_sets, dev, vocab, args.eval_chunk_size,
                f"refit+noise@{b:g}", base)
    else:
        metrics["eval"]["after"] = evaluate(student, ExactOps(record=False), None, shard_sets, dev, vocab,
                                            args.eval_chunk_size, "after (control)", base)

    metrics.update({
        "overfit_monitor": monitor, "n_trainable": n_tr, "frozen": frozen, "optimiser_steps": step,
        "tokens_seen": seen, "training_hours": train_s / 3600, "peak_gpu_memory_gb": peak_memory_gb(dev),
        "shards": {str(L): [int(x.shape[0]) for x in s] for L, s in shard_sets.items()},
    })
    save_json(args.outdir / "metrics.json", metrics)
    save_json(args.outdir / "config.json",
              {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
              | {"backend": backend, "train_info": tinfo, "kind": kind})
    if not args.no_save_model:
        torch.save(student.state_dict(), args.outdir / "model.pt")
    print(f"\nwrote {args.outdir}/" + ("   <- SMOKE TEST PASSED (pipeline runs)" if args.smoke else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
