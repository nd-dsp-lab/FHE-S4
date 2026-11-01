#!/usr/bin/env python
"""Hyunjung's SCOTUS experiment, rebuilt on Mamba-2 with Brandon's replacements in
the same slots her F + RKC occupies.

    python -m scotus.train --stage teacher     # 24-layer mamba2-130m fine-tuned on SCOTUS
    python -m scotus.train --stage student     # 12-layer student: exact control vs Brandon's gates

WHAT IS REPLACED (her scope, our operators). She replaced softplus, A, the exp in
the discretization, and the input term. In Mamba-2:
    softplus -> Delta = Q(x)^2, fitted per head           (Brandon; --degree-softplus-q)
    exp(dt*A) -> degree-4 polynomial per head, P(0) = 1   (Brandon's solved gate)
    A, input term Delta*B*x -> unchanged (A is one scalar per head; the input term
                               is already polynomial)
Both SiLUs and RMSNorm stay exact, as in her model.

PROTOCOL (1014_Meeting.pdf sec. 2.2 and her newer pipeline):
    teacher : head warm-up, then full fine-tune, CE, select by validation accuracy
    student : layers 0,2,...,22 of the teacher, new head (N(0,0.02^2), seed 0)
              head warm-up -> KD 1 epoch (exact)
              -> CONVERT: calibrate ranges on 256 training documents, install the gates
              -> KD 2 epochs (gates in)
    control : the same student from the same post-KD-1 snapshot, KD 2 epochs EXACT.
              Same total training, so the difference is the gates, not the training.
    training-free row: accuracy right after CONVERT, before any further training.
KD loss 0.5 CE + 0.5 T^2 KL (T = 2) against cached teacher logits. AdamW, linear
warm-up over 6% then linear decay, batch 1 x grad-accum 8, per-group clip 1.0.

DELIBERATE DEVIATIONS, flagged rather than hidden:
  * A_log and dt_bias stay FROZEN (she trains b_dt and A). Training them shifts each
    head's range off its fitted interval; it pushed decay factors above 1 on 2-4% of
    tokens before. The embedding is also frozen (tied to lm_head; the client holds it).
  * fp32 throughout (she used fp16 autocast with an fp32 scan).
  * The teacher recipe is provisional: her PDF does not give it. Defaults below.

VERDICT: her table (val / test accuracy), plus a paired test on the same test
documents -- McNemar on per-document correctness, and the mean per-document NLL
difference -- because at 1,400 documents accuracy alone cannot resolve differences
below ~1.2 points.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from allpoly.forward import PolyOps, calibrate
from real_mamba.model import load_model
from scotus import data
from scotus.model import Classifier, subset_layers

FROZEN_SUFFIXES = ("A_log", "dt_bias")


# --------------------------------------------------------------------- helpers
def param_groups(clf, lr, wd, train_head_only=False):
    """Her Table 3: matrices (>= 2 dims) with weight decay; biases / norm weights / D
    without. A_log, dt_bias and the embedding are frozen (see module docstring)."""
    dec, nodec = [], []
    for n, p in clf.named_parameters():
        frozen = (n.endswith(FROZEN_SUFFIXES) or "embedding" in n or ".lm_head." in n
                  or n.startswith("lm.lm_head"))
        if train_head_only and not n.startswith("head."):
            frozen = True
        p.requires_grad_(not frozen)
        if frozen:
            continue
        (dec if p.dim() >= 2 else nodec).append(p)
    return [{"params": dec, "lr": lr, "weight_decay": wd},
            {"params": nodec, "lr": lr, "weight_decay": 0.0}]


def linear_schedule(opt, total, warm_frac=0.06):
    warm = max(1, int(warm_frac * total))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))


@torch.no_grad()
def predict(clf, ids, last, dev, bs=4):
    clf.eval()
    out = []
    for i in range(0, ids.shape[0], bs):
        out.append(clf(ids[i:i + bs].to(dev), last[i:i + bs].to(dev)).float().cpu())
    return torch.cat(out)


def scores(logits, y):
    pred = logits.argmax(-1)
    nll = F.cross_entropy(logits, y, reduction="none")
    return {"acc": float((pred == y).float().mean()), "nll": float(nll.mean()),
            "correct": (pred == y).tolist(), "doc_nll": nll.tolist()}


def mcnemar(a_correct, b_correct):
    """Exact two-sided McNemar on paired per-document correctness."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only_correct": n01, "b_only_correct": n10, "p": p}


def paired_nll(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / math.sqrt(len(d))
    return {"mean_d_nll": float(d.mean()), "two_se": float(2 * se),
            "verdict": "not distinguishable" if abs(d.mean()) <= 2 * se else "SIGNIFICANT"}


def train(clf, ids, last, y, dev, *, epochs, lr, wd=0.01, head_only=False, teacher_logits=None,
          kd_lambda=0.5, T=2.0, accum=8, clip=1.0, seed=0, tag=""):
    groups = param_groups(clf, lr, wd, train_head_only=head_only)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    n = ids.shape[0]
    total = max(1, epochs * n // accum)
    sched = linear_schedule(opt, total)
    g = torch.Generator().manual_seed(seed)
    clf.train()
    step, t0, run = 0, time.time(), 0.0
    for ep in range(epochs):
        order = torch.randperm(n, generator=g)
        for j, i in enumerate(order.tolist()):
            logits = clf(ids[i:i + 1].to(dev), last[i:i + 1].to(dev))
            ce = F.cross_entropy(logits, y[i:i + 1].to(dev))
            if teacher_logits is not None:
                tl = teacher_logits[i:i + 1].to(dev)
                kl = F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(tl / T, -1),
                              reduction="batchmean", log_target=True) * T * T
                loss = (1 - kd_lambda) * ce + kd_lambda * kl
            else:
                loss = ce
            lv = float(loss.detach())
            if not math.isfinite(lv):
                raise SystemExit(f"[{tag}] DIVERGED at epoch {ep} doc {j}: loss={lv}. Report it; do not clamp it.")
            (loss / accum).backward()
            run += lv
            if (j + 1) % accum == 0:
                for gr in groups:
                    if gr["params"]:
                        torch.nn.utils.clip_grad_norm_(gr["params"], clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0:
                    print(f"    [{tag}] ep {ep} step {step}/{total} loss {run / (50 * accum):.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    run = 0.0
    return clf


def evaluate(clf, sets, dev, label):
    r = {}
    for name, (ids, last, y) in sets.items():
        lg = predict(clf, ids, last, dev)
        r[name] = scores(lg, y)
    print(f"  {label:44s} val {100 * r['val']['acc']:6.2f}   test {100 * r['test']['acc']:6.2f}", flush=True)
    return r


# --------------------------------------------------------------------- stages
def stage_teacher(a, dev, sets, train_set):
    lm, cfg, _ = load_model(backend=a.backend, device=dev, dtype=torch.float32)
    clf = Classifier(lm, data.N_CLASSES, chunk_size=a.chunk_size, seed=a.seed)
    ids, last, y = train_set
    print("\n--- teacher: head warm-up ---")
    train(clf, ids, last, y, dev, epochs=a.head_epochs, lr=1e-3, head_only=True, tag="head")
    best, best_acc = None, -1.0
    for ep in range(a.teacher_epochs):
        print(f"\n--- teacher: full fine-tune epoch {ep + 1}/{a.teacher_epochs} ---")
        train(clf, ids, last, y, dev, epochs=1, lr=a.lr, seed=a.seed + ep, tag="teacher")
        r = evaluate(clf, sets, dev, f"teacher after epoch {ep + 1}")
        if r["val"]["acc"] > best_acc:
            best_acc, best = r["val"]["acc"], copy.deepcopy(clf.state_dict())
            res = r
    clf.load_state_dict(best)
    torch.save({"state": best, "layers": len(lm.backbone.layers)}, a.outdir / "teacher.pt")
    return res


def stage_student(a, dev, sets, train_set):
    ids, last, y = train_set
    lm, cfg, _ = load_model(backend=a.backend, device=dev, dtype=torch.float32)
    teacher = Classifier(lm, data.N_CLASSES, chunk_size=a.chunk_size, seed=a.seed)
    ck = torch.load(a.outdir / "teacher.pt", map_location=dev)
    teacher.load_state_dict(ck["state"])
    out = {"teacher": evaluate(teacher, sets, dev, "teacher (24 layers, exact)")}
    print("\n--- caching teacher logits on the training set ---")
    t_logits = predict(teacher, ids, last, dev)

    keep = list(range(0, len(lm.backbone.layers), 2))
    student = Classifier(subset_layers(teacher.lm, keep), data.N_CLASSES,
                         chunk_size=a.chunk_size, seed=a.seed)
    del teacher, lm
    print(f"\n--- student: {len(keep)} layers {keep}; head warm-up ---")
    train(student, ids, last, y, dev, epochs=a.head_epochs, lr=1e-3, head_only=True, tag="s-head")
    print("\n--- student: KD 1 epoch (exact) ---")
    train(student, ids, last, y, dev, epochs=1, lr=a.lr, teacher_logits=t_logits, tag="s-kd1")
    out["student_exact_kd1"] = evaluate(student, sets, dev, "student exact, after KD 1")
    snap = copy.deepcopy(student.state_dict())

    # ---- CONVERT: calibrate on 256 training documents, install Brandon's gates
    print(f"\n--- convert: calibrating on {min(a.calib_docs, ids.shape[0])} training documents ---")
    cal = calibrate(student.lm, ids[: a.calib_docs], dev, chunk_size=a.chunk_size)
    poly = PolyOps(cal, ("exp", "softplus"), degree_softplus_q=a.degree_softplus_q,
                   degree_exp=4).to(dev)
    student.ops = poly
    poly.reset_counts()
    out["student_gates_training_free"] = evaluate(student, sets, dev, "student + gates, NO further training")
    out["gates_report_training_free"] = {"operators": poly.report(), "decay": poly.decay_report()}

    print("\n--- student + gates: KD 2 epochs ---")
    train(student, ids, last, y, dev, epochs=2, lr=a.lr, teacher_logits=t_logits, tag="s-gates")
    poly.reset_counts()
    out["student_gates_kd3"] = evaluate(student, sets, dev, "student + gates, after KD 1+2")
    out["gates_report"] = {"operators": poly.report(), "decay": poly.decay_report()}

    # ---- CONTROL: same snapshot, same 2 epochs, exact operators
    print("\n--- control: exact student, KD 2 more epochs from the same snapshot ---")
    from allpoly.forward import ExactOps
    student.load_state_dict(snap)
    student.ops = ExactOps()
    train(student, ids, last, y, dev, epochs=2, lr=a.lr, teacher_logits=t_logits, tag="s-ctrl")
    out["student_exact_kd3_control"] = evaluate(student, sets, dev, "student exact (control), after KD 1+2")

    g, c = out["student_gates_kd3"]["test"], out["student_exact_kd3_control"]["test"]
    out["verdict_test"] = {"gates_minus_control_acc": g["acc"] - c["acc"],
                           "mcnemar": mcnemar(g["correct"], c["correct"]),
                           "paired_nll": paired_nll(g["doc_nll"], c["doc_nll"])}
    v = out["verdict_test"]
    print(f"\n  VERDICT (test, gates vs control): acc {100 * v['gates_minus_control_acc']:+.2f} pts, "
          f"McNemar p={v['mcnemar']['p']:.3g}, per-doc NLL {v['paired_nll']['mean_d_nll']:+.4f} "
          f"+/- {v['paired_nll']['two_se']:.4f} ({v['paired_nll']['verdict']})")
    d = out["gates_report"]["decay"]
    print(f"  decay safety (test+val, gates in): frac a>1 {d['frac_a_gt_1']:.1e}, a<0 {d['frac_a_lt_0']:.1e}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("teacher", "student"), required=True)
    ap.add_argument("--backend", default="local", choices=("auto", "official", "local"),
                    help="local = pure PyTorch, same official weights; what allpoly and Phase A use")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--teacher-epochs", type=int, default=3, help="PROVISIONAL: her teacher recipe is not in the PDF")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--degree-softplus-q", type=int, default=13,
                    help="degree of Q in Delta = Q(x)^2. Set to Brandon's Phase A choice")
    ap.add_argument("--calib-docs", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-eval", type=int, default=None, help="smoke tests only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("runs/scotus"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = a.device
    cache = a.outdir / "data"
    train_set = data.get("train", a.seq_len, cache, a.max_train)
    sets = {"val": data.get("validation", a.seq_len, cache, a.max_eval),
            "test": data.get("test", a.seq_len, cache, a.max_eval)}
    print(f"[scotus] train {train_set[0].shape[0]}  val {sets['val'][0].shape[0]}  "
          f"test {sets['test'][0].shape[0]}  L={a.seq_len}")
    t0 = time.time()
    res = stage_teacher(a, dev, sets, train_set) if a.stage == "teacher" else stage_student(a, dev, sets, train_set)
    strip = lambda r: {k: {s: {m: v for m, v in d.items() if m not in ("correct", "doc_nll")}
                           for s, d in r[k].items()} if isinstance(r[k], dict) and "val" in r[k] else r[k]
                       for k in r}
    (a.outdir / f"{a.stage}_metrics.json").write_text(json.dumps(strip(res), indent=1, default=float))
    torch.save(res, a.outdir / f"{a.stage}_full.pt")         # per-document results, for re-analysis
    print(f"\n[scotus] {a.stage} done in {(time.time() - t0) / 60:.1f} min -> {a.outdir}")


if __name__ == "__main__":
    main()
