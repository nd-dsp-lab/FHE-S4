#!/usr/bin/env python
"""E3/E4/E5/E6 -- the checks a reviewer asks for after the headline number.

    python robustness_suite.py --experiments longctx lambada bootstrap margin

E3 longctx   perplexity at 1024 / 2048 / 4096 / 8192. Mamba's whole selling point
             is long context, and Part 4 showed gate error COMPOUNDS with length.
             Part 8 checked state norms; this checks the thing people quote.
E4 lambada   zero-shot last-word accuracy. Perplexity is an average and can hide
             damage concentrated on the tokens that matter. Accuracy on a
             long-range-dependency task cannot.
E5 margin    how wide must the per-head interval be? Sweeps --interval-margin and
             reports where it breaks. Tells us whether 0.25 was lucky.
E6 bootstrap is +0.035 perplexity even real? Resamples eval blocks to put a
             confidence interval on the polynomial-vs-exact difference, using
             PAIRED per-block losses so the model-to-model comparison is exact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baby_mamba.transition import ExactExp
from real_mamba.data import get_blocks, get_tokenizer, load_docs
from real_mamba.model import DEFAULT_MODEL, load_model, recommended_dtype
from real_mamba.patch import patch_transition, set_transition
from real_mamba.transitions import build_per_head_transitions


# --------------------------------------------------------------------------
@torch.no_grad()
def per_block_nll(model, blocks, device, vocab_size, batch_size=4):
    """Mean NLL for each block separately -- needed for a PAIRED comparison."""
    out = []
    for i in range(0, blocks.shape[0], batch_size):
        ids = blocks[i:i + batch_size].to(device)
        logits = model(ids).logits[:, :-1].float()[..., :vocab_size]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              ids[:, 1:].reshape(-1), reduction="none")
        out.append(nll.view(ids.shape[0], -1).mean(dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def lambada_accuracy(model, tok, device, vocab_size, max_examples=800, batch_size=8):
    """Zero-shot LAMBADA: does the model predict the FINAL WORD of the passage?

    Greedy over the whole final word (which may be several tokens), scored by
    teacher forcing. This is the standard setup and it is perplexity-independent:
    a model can keep its average loss and still stop getting the last word right.
    """
    docs = load_docs("lambada", "test", max_docs=max_examples)
    correct = total = 0
    nll_last, ntok_last = 0.0, 0
    for d in docs:
        text = d.strip()
        if " " not in text:
            continue
        ctx, last = text.rsplit(" ", 1)
        ctx_ids = tok(ctx, return_tensors=None)["input_ids"]
        tgt_ids = tok(" " + last, return_tensors=None)["input_ids"]
        if not tgt_ids or len(ctx_ids) + len(tgt_ids) > 1024:
            continue
        ids = torch.tensor([ctx_ids + tgt_ids], device=device)
        logits = model(ids).logits[0].float()[..., :vocab_size]
        # positions predicting the target tokens
        start = len(ctx_ids) - 1
        pred = logits[start:start + len(tgt_ids)].argmax(-1)
        tgt = torch.tensor(tgt_ids, device=device)
        correct += int(bool((pred == tgt).all()))
        lp = F.cross_entropy(logits[start:start + len(tgt_ids)], tgt, reduction="sum")
        nll_last += float(lp); ntok_last += len(tgt_ids)
        total += 1
    return {"lambada_accuracy": correct / max(total, 1),
            "lambada_n": total,
            "lambada_last_word_ppl": math.exp(nll_last / max(ntok_last, 1))}


def bootstrap_ci(a: torch.Tensor, b: torch.Tensor, n_boot=10000, seed=0):
    """Paired bootstrap on per-block NLL differences (b - a), in nats."""
    rng = np.random.default_rng(seed)
    d = (b - a).numpy()
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean_nll_diff": float(d.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
            "frac_boot_favouring_poly": float((means < 0).mean()),
            "n_blocks": int(len(d))}


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=("auto", "float32", "bfloat16", "float16"))
    ap.add_argument("--stats-json", type=Path,
                    default=Path("runs/e1_cross_domain/stats_fitted_on_wikitext103.json"))
    ap.add_argument("--experiments", nargs="+",
                    default=["longctx", "lambada", "bootstrap", "margin"],
                    choices=["longctx", "lambada", "bootstrap", "margin"])
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--eval-data", default="wikitext2")
    ap.add_argument("--eval-blocks", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--lambada-examples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("runs/e3456_robustness"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)
    tok = get_tokenizer()
    if not args.stats_json.exists():
        raise SystemExit(f"need {args.stats_json}; run cross_domain_eval.py or "
                         f"collect_transition_stats.py first")
    polys = {d: build_per_head_transitions(args.stats_json, d, pin_zero=True,
                                           margin=args.margin)
             for d in args.degrees}
    handle = patch_transition(model, ExactExp(), chunk_size=args.chunk_size)
    results = {}

    # ---- E3: long context ------------------------------------------------
    if "longctx" in args.experiments:
        print("\n=== E3: perplexity vs sequence length ===")
        rows = []
        for L in args.lengths:
            try:
                blocks, _, _ = get_blocks(args.eval_data, "validation", L, verbose=False)
            except ValueError as e:
                print(f"  L={L}: {e}"); continue
            blocks = blocks[: max(4, args.eval_blocks // (L // 1024))]
            set_transition(handle, ExactExp())
            base = float(per_block_nll(model, blocks, args.device, cfg.vocab_size,
                                       args.batch_size).mean())
            row = {"seq_len": L, "n_blocks": int(blocks.shape[0]),
                   "exact_ppl": math.exp(base)}
            for d, t in polys.items():
                set_transition(handle, t)
                v = float(per_block_nll(model, blocks, args.device, cfg.vocab_size,
                                        args.batch_size).mean())
                row[f"poly{d}_ppl"] = math.exp(v) if v < 700 else float("inf")
                row[f"poly{d}_delta_ppl"] = row[f"poly{d}_ppl"] - row["exact_ppl"]
            rows.append(row)
            print(f"  L={L:>5}  exact {row['exact_ppl']:8.4f}  " +
                  "  ".join(f"poly{d} {row[f'poly{d}_ppl']:8.4f} "
                            f"({row[f'poly{d}_delta_ppl']:+.4f})" for d in polys))
        results["longctx"] = rows

    # ---- E4: LAMBADA accuracy --------------------------------------------
    if "lambada" in args.experiments:
        print("\n=== E4: LAMBADA zero-shot accuracy ===")
        rows = []
        set_transition(handle, ExactExp())
        base = lambada_accuracy(model, tok, args.device, cfg.vocab_size,
                                args.lambada_examples)
        base["transition"] = "exact"; rows.append(base)
        print(f"  exact : acc {base['lambada_accuracy']:.4f}  "
              f"last-word ppl {base['lambada_last_word_ppl']:.3f}  (n={base['lambada_n']})")
        for d, t in polys.items():
            set_transition(handle, t)
            r = lambada_accuracy(model, tok, args.device, cfg.vocab_size,
                                 args.lambada_examples)
            r["transition"] = f"poly{d}"
            r["delta_accuracy"] = r["lambada_accuracy"] - base["lambada_accuracy"]
            rows.append(r)
            print(f"  poly{d} : acc {r['lambada_accuracy']:.4f} "
                  f"({r['delta_accuracy']:+.4f})  last-word ppl "
                  f"{r['lambada_last_word_ppl']:.3f}")
        results["lambada"] = rows

    # ---- E6: paired bootstrap --------------------------------------------
    if "bootstrap" in args.experiments:
        print("\n=== E6: is the difference real? paired bootstrap over eval blocks ===")
        blocks, _, _ = get_blocks(args.eval_data, "validation", 1024, verbose=False)
        blocks = blocks[: args.eval_blocks]
        set_transition(handle, ExactExp())
        a = per_block_nll(model, blocks, args.device, cfg.vocab_size, args.batch_size)
        rows = []
        for d, t in polys.items():
            set_transition(handle, t)
            b = per_block_nll(model, blocks, args.device, cfg.vocab_size, args.batch_size)
            ci = bootstrap_ci(a, b, seed=args.seed); ci["transition"] = f"poly{d}"
            rows.append(ci)
            sig = "NOT distinguishable from exact" if ci["ci95_low"] < 0 < ci["ci95_high"] \
                else "significantly different"
            print(f"  poly{d} : mean NLL diff {ci['mean_nll_diff']:+.6f} nats  "
                  f"95% CI [{ci['ci95_low']:+.6f}, {ci['ci95_high']:+.6f}]  -> {sig}")
        results["bootstrap"] = rows

    # ---- E5: margin sensitivity ------------------------------------------
    if "margin" in args.experiments:
        print("\n=== E5: how much interval margin do we need? ===")
        blocks, _, _ = get_blocks(args.eval_data, "validation", 1024, verbose=False)
        blocks = blocks[: args.eval_blocks]
        set_transition(handle, ExactExp())
        base = float(per_block_nll(model, blocks, args.device, cfg.vocab_size,
                                   args.batch_size).mean())
        rows = []
        for mg in args.margins:
            for d in args.degrees:
                t = build_per_head_transitions(args.stats_json, d, pin_zero=True, margin=mg)
                set_transition(handle, t)
                v = float(per_block_nll(model, blocks, args.device, cfg.vocab_size,
                                        args.batch_size).mean())
                ppl = math.exp(v) if v < 700 else float("inf")
                rows.append({"margin": mg, "degree": d, "ppl": ppl,
                             "delta_ppl": ppl - math.exp(base)})
                print(f"  margin {mg:<5} poly{d}  ppl {ppl:10.4f}  "
                      f"(delta {ppl - math.exp(base):+.4f})")
        results["margin"] = rows

    handle.restore()
    (args.outdir / "robustness.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.outdir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"backend": backend, "dtype": str(dtype)}, indent=2) + "\n")
    for name, rows in results.items():
        if rows:
            cols, seen = [], set()
            for r in rows:
                for k in r:
                    if k not in seen: seen.add(k); cols.append(k)
            with (args.outdir / f"{name}.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, restval="")
                w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
