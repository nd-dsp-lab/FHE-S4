#!/usr/bin/env python
"""PHASE 0 -- how big is the eval noise floor?

    python eval/noise_floor.py --shards 8 --lengths 512 2048

WHY THIS COMES FIRST
Every quality claim in this project is a single perplexity number on one
evaluation set. GATES.md records differences of +0.0002 (exp), +0.278
(softplus), +1.69 (SiLU-norm) and +3.95 (SiLU-conv) and treats them as
comparable. But we have never measured how much perplexity moves between
DISJOINT SHARDS OF THE SAME CORPUS with the model untouched. Without that,
"+0.278" might be noise and "+1.69" might be nothing.

This evaluates the UNTOUCHED pretrained model on K disjoint shards and reports
the spread. Everything downstream is then judged against

    SIGNIFICANT  <=>  |delta ppl| > 2 * sigma

where sigma is the per-shard standard deviation measured here, at that
sequence length.

Shards are disjoint in TOKEN space and no evaluation block straddles a shard
boundary, so the shards are genuinely independent samples of the corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# eval/ is a subdirectory, so the project root has to be importable when this is
# run as `python eval/noise_floor.py` rather than as a module.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baby_mamba.transition import ExactExp
from real_mamba.data import get_tokenizer, load_text
from real_mamba.model import DEFAULT_MODEL, load_model, recommended_dtype
from real_mamba.patch import patched


@torch.no_grad()
def shard_nll(model, blocks, device, vocab_size, batch_size=1):
    """Total NLL and token count over `blocks`, so shards combine correctly."""
    tot_nll, tot_tok = 0.0, 0
    for i in range(0, blocks.shape[0], batch_size):
        ids = blocks[i:i + batch_size].to(device)
        logits = model(ids).logits[:, :-1].float()[..., :vocab_size]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              ids[:, 1:].reshape(-1), reduction="sum")
        tot_nll += float(nll)
        tot_tok += ids[:, 1:].numel()
    return tot_nll, tot_tok


def make_shards(ids, n_shards, seq_len):
    """Disjoint token ranges -> (n_blocks, seq_len) per shard, no straddling."""
    per = len(ids) // n_shards
    out = []
    for k in range(n_shards):
        chunk = ids[k * per:(k + 1) * per]
        nb = len(chunk) // seq_len
        if nb == 0:
            continue
        t = torch.tensor(chunk[: nb * seq_len], dtype=torch.long).view(nb, seq_len)
        out.append(t)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default="auto", choices=("auto", "official", "local"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32",
                    choices=("auto", "float32", "bfloat16", "float16"))
    ap.add_argument("--data", default="wikitext2")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eval/noise_floor.json"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dtype = (recommended_dtype(args.device) if args.dtype == "auto"
             else getattr(torch, args.dtype))
    model, cfg, backend = load_model(args.model, args.backend, args.device, dtype)

    tok = get_tokenizer()
    text = load_text(args.data, args.split)
    ids = tok(text, return_tensors=None)["input_ids"]
    print(f"\n[noise floor] {args.data}/{args.split}: {len(ids):,} tokens "
          f"-> {args.shards} disjoint shards of {len(ids)//args.shards:,}")

    results = {}
    with patched(model, ExactExp(), chunk_size=args.chunk_size):
        for L in args.lengths:
            shards = make_shards(ids, args.shards, L)
            if len(shards) < 2:
                print(f"  L={L}: not enough tokens for {args.shards} shards; skipped")
                continue
            ppls, t0 = [], time.time()
            for k, blocks in enumerate(shards):
                n, t = shard_nll(model, blocks, args.device, cfg.vocab_size,
                                 args.batch_size)
                ppl = math.exp(n / t)
                ppls.append(ppl)
                print(f"\r  L={L:>5} shard {k + 1}/{len(shards)}  "
                      f"{blocks.shape[0]:>3} blocks  ppl {ppl:8.4f}  "
                      f"({time.time() - t0:.0f}s)", end="", flush=True)
            print()
            mean = statistics.fmean(ppls)
            sd = statistics.stdev(ppls) if len(ppls) > 1 else float("nan")
            results[str(L)] = {
                "seq_len": L, "n_shards": len(shards),
                "blocks_per_shard": [int(b.shape[0]) for b in shards],
                "tokens_per_shard": [int(b.numel()) for b in shards],
                "per_shard_ppl": ppls,
                "mean_ppl": mean, "std_ppl": sd,
                "two_sigma": 2 * sd,
                "sem": sd / math.sqrt(len(ppls)) if len(ppls) > 1 else float("nan"),
                "min_ppl": min(ppls), "max_ppl": max(ppls),
                "range_ppl": max(ppls) - min(ppls),
            }

    payload = {"config": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
               "backend": backend, "n_tokens_total": len(ids),
               "results": results}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    # ---------------- the table that everything else is judged against -------
    print()
    print("=" * 92)
    print("EVAL NOISE FLOOR -- untouched pretrained model, disjoint shards of the same corpus")
    print("=" * 92)
    hdr = (f"{'L':>6} {'shards':>7} {'mean ppl':>10} {'std':>9} {'2 sigma':>9} "
           f"{'min':>9} {'max':>9} {'range':>9}")
    print(hdr); print("-" * len(hdr))
    for L, r in results.items():
        print(f"{r['seq_len']:>6} {r['n_shards']:>7} {r['mean_ppl']:10.4f} "
              f"{r['std_ppl']:9.4f} {r['two_sigma']:9.4f} {r['min_ppl']:9.4f} "
              f"{r['max_ppl']:9.4f} {r['range_ppl']:9.4f}")
    print("-" * len(hdr))
    print()
    print("SIGNIFICANT means |delta ppl| > 2 sigma at that length. Applying that to")
    print("the differences already recorded in GATES.md:")
    recorded = [("exp", 0.0002), ("softplus (squared)", 0.278),
                ("SiLU in gated norm", 1.69), ("SiLU after conv1d", 3.95)]
    for L, r in results.items():
        ts = r["two_sigma"]
        print(f"\n  at L={r['seq_len']}  (2 sigma = {ts:.4f} ppl)")
        for name, d in recorded:
            verdict = "SIGNIFICANT" if abs(d) > ts else "within noise -- NOT significant"
            print(f"    {name:22s} {d:+8.4f}   {verdict}")
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
