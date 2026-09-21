"""Shared evaluation: negative log likelihood and perplexity, done honestly.

`evaluate()` is used by Part 7 (no training), Part 8 (stability), Part 9
(fine-tuning) and Part 10 (distillation), so every number in this project is
computed by the same code path.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, blocks, device="cpu", batch_size=1, vocab_size=None,
             progress=False, max_blocks=None, amp_dtype=None,
             nan_early_stop=True):
    """Token-level mean NLL (nats) and perplexity over `blocks`.

    blocks: (n_blocks, seq_len) int64. Each block is scored independently; we
    predict token t+1 from tokens <=t, so a block of length L contributes L-1
    predictions. Summing NLL over tokens and dividing once at the end (rather
    than averaging per-batch means) keeps the result independent of batch size.

    vocab_size: truncate logits to the true vocab (50277) before the softmax.
    The checkpoint pads to 50288; those 11 slots are untrained. Including them
    changes perplexity slightly, and excluding them is the convention. We are
    explicit about it rather than silent.

    nan_early_stop: once the accumulated NLL is non-finite it can never recover,
    so stop and report infinity. A diverged polynomial model would otherwise burn
    the whole eval set producing NaN. The result records where it stopped.
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    n = blocks.shape[0] if max_blocks is None else min(max_blocks, blocks.shape[0])
    t0 = time.time()
    for i in range(0, n, batch_size):
        ids = blocks[i: i + batch_size].to(device)
        ctx = (torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype)
               if amp_dtype is not None else torch.autocast(device_type="cpu", enabled=False))
        with ctx:
            logits = model(ids).logits
        logits = logits[:, :-1].float()
        if vocab_size is not None:
            logits = logits[..., :vocab_size]
        targets = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              targets.reshape(-1), reduction="sum")
        total_nll += float(nll)
        total_tokens += targets.numel()
        if nan_early_stop and not math.isfinite(total_nll):
            if progress:
                print(f"\n  [eval] STOPPED after {min(i + batch_size, n)}/{n} blocks: "
                      f"the accumulated NLL is not finite. Reporting ppl = inf.")
            return {"loss": float("nan"), "nll": float("nan"), "perplexity": float("inf"),
                    "n_tokens_scored": total_tokens, "n_blocks": min(i + batch_size, n),
                    "eval_seconds": time.time() - t0, "diverged": True}
        if progress:
            done = min(i + batch_size, n)
            print(f"\r  [eval] {done}/{n} blocks  "
                  f"nll={total_nll / total_tokens:.4f}  ppl={math.exp(total_nll / total_tokens):.3f}  "
                  f"({time.time() - t0:.0f}s)", end="", flush=True)
    if progress:
        print()
    mean_nll = total_nll / total_tokens
    # A diverged model can produce inf/nan NLL. Report it instead of crashing.
    ppl = math.exp(mean_nll) if math.isfinite(mean_nll) and mean_nll < 700 else float("inf")
    return {
        "loss": mean_nll,
        "nll": mean_nll,
        "perplexity": ppl,
        "n_tokens_scored": total_tokens,
        "n_blocks": n,
        "eval_seconds": time.time() - t0,
        "diverged": False,
    }


def peak_memory_gb(device="cpu") -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return float("nan")


def reset_peak_memory(device="cpu"):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
