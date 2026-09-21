# Mamba-2: FHE-friendly replacement for `exp(A·Δ)`

Work lives in **[`mamba2-poly-exp/`](mamba2-poly-exp/)**. Start with
[`mamba2-poly-exp/HANDOFF.md`](mamba2-poly-exp/HANDOFF.md).

## What it is

Mamba-2 gates its recurrent memory with `a = exp(A·Δ)`, where `Δ` is
input-dependent and therefore a ciphertext under FHE. CKKS can only add and
multiply, so `exp` has to go. This replaces it with a **degree-4 polynomial, one
per attention head**, at **ct-ct depth 2**, with **no retraining**.

## Result

Measured on the official `state-spaces/mamba2-*` checkpoints:

| test | degree-4, per-head |
|---|---|
| 4 model scales, 130m → 1.3b | Δ perplexity ≤ 0.006 |
| held-out domains (The Pile, LAMBADA), intervals fitted on wikitext-103 only | +0.013 / +0.011 |
| sequence length to 8192 | −0.012 |
| LAMBADA zero-shot accuracy | −0.0013 (1 example in 800) |
| paired bootstrap, 95% CI | **indistinguishable from `exp`** |
| verification vs the official Triton kernel | rel. error 2.9e−04 |
| a single **global** approximation interval | **inf at every scale** |

That last row is the methodological point: the interval is a **per-head** quantity
that has to be **measured**. `|A|` spans 5–7 orders of magnitude across heads
within one model, so the obvious choice of `[-8, 0]` for all heads fails
completely. Because `A` is a *weight*, per-head coefficients are plaintext and
therefore free — same degree, same circuit, same depth.

## Documents

| | |
|---|---|
| [`HANDOFF.md`](mamba2-poly-exp/HANDOFF.md) | model sizes and params, exactly what was replaced, code map, traps, open items |
| [`FINDINGS.md`](mamba2-poly-exp/FINDINGS.md) | the evidence, the statistics, and what this does *not* show |
| [`README.md`](mamba2-poly-exp/README.md) | stage-by-stage walkthrough, written for undergraduate researchers |
| [`PART0_SOURCE_INSPECTION.md`](mamba2-poly-exp/PART0_SOURCE_INSPECTION.md) | what the official Mamba-2 source actually does, with line numbers |
| [`PROGRESS.md`](mamba2-poly-exp/PROGRESS.md) | chronological log: implemented / learned / still uncertain, per milestone |

## Status

No FHE implementation yet — "depth 2" is a cost proxy from the evaluation graph,
with no CKKS parameters, noise budget or latency behind it. That is the natural
next step and the largest gap in the story. Other open items are listed in
`HANDOFF.md` §8.
