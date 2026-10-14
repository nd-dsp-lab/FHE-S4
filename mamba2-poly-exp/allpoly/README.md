# allpoly — Mamba-2 with every non-polynomial operator replaced, in plaintext

Under FHE a server can only add and multiply. This package runs the 130M model
with each non-polynomial operator swapped for the polynomial an encrypted circuit
would actually evaluate, plus the noise a CKKS bootstrap adds. It answers, before
anyone writes encrypted code: **does the model still work, and at what cost in
levels?**

## What gets replaced

| operator | where | polynomial form | fitted per |
|---|---|---|---|
| `exp(A·Δ)` | the decay gate | degree 4, `P(0)=1` pinned | head (the solved one) |
| `softplus` | Δ, before the scan | `Δ = Q(x)²`: never negative | head |
| `SiLU` | after conv1d | Chebyshev, degree d | channel (1792) |
| `SiLU` | inside the gated norm | Chebyshev, degree d | channel (1536) |
| `1/√v` | the three RMSNorm sites | Chebyshev guess + Newton | layer |

`ops.py` has the forms; `forward.py` has the model forward written out explicitly
(`allpoly_forward`) and the fitting (`PolyOps`). In exact mode it reproduces
`model(ids).logits` to 8.6e-6 relative — `allpoly/tests/` checks this.

## How it measures (and why each step is there)

1. **Calibrate** on wikitext-2 **train** with the exact model: every operator's
   input range, per channel / head / layer. *Measure before you fit* — guessing a
   range is what produced every `inf` in this project.
2. **Fit** each polynomial to its measured range, in the **Chebyshev basis**
   (stable at any degree; the power basis breaks down in fp32 above ~6, and
   Chebyshev is also what OpenFHE evaluates).
3. **Evaluate** on wikitext-2 **validation**, as **paired** per-shard differences
   against the exact model on the same tokens. The unpaired spread is the wrong
   error bar for an operator swap — it is ~100x too loose.

## Phase A: run it

```bash
bash allpoly/run_phase_a.sh                                  # degree sweep, one operator at a time
SILU_D=27 SP_Q=13 bash allpoly/run_phase_a.sh combined       # your chosen degrees together, + noise
```

Knobs: `DEV=cuda`, `SHARDS=8`, `LENGTHS="512 2048"`, `PY=python`. If L=2048 runs
out of memory on your card, use `LENGTHS=512` and say so in your write-up.

Step 0 runs the exp gate alone. **It must come out "not distinguishable"**, with a
decay-above-1 FRACTION near the baselines below (the max will be larger) — that is the known answer (checked:
−0.0048 ± 0.0076). If it does not, stop: your setup differs and nothing after it
can be trusted.

Smoke test on a laptop (about a minute, tiny data): `python -m allpoly.eval_allpoly --device cpu --smoke`.
It only checks that the pipeline runs: it passes if it ends with `SMOKE TEST PASSED`.
Its verdicts mean nothing — 256 calibration tokens leave inputs outside the fitted ranges.

## Reading a result line

```
only:silu_conv   d ppl  +0.1234 +/- 0.0456  SIGNIFICANT   frac_out 0.0e+00  a>1 1.1e-07 (max 1.049)  a<0 8.2e-03
```

| field | meaning | what to watch for |
|---|---|---|
| `d ppl ± 2SEM` | paired perplexity change vs exact | the verdict |
| `frac_out` | fraction of inputs **outside** a fitted interval | should be ~0. A polynomial outside its interval is how every blow-up in this repo started |
| `a>1 (max)` | fraction of decay factors above 1, and the largest | above 1 the recurrence **expands**. Compare to the baselines below |
| `a<0` | fraction of negative decays | ~0.8% for the exp gate; report it, don't hide it |

**Decay-factor baselines.** Judge by the **fraction** of decays above 1, not the
maximum. The maximum is set by a single rare token whose input falls outside the
fitted interval, so it grows with how much text you evaluate:

| exp gate alone, L = 2048 | calibration | decays evaluated | frac(a > 1) | max a |
|---|---|---|---|---|
| small check | 32 × 2048 | ~9M | 1.1e-7 | 1.049 |
| 8× more text | 32 × 2048 | ~75M | 6.6e-8 | **17.3** |
| default calibration | **64** × 2048 (default) | ~75M | **1.3e-8** | 4.18 |
| + 11.35-bit noise | 32 × 2048 | ~9M | 5.5e-3 | 1.013 |

So a max of 4–20 on a full run is normal. More calibration data shrinks the fraction
(64 blocks: 5× lower than 32). frac(a < 0) is ~8e-3 in every row.

Bootstrap noise pushes ~0.5% of decays slightly above 1. Perplexity does not notice at
L = 2048 — whether it matters at longer lengths is open. Any operator that raises
these numbers is a problem even if its perplexity looks fine.

The JSON also has, per operator, the **OpenFHE depth** — from OpenFHE's own
degree→depth table, so a degree 3–5 polynomial costs 4 levels, 6–13 costs 5,
14–27 costs 6, 28–59 costs 7 (`Q²` adds 1). One bootstrap leaves **23 levels**
(fhe/EPOCH.md), so this is the cost side of every row.

## Noise levels

`--noise-bits 11.35 18.23` are the measured mean-abs precision after one bootstrap
at our parameters (N = 2^17, uniform / sparse secret). Noise goes in at each
refresh point: the residual stream entering every block, and the scan's inputs and
output. Each channel's noise is relative to that channel's own range,
as the per-slot scaling a real circuit does before bootstrapping would make it.

The noise-only baseline is `exp+noise`, not "exact + noise": with exact `exp`, a
tiny error on Δ where Δ≈0, multiplied by |A| (up to ~2.6e5), overflows `exp`. The
encrypted circuit never computes exact `exp`, so that configuration can't occur.

## Phase B (fine-tuning) builds ONE model

```
exact model
 → install softplus poly → fine-tune → check
 → install SiLU-norm poly (on that model) → fine-tune → check
 → install SiLU-conv poly → fine-tune → check
 = ONE model with every replacement
```

After each stage, re-calibrate and re-fit (training moves the ranges), and every
fine-tune needs a **control**: the exact model fine-tuned the same way.
`A_log` and `dt_bias` stay frozen — training them once pushed `a` above 1 on 2–4%
of tokens.
