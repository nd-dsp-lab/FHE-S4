# Can a low-degree polynomial replace `exp(A·Δ)` in Mamba-2?

**Yes — degree 4, one polynomial per head, ciphertext-ciphertext depth 2, with no
training at all.** This document is the evidence, including what it does not show.

Everything below is measured. Raw rows are in `runs/gpu/`; every number is
reproducible with the command given beside it.

---

## The claim

> Replacing Mamba-2's `a = exp(A·Δ)` with a **degree-4 polynomial, fitted per head
> on that head's own measured range of `z = A·Δ`**, changes the pretrained model's
> language-modelling quality by less than **0.014 perplexity** on every corpus,
> sequence length and model scale we tested, requires **no fine-tuning**, and costs
> **2 sequential ciphertext-ciphertext multiplications**.

The per-head part is not an optimisation, it is the whole result. A single global
interval gives **infinite perplexity at every model scale**.

---

## 1. It is the same function as the official kernel

Our scan is a reformulation (per-step products instead of exponentiated prefix
sums), so the first obligation is to show it computes what upstream computes.

| check | result |
|---|---|
| vs official `mamba_chunk_scan_combined`, chunk 64/128/256 | rel. error **2.97e−04 / 2.97e−04 / 2.96e−04** |
| with the `D` skip term | 2.88e−04 |
| at sequence length 2048 | 2.89e−04 |
| our chunk 32/64/128/256 vs their chunk 128 | 2.79e−04, identical to 3 s.f. |
| vs the official pure-PyTorch `ssd_minimal_discrete` | **1e−12** |
| vs a literal `for t in range(L)` loop, any transition | **1e−11** |

The 2.9e−04 is fp16 kernel arithmetic against an fp32 reference (the kernel
compiles in fp16 only on sm_75). A time-reversed control in the same test is >10×
worse, so the tolerance cannot be passing a wrong answer.

`qsub cluster/job_preflight.sh`

---

## 2. Why per-head, and why global fails

`A` is **one scalar per head**, and in every pretrained checkpoint those scalars
span 5–7 orders of magnitude:

| model | layers × heads | total `A` scalars | `\|A\|` range |
|---|---|---|---|
| mamba2-130m | 24 × 24 | 576 | 4.0e−04 … 3.6e+04 |
| mamba2-370m | 48 × 32 | 1,536 | 1.1e−03 … 2.3e+03 |
| mamba2-780m | 48 × 48 | 2,304 | 8.6e−04 … 3.4e+03 |
| mamba2-1.3b | 48 × 64 | 3,072 | 6.1e−04 … 9.8e+02 |

So `z = A·Δ` ranges from `[-0.004, 0]` for the gentlest head to `[-1.8e5, -0.47]`
for the most extreme, and the measured global minimum is **−263,700**. A degree-4
polynomial on `[-1.8e5, 0]` is numerically the zero function.

Since `A` is a **weight**, it is plaintext under FHE — so per-head coefficients are
free. Same degree, same circuit, same depth; different plaintext constants. All
576 polynomials fit in **0.04 s** on a laptop.

---

## 3. Four independent generalisation tests

### 3a. Model scale — `qsub cluster/job_e2_scale.sh`

wikitext-2, 100 × 1024 tokens, intervals measured per model.

| model | exact ppl | poly2 (depth 1) | poly3 (depth 2) | **poly4 (depth 2)** | global `[-8,0]` |
|---|---|---|---|---|---|
| 130m | 22.3548 | +0.0109 | −0.0035 | **−0.0008** | inf |
| 370m | 15.4460 | +0.0055 | −0.0033 | **−0.0007** | inf |
| 780m | 12.8280 | +0.0158 | +0.0021 | **+0.0058** | inf |
| 1.3b | 11.3820 | −0.0121 | −0.0073 | **−0.0014** | inf |

`frac(a > 1) = 0` at every scale and degree. The global interval fails at every
scale — this is not a 130m artefact.

### 3b. Held-out domains — `qsub cluster/job_e1_crossdomain.sh`

**Intervals fitted on wikitext-103 only.** No refitting on any evaluation corpus.

| corpus | exact ppl | poly2 | poly3 | **poly4** | `z` outside its interval |
|---|---|---|---|---|---|
| wikitext-2 | 22.3548 | +0.019 | +0.000 | **+0.001** | 2.9e−06 |
| **The Pile** (web, code, papers) | 11.6427 | +0.089 | +0.025 | **+0.013** | 1.6e−04 |
| **LAMBADA** (narrative) | 26.5562 | +0.202 | +0.050 | **+0.011** | 1.1e−04 |

Only ~1 token in 6,000 leaves its fitted interval even on The Pile. **This is where
degree 2 stops being defensible and degree 4 keeps working**: poly2 degrades 10×
out of domain (+0.019 → +0.202), poly4 stays within +0.013.

### 3c. Sequence length — `qsub cluster/job_e3456_robustness.sh`

| L | n blocks | exact | poly2 | poly3 | **poly4** |
|---|---|---|---|---|---|
| 1024 | 100 | 22.3548 | +0.0188 | +0.0000 | **+0.0009** |
| 2048 | 50 | 20.2484 | +0.0179 | −0.0002 | **+0.0008** |
| 4096 | 25 | 19.3892 | −0.0821 | −0.0073 | **+0.0002** |
| 8192 | 12 | 22.3613 | −0.9029 | −0.0874 | **−0.0122** |

poly4 holds to 8192. poly2 and poly3 drift, increasingly, with length — exactly the
compounding effect Part 4 predicted on the toy model. **Caveat: only 12 blocks at
L=8192, so those numbers are noisy; the *trend* is the finding, not the value.**

### 3d. A task, not just perplexity

LAMBADA zero-shot last-word accuracy, 800 examples:

| | accuracy | Δ | last-word ppl |
|---|---|---|---|
| exact | 0.4425 | — | 7.523 |
| poly2 | 0.4412 | −0.0013 | 7.560 |
| poly3 | 0.4412 | −0.0013 | 7.565 |
| **poly4** | **0.4412** | **−0.0013** | 7.543 |

One example in 800, identical across degrees. Perplexity is an average and can hide
damage concentrated on the tokens that matter; this is the check that it does not.

---

## 4. Is the difference even real?

Paired bootstrap over 100 eval blocks (10,000 resamples), per-block NLL in nats:

| | mean diff | 95% CI | verdict |
|---|---|---|---|
| poly2 | +0.000842 | [+0.00027, +0.00140] | **significantly different** from exact |
| poly3 | +0.000001 | [−0.00019, +0.00019] | not distinguishable |
| **poly4** | **+0.000042** | **[−0.000065, +0.00015]** | **not distinguishable** |

Degree 4 is statistically indistinguishable from the exponential it replaces.
Degree 2 is distinguishable — the effect is tiny (~0.02 perplexity) but it is real,
which is another reason the defensible claim is degree 4, not degree 2.

---

## 5. Is it sensitive to how we choose the interval?

Margin = how far the interval is widened beyond the measured range.

| margin | poly2 Δppl | poly3 Δppl | **poly4 Δppl** |
|---|---|---|---|
| 0.00 | +0.0026 | +0.0009 | **+0.0005** |
| 0.10 | +0.0071 | +0.0006 | **+0.0007** |
| 0.25 | +0.0188 | +0.0000 | **+0.0009** |
| 0.50 | +0.0547 | −0.0008 | **+0.0011** |
| 1.00 | +0.4381 | +0.0391 | **+0.0075** |

**poly4 varies by 0.007 perplexity across a 100× change in the margin.** poly2 varies
by 0.44. There is no hyperparameter to tune carefully at degree 4 — another reason
it is the defensible choice.

---

## 6. What this does NOT show

State these before anyone else does.

1. **No FHE implementation exists.** "ct-ct depth 2" is a cost *proxy* derived from
   the evaluation graph. No CKKS parameters, noise budget, bootstrap placement or
   latency has been measured. That is the obvious next project, not this one.
2. **`frac(a < 0) ≈ 0.008`.** Roughly 1 token in 120 gets a *negative* decay, which
   `exp` can never produce. It does not destabilise state norms up to L=2048 and
   does not move perplexity or LAMBADA accuracy, but it is a genuine departure and
   we do not have a mechanistic account of why the model tolerates it.
3. **Odd degrees have an unbounded negative tail.** poly3 reached `a = −130.7` on
   The Pile. It did not matter there, but degree 3 should be considered less safe
   than degree 4 despite the identical depth.
4. **Fine-tuning can break the intervals.** Training `A_log`/`dt_bias` moves `z`
   away from the range the polynomials were fitted on. MODE A survives
   (`frac(a>1) = 0`), but MODE B and MODE C push `frac(a>1)` to 4.4% and 2.1%.
   **The best perplexity we ever recorded (16.78, MODE C) has the second-worst-behaved
   gate.** Re-measuring and re-fitting after fine-tuning is the untested fix.
5. **English text only**, four model sizes, one architecture family, one seed for
   the interval measurement.
6. **L=8192 rests on 12 blocks.**

---

## 7. The bottom line

For the narrow question asked — can `exp(A·Δ)` be replaced by something a CKKS
circuit can evaluate, without wrecking the model — the answer is **yes, with
degree 4 and per-head intervals, at depth 2, with no training.**

The defensible version of the story is *not* "a quadratic is free". In-domain it
looks that way, and four independent tests (domain, length, statistics, margin
sensitivity) all say degree 2 is the point where the approximation starts to show.
**Degree 4 passes all four.** It costs one extra ciphertext-ciphertext
multiplication over degree 2 and no extra depth.

The single most important methodological point: the interval is a **per-head**
quantity that must be **measured**, not assumed. Assuming `[-8, 0]` — the natural
guess, and the one the project started with — gives infinite perplexity at every
model scale.
