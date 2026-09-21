# FHE-friendly Mamba-2: can a polynomial replace `exp(A·Δ)`?

An educational research prototype. One narrow question, answered step by step.

> **Original Mamba-2 computes an input-dependent memory decay using `exp(A * delta)`.
> Exponential is expensive under FHE.
> We therefore replace `exp` with a low-degree polynomial.
> A quadratic costs roughly one sequential encrypted multiplication level.
> Cubic/quartic polynomials can cost roughly two.
> We then ask whether the pretrained Mamba can adapt to this cheaper transition
> through lightweight fine-tuning or distillation.**

Nothing else about Mamba is redesigned. `SiLU`, `RMSNorm`, `softplus` and the scan
are untouched. No FHE library is used yet. No language model is pretrained.

---

## The answer, up front

**Yes — degree 4, one polynomial per head, ct-ct depth 2, with zero training.**

> **New to this project? Start with [HANDOFF.md](HANDOFF.md)** — model sizes,
> exactly what was replaced with what, the code map, the traps, and the open items.
>
> **[FINDINGS.md](FINDINGS.md) is the evidence document.** It has the four
> generalisation tests (model scale, held-out domain, sequence length, downstream
> task), the statistical test, the sensitivity analysis, and an explicit list of
> what the work does *not* show. Read that if you want the defensible claim rather
> than the walkthrough.

Headline, measured across 4 model scales (130m–1.3b), 3 corpora including two the
intervals never saw, and sequence lengths to 8192: **degree 4 changes perplexity by
less than 0.014 everywhere**, is statistically indistinguishable from `exp`
(bootstrap 95% CI spans 0), loses 1 LAMBADA example in 800, and varies by 0.007
perplexity across a 100× change in its only hyperparameter. A single global
interval gives **infinite perplexity at every model scale**.

Degree 2 (depth 1) works in-domain and is *not* defensible out of it — it degrades
10× on held-out domains and is statistically distinguishable from exact. That
distinction is the main thing the GPU campaign bought.

| transition | ct-ct depth | perplexity | Δ vs exact | verdict |
|---|---|---|---|---|
| exact `exp` | n/a | **22.2425** | — | the pretrained model |
| `poly2`, one global interval `[-8,0]` | 1 | **inf** | diverges | dead on arrival |
| `poly3`, one global interval `[-8,0]` | 2 | **inf** | diverges | dead on arrival |
| `poly4`, one global interval `[-8,0]` | 2 | **inf** | diverges | dead on arrival |
| `poly2`, per-head intervals | **1** | 22.2392 | **−0.0033** | indistinguishable |
| `poly3`, per-head intervals | 2 | 22.2392 | **−0.0033** | indistinguishable |
| `poly4`, per-head intervals | 2 | 22.2419 | **−0.0006** | indistinguishable |

Verified on GPU with the **official** `mamba-ssm` backend (2026-09-19): the same
12-block evaluation gives 18.9538 there vs 18.9532 from the pure-PyTorch loader
on a laptop CPU — a 3e-5 relative difference, i.e. the two backends agree.

*(wikitext-2 validation, 120 × 1024 = 122,880 tokens, `state-spaces/mamba2-130m`,
fp32, `--pin-zero`. Reproduce with `python eval_poly_exp.py --sweep --blocks 120`;
raw rows in `runs/part7_zero_training/results.csv`.)*

The per-head deltas are **−0.003 perplexity out of 22.24**, i.e. 0.015% — below
the noise of the eval set, and negative, so a depth-1 quadratic is not measurably
worse than the exponential it replaced. Meanwhile the same polynomials on a single
global interval produce `inf`.

The single finding that makes this work: **`A` is one scalar per head, and in the
pretrained checkpoint those 576 scalars span five orders of magnitude**
(`|A|` from 4.0e−04 to 3.6e+04). So `z = A·Δ` per head ranges from `[-0.004, 0]`
for the gentlest head to `[-1.8e5, -0.47]` for the most extreme. A single global
interval must cover the worst head, and a degree-4 polynomial on `[-1.8e5, 0]`
is literally the zero function.

But `A` is a **weight**, so under FHE it is plaintext — which means per-head
coefficients are free. Same degree, same depth, different plaintext constants.
Narrowing the interval is worth far more than raising the degree:

| interval | degree-4 max abs error | `P(0)` | fraction of interval where `P(z) < 0` |
|---|---|---|---|
| `[-8, 0]` | 3.4e−02 | 0.966 | 17% |
| `[-4, 0]` | 3.7e−03 | 0.996 | **0%** |
| `[-2, 0]` | 2.4e−04 | 0.9998 | 0% |
| `[-1, 0]` | 1.1e−05 | 0.99999 | 0% |

and **523 of 576 heads have `z_min ≥ -8`**, 322 of them `≥ -1`.

That is the result. The rest of this README is how to get there yourself, and
why each step exists.

---

## Install and check

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest baby_mamba/tests real_mamba/tests -q   # 63 pass in ~2 s; 6 need CUDA
```

Stages 0–8 run on a **laptop CPU** in about 15 minutes; Stages 9–10 want the GPU.
Every command below can be copied straight out of this README, or run the lot:

```bash
./run_all.sh                  # Stages 0-8
./run_all.sh --with-training   # also Stages 9-10
```

First time through, run them one at a time and read the output. The point of the
project is the numbers, not the script.

---

## Stage 0 — read the real source before writing any code

**What question are we asking?** Where, exactly, does Mamba-2 compute
`exp(A·Δ)`, and what shape is it?

**What should I run?** Nothing. Read
[`PART0_SOURCE_INSPECTION.md`](PART0_SOURCE_INSPECTION.md), with
[`third_party/mamba/`](third_party/mamba) open beside it.

**What should I expect to see?** Three surprises.

1. `A` is `(nheads,)` — **one scalar per head**, 24 numbers per layer, not one per
   state dimension (`mamba2.py:182`).
2. **The production code never computes `exp(A·Δ)` per timestep.** It computes
   `cumsum(A·Δ)` and exponentiates *differences of prefix sums*
   (`ssd_chunk_state.py:84-86`), exploiting `exp(Σz) = Π exp(z)`.
3. `use_mem_eff_path=False` does **not** un-fuse the model. Both branches call
   Triton. Even `softplus` is inside the kernel.

**Success:** you can point at the line and say what shape it is.
**Failure:** you assume the paper's notation is the code's notation. It is not,
and §0.2 of the report explains the discrepancy and which reading we take.

**Why does this matter for FHE?** Because `P(Σz) ≠ Π P(z)`. The cumsum trick that
makes the real kernel fast is *exactly* what a polynomial breaks. "Replace exp
with P" is ambiguous until you say which. We take the **per-step** reading —
`a_t = P(z_t)`, accumulate multiplicatively — because that is what an FHE
implementation must actually do.

---

## Stage 1 — see the transition with your own eyes

**What question are we asking?** What actually happens in the Mamba-2 state
transition, concretely, with numbers?

**What should I run?**

```bash
python -m baby_mamba.demo_transition
```

**What should I expect to see?** `u`, `A`, `delta`, `z = A·delta`, `a = exp(z)`,
and every recurrent state `h_t`, printed in full, for a model with
`batch=2, seqlen=8, nheads=4, headdim=2, d_state=2`. Then read
[`baby_mamba/transition.py`](baby_mamba/transition.py) — it is ~200 lines and
every step carries the upstream line number it mirrors.

The four sentences to take away:

> `A` controls how quickly memory decays.
> `delta` is input-dependent.
> `exp(A * delta)` becomes a number controlling how much previous state survives.
> Since input is private in FHE, this transition value becomes encrypted too.

**Success:** you can explain why `a = 1` means "remember everything" and `a = 0`
means "forget everything", and why `a` is a *scalar per head* that multiplies all
`headdim × d_state` numbers of that head's state at once.
**Failure:** the code still looks like magic. Re-read with the demo output beside it.

**Why does this matter for FHE?** `delta` is a function of the token, so under FHE
it is a ciphertext, so `z` is a ciphertext, so `exp` has to be evaluated
homomorphically — and CKKS can only add and multiply.

---

## Stage 2 — fit the polynomial

**What question are we asking?** How accurately can a degree-2/3/4 polynomial
approximate `exp` on a negative interval?

**What should I run?**

```bash
python fit_exp_polynomial.py --degree 4 --xmin -8 --xmax 0 --compare
python fit_exp_polynomial.py --degree 4 --xmin -4 --xmax 0 --method lobatto --pin-zero
```

Four fitting methods, none of them Taylor (Taylor's error grows like `|z|^(n+1)`,
which is precisely wrong on a wide interval):

* `chebyshev` — interpolation at Chebyshev roots. Near-minimax, never fails. **Default.**
* `lobatto` — nodes include the endpoints, so with `--xmax 0` you get `P(0) = 1` free.
* `remez` — true minimax, our own Remez exchange. Raises rather than returning garbage.
* `lstsq` — weighted least squares; the only one supporting `--weight relative`.

**What should I expect to see?** On `[-8, 0]`:

| degree | ct-ct depth | max abs err | mean abs err | RMSE | `P(0)` | frac `P<0` |
|---|---|---|---|---|---|---|
| 2 | 1 | 2.78e−01 | 6.76e−02 | 8.39e−02 | 0.7219 | 0.403 |
| 3 | 2 | 1.04e−01 | 3.02e−02 | 3.65e−02 | 0.8959 | 0.298 |
| 4 | 2 | 3.37e−02 | 1.10e−02 | 1.31e−02 | 0.9663 | 0.169 |

**Success:** two things in that table bother you.
`P(0) = 0.9663` instead of `1` looks like a 3% error — but `a` is multiplied once
per timestep, and `0.9663^1024 ≈ 5e-16`, so all long-range memory silently dies.
And `P(z) < 0` on 17% of the interval is something `exp` can never do: a negative
decay flips the sign of a head's entire state in one step.
**Failure:** you report "3% error" and move on.

**Why does this matter for FHE?** Degree sets the depth, and depth is the FHE
budget. But the *interval* is the real cost driver, and we do not know it yet.

---

## Stage 3 — count the encrypted multiplications honestly

**What question are we asking?** How many *sequential* ciphertext×ciphertext
multiplications does each polynomial cost?

**What should I run?** The depth is printed by `fit_exp_polynomial.py`, computed
from the actual evaluation graph in
[`baby_mamba/polynomial.py`](baby_mamba/polynomial.py):

```
z2 = z * z        # ciphertext x ciphertext, depth 1
z3 = z2 * z       # ciphertext x ciphertext, depth 2
z4 = z2 * z2      # ciphertext x ciphertext, depth 2
P4 = c0 + c1*z + c2*z2 + c3*z3 + c4*z4     # ciphertext x PLAINTEXT, no extra depth
```

**What should I expect to see?**

| degree | ct-ct mults | ct-ct depth | Horner's depth |
|---|---|---|---|
| 2 | 1 | **1** | 2 |
| 3 | 2 | **2** | 3 |
| 4 | 3 | **2** | 4 |

**We deliberately do not use Horner.** Horner uses one fewer multiplication, but
they are *sequential*: depth 4 instead of 2 at degree 4. Under CKKS, depth is
what forces larger parameters or a bootstrap; the raw multiplication count barely
matters. `PowerSchedule` builds the power graph as data and **computes** the depth
from it, so the claim is auditable instead of asserted in a comment; the tests
check `ceil(log2(degree))` for degrees 2–32.

**Success:** depth 2 at degree 4, and you can say why multiplying by `c_i` is free.
**Failure:** you write `P4` with Horner "because it's fewer multiplies".

**Why does this matter for FHE?** This is the entire cost model. One level of
ct-ct depth is the unit of currency.

---

## Stage 4 — watch a tiny gate error become a large model error

**What question are we asking?** Does pointwise approximation error predict what
happens inside a recurrence?

**What should I run?**

```bash
python -m baby_mamba.error_propagation
python -m baby_mamba.error_propagation --method lobatto --pin-zero
python -m baby_mamba.error_propagation --delta-scale 6
```

**What should I expect to see?** For degree 4 on `[-8, 0]`, the gate error is
*flat* and the state error *grows*:

| L | mean \|Δa\| | relative error of final state |
|---|---|---|
| 16 | 2.3e−02 | 6.2e−02 |
| 64 | 2.3e−02 | 4.9e−01 |
| 256 | 2.3e−02 | 4.6e−01 |
| 1024 | 2.3e−02 | **4.3e−01** |

7× amplification from L=16 to L=1024. Then `--pin-zero` (`P(0) = 1` enforced)
drops the L=1024 state error from **0.43 to 0.13** — 3.4× better at identical FHE
depth, even though its pointwise max error is slightly *worse* (4.0e−02 vs
3.4e−02). A pointwise-only evaluation would have rejected the better polynomial.

And the failure you must never walk into — `--delta-scale 6` pushes `z` to −24.75,
outside the fit interval:

| poly | min `a` at L=1024 | frac `a < 0` |
|---|---|---|
| P2 | −0.06 | 0.059 |
| P3 | **−45.8** | 0.099 |
| P4 | −0.01 | 0.011 |

`min a = −45.8` means one timestep multiplied a head's whole state by −45.8.
**A polynomial does not decay outside its fit interval; it diverges.**

> The polynomial does not merely approximate one activation.
> Its output controls how memory is repeatedly multiplied through time.
> Therefore long-sequence stability matters more than pointwise approximation
> error alone.

**Success:** you can predict that the interval must cover the real `z` range with
margin, and you want to go measure it.
**Failure:** you pick `[-8, 0]` because it was in the brief. Stage 6 shows what
that costs.

**Why does this matter for FHE?** There is no cheap rescue. `torch.clamp` would fix
`a < 0` instantly — and a clamp is a *comparison*, which under CKKS needs a
high-degree sign polynomial costing far more depth than the polynomial we are
trying to save. So we never clamp, anywhere (enforced by a test that scans the
source).

---

## Stage 5 — instrument the real pretrained model

**What question are we asking?** How do we see `z` and `a` inside the actual
130M checkpoint, without changing anything else?

**What should I run?** Nothing on its own; this stage is the machinery Stages 6–10
use. Read [`real_mamba/patch.py`](real_mamba/patch.py) and
[`real_mamba/reference_ssd.py`](real_mamba/reference_ssd.py).

**What should I expect to see?** `patch_transition` replaces `Mamba2.forward` with a
pure-PyTorch forward that recomputes the mixer **from that layer's own pretrained
parameters**, routing the scan through an SSD written in per-step *product* form —
every `cumsum` in `ssd_minimal.py` becomes a `cumprod`, every `exp(segsum(...))` a
`segprod(...)`. The only swappable object is a module mapping `z → a`.

It does **not** monkey-patch `torch.exp`. `A = -exp(A_log)` is an exp of a *weight*
— plaintext under FHE, therefore free — and is left exactly as upstream has it.

**Success:** `python -m pytest real_mamba/tests -q` passes. The load-bearing test
asserts our product form equals the **official** `ssd_minimal_discrete` to 1e−12
when the transition is `exp`, and equals a literal `for t in range(L)` loop to
1e−11 for *any* transition, including polynomials that go negative.
**Failure:** you trust the instrumentation without that test. Every number in
Stages 6–10 rests on it.

**Why does this matter for FHE?** We need the per-step `a` to exist as a tensor at
all. In the real kernel it does not.

**Two backends, one checkpoint.** `--backend official` uses
`MambaLMHeadModel.from_pretrained` from the installed `mamba-ssm` package (needs
CUDA + Triton); `--backend local` loads the *same* `pytorch_model.bin` into a
minimal module tree with byte-identical parameter names, so the project runs on a
laptop. `--backend auto` (default) tries official first. Sanity check: the local
backend gives wikitext-2 validation perplexity **17.60** on 6 blocks and **22.24**
on 120 — a plausible pretrained 130M model, not noise.

---

## Stage 6 — measure `z` on the real model. **Do this before choosing an interval.**

**What question are we asking?** Where does `z = A·Δ` actually live in the trained
model?

**What should I run?**

```bash
python collect_transition_stats.py --blocks 32 --seq-len 1024
```

The model is untouched here (`transition = exact exp`); we only observe. Streaming
Welford moments plus reservoir sampling, so nothing large is retained.

**What should I expect to see?** Something that invalidates `[-8, 0]`.

```
observed z min:          -179766.1
fraction(z < -4):         0.056
fraction(z < -8):         0.048
fraction(z < -20):        0.041
```

Per layer, `z_min` ranges from `-0.46` (layer 0) to `-179766` (layer 4). And
per head — the statistic that matters, because `A` is per head:

| \|z_min\| bucket | heads (of 576) |
|---|---|
| < 1 | 322 |
| 1 – 4 | 171 |
| 4 – 8 | 30 |
| 8 – 32 | 15 |
| 32 – 1000 | 21 |
| > 1000 | 17 |

**523 of 576 heads sit inside `[-8, 0]`. 53 do not, and 38 are catastrophically
outside it.** Those 38 are "instant-forget" heads: `|A| > 100`, so `exp(A·Δ) ≈ 0`
for essentially every token they see. One head has `exp(z) = 0` over its *entire*
observed range, so its exact optimal polynomial is the constant `0` — degree 0,
depth 0, error 3e−32. Not an approximation: the exact answer.

**Success:** you now know the interval is a *per-head* quantity, and you can see
why. **Failure:** you skip this stage. Stage 7 shows exactly what that costs.

**Why does this matter for FHE?** The interval is the dominant cost driver (Stage 2's
table), and it cannot be guessed. Measuring it first is the difference between
`inf` and parity.

---

## Stage 7 — drop the polynomial in, train nothing

**What question are we asking?** How much perplexity does the polynomial cost
before any adaptation?

**What should I run?**

```bash
python eval_poly_exp.py --sweep --blocks 120          # the whole table
python eval_poly_exp.py --transition poly4 --interval-mode per-head --pin-zero
```

`--interval-mode global` is the naive baseline (one interval for all 576 heads).
`--interval-mode per-head` uses Stage 6's measurements; it costs nothing extra
under FHE because `A` is plaintext.

**What should I expect to see?** The table at the top of this README.
Global `[-8, 0]` gives `inf` at every degree — `a` reaches 6e17 and 79% of
transition values are NaN. Per-head gives `−0.0033 / −0.0033 / −0.0006`
perplexity for degree 2 / 3 / 4, with **no training at all**.

One number in that table deserves a stare: per-head `poly3` reaches
`min a = −14.7`. One head, on some token, multiplied its entire state by −14.7 —
and the perplexity did not move. The affected heads are the `|A| > 100`
"instant-forget" heads whose output the rest of the network has evidently learned
to rely on very little. That is a genuine departure from `exp`, it is recorded
rather than clamped, and it is the main thing we would want to understand before
claiming this is safe in general.

**Success:** degree ≤ 4, depth ≤ 2, perplexity delta under ~0.05, no NaN.
**Failure:** `inf`, or a delta of several perplexity points, or `frac a > 1`
above a rounding error.

**Why does this matter for FHE?** This is the headline: a **depth-1** quadratic
costs 0.02 perplexity on a pretrained model with zero retraining.

**Implementation detail worth knowing.** Per-head polynomials are stored in a
*scale-normalised* basis: `P(z) = Σ c_k (z/s_h)^k` with `s_h = |xmin_h|` a
plaintext constant. In the raw `z` basis the coefficients span **1e−45 to 1**
across heads — some are fp32 subnormals, and `z⁴` at `z = -1.8e5` is 1e21. The
substitution is mathematically identical (`1/s_h` is a plaintext multiply, no
extra depth) but keeps every coefficient O(1). Stage 9 could not train at all
without it.

---

## Stage 8 — is it stable, or just lucky?

**What question are we asking?** Perplexity can look fine while the state does
something insane. Does it?

**What should I run?**

```bash
python stability_check.py --sweep --lengths 128 512 1024 2048
```

**What should I expect to see?**

| transition | depth | L | min `a` | max `a` | frac `a<0` | frac `a>1` | frac NaN | max ‖h‖ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| exact | — | 128 | 0 | 1.000 | 0 | 0 | 0 | 922.6 | ok |
| exact | — | 2048 | 0 | 1.000 | 0 | 0 | 0 | 916.6 | ok |
| poly2, global | 1 | 2048 | −0.059 | **5.3e8** | 7e−04 | 4.0e−03 | **0.79** | 264.9 | FLAG |
| poly4, global | 2 | 2048 | −0.010 | **6.2e17** | 2e−04 | 4.3e−03 | **0.79** | 409 | FLAG |
| poly2, per-head | 1 | 128 | −0.125 | 1.000 | 1.3e−03 | **0** | 0 | 935.3 | ok |
| poly2, per-head | 1 | 2048 | −0.125 | 1.000 | 1.1e−03 | **0** | 0 | 924.6 | ok |
| poly3, per-head | 2 | 2048 | −0.176 | 1.000 | 6.8e−03 | **0** | 0 | 918.4 | ok |
| poly4, per-head | 2 | 128 | −0.194 | 1.000 | 8.5e−03 | **0** | 0 | 923.7 | ok |
| poly4, per-head | 2 | 2048 | −0.194 | 1.000 | 7.6e−03 | **0** | 0 | 917.5 | ok |

Per-head: state norms track the exact model to three significant figures at every
length (923.7 vs 922.6 at L=128; 917.5 vs 916.6 at L=2048), no NaN, and
`frac(a > 1) = 0` — which is the property that actually guarantees the recurrence
cannot expand. Nothing grows with `L`: the compounding explosion Stage 4
demonstrated on the baby model does not happen here, because the per-head interval
keeps `|a| ≤ 1`.

Note the global rows' small `max‖h‖`: that is not health, it is NaN poisoning the
norm statistic. Read the NaN column first.

Read this table together with Stage 7's. Here (2 sequences per length) per-head
`poly3` bottoms out at `a = −0.176`; over Stage 7's 120 sequences the same
configuration reached `a = −14.7` once. The negative excursions are **rare tail
events**, not the typical case — which is why both a small-sample range and a
large-sample minimum are worth reporting, and why neither is clamped.

**Success:** no NaN, no exploding `‖h‖`, `frac a > 1` at zero.
**Failure:** either FLAG column above, or state norms growing with `L`.

**Why does this matter for FHE?** `torch.clamp` is banned here and the ban is
enforced by `test_no_clamp_anywhere`, which scans the source. Clamping would make
every row read "ok" and would not be implementable under CKKS, so a clamped
result would invalidate the whole experiment.

---

## Stage 9 — can lightweight fine-tuning close the remaining gap?

**What question are we asking?** After the polynomial swap, can a *small* number
of trainable parameters recover the loss?

**What should I run?** Progressive modes, smallest budget first:

```bash
# MODE A: polynomial FROZEN; only dt_bias / A_log / D trainable (1,728 params, 0.0013%)
python finetune_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
    --mode A --token-budget 100000

# MODE B: + the polynomial coefficients themselves
python finetune_poly_exp.py ... --mode B --token-budget 1000000

# MODE C: + LoRA on in_proj / out_proj
python finetune_poly_exp.py ... --mode C --token-budget 5000000

# THE CONTROL -- run this, it is not optional
python finetune_poly_exp.py --transition exact --mode A --token-budget 100000
```

**What should I expect to see?** Measured on the CRC RTX 6000, 1,001,472 tokens
of wikitext-2 train, evaluated on 120 × 1024 tokens of validation, with a
**control per mode**:

| run | trainable params | start ppl | **end ppl** | peak GPU | train time |
|---|---|---|---|---|---|
| **exact control**, MODE A | 1,728 | 22.2432 | **21.1932** | 10.97 GB | 209 s |
| `poly2` per-head, MODE A | 1,728 | 22.2702 | **21.1758** | 10.98 GB | 207 s |
| `poly2` per-head, MODE B | 3,456 | 22.2702 | 21.7048 | 10.98 GB | 187 s |
| **exact control**, MODE C | 1,235,136 | 22.2432 | **16.8255** | 11.56 GB | — |
| `poly2` per-head, MODE C | 1,236,864 | 22.2702 | **16.7826** | 11.57 GB | 202 s |

**At both adaptation levels the polynomial matches its control and is slightly
ahead** — 21.1758 vs 21.1932, and 16.7826 vs 16.8255. The ~1.05 and ~5.42
perplexity drops are domain adaptation, not exp-recovery: the controls move by
the same amount. (The same result on a laptop CPU at 100k tokens: 21.019501 vs
21.019528.)

**The polynomial model and the exact model land on the same perplexity to five
decimal places** (21.0195 both; the polynomial is 0.000027 lower, i.e. identical).
After the same light fine-tuning, replacing `exp` with a depth-1 quadratic costs
*nothing that this eval can measure*.

**Read the control — it is the whole point of the table.** Both runs improved by
about **−0.49** perplexity, and none of that is about the polynomial: it is
domain adaptation from training on wikitext and evaluating on wikitext. Without
the control you would have reported "fine-tuning recovered 36× the gap", which is
meaningless (the script now refuses to print that ratio when the starting gap is
under 0.05 and tells you the control command to run instead).

**MODE B is worse than MODE A, and we know why.** Letting the polynomial
coefficients train pushed the ending perplexity to 21.7048, and — the mechanism —
raised `frac(a < 0)` from **0.00091 to 0.02302**, a 25× increase. Freed from the
Chebyshev fit, gradient descent drifted somewhere that made the gate negative
*more* often. This replicates: same direction, same mechanism, at 100k tokens on
a CPU and at 1M tokens on a GPU. The *lightest* mode wins, so a full fine-tune
was never needed.

**Success:** MODE A matches the exact control. It does.
**Failure:** MODE A does nothing, MODE B diverges, and only a full fine-tune
helps — which would mean the polynomial broke something structural.

**Why does this matter for FHE?** If quality needs a full retrain, the approach is
not practical. If 1,728 parameters and 100k tokens suffice, it is nearly free.

**Divergence is reported, not worked around.** Our first distillation attempt died
in four steps; the cause was the 1e45 coefficient spread described in Stage 7, and
the fix was the normalised basis, not a lower learning rate or a clamp.

---

## Stage 10 (optional) — distillation

**What question are we asking?** Does matching the teacher's logits beat plain
language-modelling loss?

**What should I run?** Only after Stage 9 works.

```bash
python distill_poly_exp.py --transition poly2 --interval-mode per-head --pin-zero \
    --mode A --token-budget 1000000 --lambda-kd 1.0 --temperature 2.0

# optionally also match the gate itself
python distill_poly_exp.py ... --lambda-transition 1.0
```

```
L = lambda_lm * cross_entropy(student, labels)
  + lambda_kd * T^2 * KL(student/T || teacher/T)
  + lambda_transition * MSE(a_student, a_teacher)
```

Teacher is the untouched checkpoint, frozen, always under `torch.no_grad()`.
Hidden-state matching is deliberately **not** implemented.

**What should I expect to see?** Measured at 1M tokens on the RTX 6000:

| config | teacher | student before | student after | Δ vs teacher |
|---|---|---|---|---|
| `λ_kd=1, T=2, λ_transition=0` | 22.2432 | 22.2702 | **22.3564** | +0.1132 |
| `λ_kd=1, T=2, λ_transition=1` | 22.2432 | 22.2702 | **22.3564** | +0.1132 |

**Distillation is worse than plain fine-tuning here** — 22.3564 against MODE A's
21.1758 at the same budget. The KD term is 4.45 of a 7.62 total loss, so it
dominates, and it pulls the student toward a teacher that is itself only 22.24.
With a starting gap of +0.027 perplexity there is nothing for KD to recover and
plenty for it to give up. **This is the expected outcome of a method applied where
there is no problem to solve**, and it is worth reporting as such.

**Transition matching at `λ_transition=1` changed nothing measurable** — both rows
are identical to four decimals, and `lm`/`kd` are bit-identical at every logged
step. Not a bug: a standalone gradient check confirms the term produces real
gradients on 5 of 6 transition parameters. It is simply **0.2% of the loss**
(`tr ≈ 0.016` of 7.84), so its gradient is swamped. It would need `λ` of order
100–1000 to matter, and that has not been tried.

Note the memory trick: teacher and student are the *same* 129M weights except for
the transition, so only one model is loaded — the teacher pass runs it with
`ExactExp` under `no_grad`. Halves memory, and guarantees the two differ in
nothing but the transition.

**Success:** distillation recovers at least as much as Stage 9 at the same budget.
**Failure:** it recovers less — which is what happened.

---

## Stage 11 — keep it interpretable

Every script writes `config.json`, `metrics.json`,
`polynomial_coefficients.json` and (for training runs) `training_log.csv` into
its own `runs/<...>/` directory. Then:

```bash
python make_results_summary.py
```

produces `results_summary.csv` with one row per experiment: degree, interval,
whether the polynomial was trainable, fine-tuning mode, token budget, validation
loss, perplexity, delta perplexity, transition min/max, `frac(a<0)`, `frac(a>1)`,
estimated ct-ct depth, training hours and peak GPU memory. Seeds are set through
`set_seed` and recorded in every `config.json`.

---

## Stage 12 — acceptance criteria (Part 13), scored

| criterion | required | measured | |
|---|---|---|---|
| polynomial degree | ≤ 4 | 2, 3, 4 | ✅ |
| estimated nonlinear depth | ≤ 2 | 1 (deg 2), 2 (deg 3–4) | ✅ |
| no recurrent numerical explosion | — | per-head: `max‖h‖` 796.8 vs exact 796.9, no NaN | ✅ |
| transition values in a sensible range | — | per-head: `a ∈ [−1.0, 1.0]`, `frac a>1 = 0`, `frac a<0 ≈ 1%` | ⚠️ mostly |
| perplexity degradation small | — | **−0.003 (deg 2), −0.0006 (deg 4)** before any training; **21.019501 vs 21.019528** for the exact control after 100k tokens of MODE A | ✅ |
| fits on one 24 GB GPU | — | 129M params; MODE A trains 1,728; whole project runs on a *laptop CPU* | ✅ |
| requires modifying unrelated nonlinearities | must not | SiLU / RMSNorm / softplus untouched; `torch.exp` never patched globally | ✅ |

The one honest caveat: `frac(a < 0) ≈ 1%`. `exp` can never be negative. It is
concentrated in the ~70 heads whose `z` range spans several orders of magnitude,
it does not destabilise the state norms at `L = 2048`, and it does not move
perplexity — but it is a real departure from the true transition and it is
recorded in every table rather than clamped away.

**And the clearest negative results**, both worth as much as the positive one:

1. **A single global interval fails completely** — `inf` perplexity at every
   degree from 2 to 4, pinned or not. Measuring before fitting was the difference
   between "this doesn't work" and "a depth-1 quadratic is free".
2. **Training the polynomial coefficients (MODE B) makes things worse**, and the
   mechanism is visible: `frac(a < 0)` goes from 0.0012 to 0.0236. The Chebyshev
   fit was already better than what gradient descent found at this budget.

### What we did not measure

* ~~No GPU number in this repo is real.~~ **Done, 2026-09-19.** Full eval peaks at
  **2.79 GB** (batch 4 × 1024) and takes 16–22 s for 244 blocks; fine-tuning peaks
  at **10.97 GB** (MODE A) to **11.57 GB** (MODE C) and takes ~200 s per 1M
  tokens. Everything fits in 22 GB with room to spare. `results_summary_gpu.csv`
  has the rows.
* ~~`test_parity.py` has never run.~~ **Done, 2026-09-19 on the CRC Quadro
  RTX 6000.** `TestOfficialSSDParity` compares our per-step product-form scan
  against the official `mamba_chunk_scan_combined` Triton kernel on identical
  inputs: **relative error 2.9e-04** at chunk sizes 64/128/256, with the `D`
  skip, and at sequence length 2048 — the size of fp16 kernel arithmetic, not of
  a semantic difference. Our chunk size makes no difference at all (2.789e-04 for
  32/64/128/256 against one official run), confirming the chunked form is the
  algebraic identity it claims to be. The whole-mixer tests
  (`TestOfficialParity`) remain skipped: they need the `causal_conv1d` CUDA
  extension, which is not installed. See `cluster/README.md`.
  Until then, "our forward == the official forward" rests on the
  `ssd_minimal_discrete` equivalence test (1e−12) plus a plausible perplexity,
  not on a direct comparison.
* ~~Token budgets of 1M, and MODE C, and distillation.~~ **Done** — see the tables
  in Stages 9 and 10 above.
* **The 5M-token budget** still has not run: it needs wikitext-103 fetched, and
  `$HOME` on the cluster is at 96%.
* **Whether MODE B's degradation is under-training or real**, and whether a much
  larger `λ_transition` helps.

---

## Repository map

| path | what |
|---|---|
[`HANDOFF.md`](HANDOFF.md) | **start here if you are new**: sizes, what was replaced, code map, traps, open items |
[`FINDINGS.md`](FINDINGS.md) | the evidence: generalisation tests, statistics, and what this does *not* show |
[`PART0_SOURCE_INSPECTION.md`](PART0_SOURCE_INSPECTION.md) | Stage 0: the official source, with line numbers, and the cumsum-vs-product discrepancy |
[`PROGRESS.md`](PROGRESS.md) | one entry per milestone: implemented / learned / still uncertain |
[`baby_mamba/`](baby_mamba) | Stages 1–4. Transition sandbox, polynomial fitting + depth accounting, error propagation, 37 tests. [Its own README](baby_mamba/README.md) |
[`fit_exp_polynomial.py`](fit_exp_polynomial.py) | Stage 2 CLI |
[`real_mamba/`](real_mamba) | Stage 5. Reference SSD, the surgical patch, model loading, data, eval, training utilities, 15 tests |
[`collect_transition_stats.py`](collect_transition_stats.py) | Stage 6 |
[`eval_poly_exp.py`](eval_poly_exp.py) | Stage 7 |
[`stability_check.py`](stability_check.py) | Stage 8 |
[`finetune_poly_exp.py`](finetune_poly_exp.py) | Stage 9 |
[`distill_poly_exp.py`](distill_poly_exp.py) | Stage 10 |
[`make_results_summary.py`](make_results_summary.py) | Stage 11 |
[`run_all.sh`](run_all.sh) | every stage in Part 14's order, as one script |
[`cluster/`](cluster) | CRC (Notre Dame) SGE job scripts — queue `gpu@@jung_gpu`, 1 RTX 6000. Parameterised; see [its own README](cluster/README.md) for the variables to set. |
[`results_summary.csv`](results_summary.csv) | one row per experiment, regenerated by Stage 11 |
[`third_party/mamba/`](third_party/mamba) | read-only trimmed copy of the official repo, commit `e9594ce1` |

## What is explicitly out of scope

No FHE library. No CKKS parameter selection, noise budget or bootstrap placement —
"ct-ct depth" here is a *cost proxy*, not a measured latency. `SiLU`, `RMSNorm`,
`softplus` and the scan structure are untouched. No pretraining. Mamba-2 only
(`Mamba3` is in the same upstream repo and has a different transition; the loader
refuses non-Mamba-2 checkpoints rather than silently mis-measuring one).
