# PROGRESS

One entry per milestone from Part 14. Each entry says what was implemented, what
was learned, and what is still uncertain. Appended to, never rewritten.

Environment for everything below: macOS (Apple Silicon), CPU-only, Python 3.12,
torch 2.14. The 24 GB NVIDIA GPU is a **separate machine**; anything that needs
it is marked accordingly.

---

## Milestone 1 — inspect the official Mamba-2 source ✅

**Implemented.** [`PART0_SOURCE_INSPECTION.md`](PART0_SOURCE_INSPECTION.md), plus a
trimmed read-only copy of the upstream repo at
[`third_party/mamba/`](third_party/mamba) (commit `e9594ce1`, 2026-07-23) so the
quoted line numbers stay valid.

**What was learned.**

1. `A` is `(nheads,)` — **one scalar per head** (`mamba2.py:182`). For the 130M
   checkpoint that is 24 numbers per layer. `z` and `a` therefore carry only
   `B·L·24` distinct values, not one per state dimension. The transition we are
   attacking is a tiny, low-dimensional object; 24 of the 3352 projected channels.
2. **The production code never computes `exp(A·delta)`.** It computes
   `dA_cumsum = cumsum(A·dt)` (`ssd_chunk_state.py:84-86`) and then takes `exp`
   of *differences of prefix sums*. It is exploiting
   `exp(Σ z) = Π exp(z)` to turn a serial product into one cumsum.
3. **That identity is exactly what a polynomial breaks**: `P(Σz) ≠ ΠP(z)`. So
   "replace exp with P" is ambiguous, and the two readings are different models.
   We take the **per-step** reading (`a_t = P(z_t)`, accumulate multiplicatively),
   because that is the one an FHE implementation would actually have to do.
   Documented at length in §0.2 of the Part 0 report.
4. Only three places in the whole repo write `a = exp(A·dt)` per step:
   `mamba2.py:314` (single-token decode fallback),
   `selective_scan_interface.py:162` (the Mamba-1 pure-PyTorch reference), and
   nothing else. `ssd_minimal.py` is cumsum-form.
5. **`use_mem_eff_path=False` does not un-fuse the model.** Both branches call
   Triton (`mamba_split_conv1d_scan_combined` or `mamba_chunk_scan_combined`).
   Even `softplus` lives inside the kernel (`ssd_chunk_state.py:77`), so `delta`
   is not observable from Python on the default path. This is a trap.
6. **There is no `dt` clamp in the pretrained model.** `dt_limit` defaults to
   `(0.0, inf)`, so the Python-side clamp is skipped and the in-kernel clamp is a
   no-op. `z` is unbounded below, set purely by the data — hence Part 6.
7. `state-spaces/mamba2-130m/config.json` confirms `d_model=768`, `n_layer=24`,
   `d_intermediate=0` (so no MLP blocks), `vocab_size=50277`,
   `pad_vocab_size_multiple=16`. With `Mamba2` defaults that gives
   `d_inner=1536`, `nheads=24`, `headdim=64`, `d_state=128`, `ngroups=1`.

**Still uncertain.** The `z` distribution on the real pretrained model — we have
the *initialisation* range but not the trained one. Everything about the
polynomial interval depends on it, and guessing would be the single easiest way
to waste the whole project. Answered in Milestone 6.

---

## Milestone 2 — build the baby transparent transition ✅

**Implemented.** [`baby_mamba/transition.py`](baby_mamba/transition.py) (the
transition + a serial recurrence, with the upstream line number beside each step),
[`baby_mamba/demo_transition.py`](baby_mamba/demo_transition.py) (Part 1's
one-command walkthrough), [`baby_mamba/README.md`](baby_mamba/README.md).

Tiny dimensions: `batch=2, seqlen=8, d_model=4, d_inner=8, headdim=2, nheads=4,
d_state=2`. Faithful to upstream in shapes, sign conventions and initialisation;
explicitly simplified by dropping the causal conv1d, the gated RMSNorm and the
output projection (all listed in the README so nobody is misled).

`BabyConfig.delta_scale` is a **pedagogical knob that does not exist in the real
model**: raising it pushes `z` more negative so students can watch polynomials
fail outside their fit interval. Flagged in the code, the README and every
printout.

**What was learned.** With freshly-initialised weights, `z ∈ [-1.41, 0]` at
`seqlen=8` and `[-4.13, 0]` at `seqlen=1024`. So `[-8, 0]` is *generous* at
initialisation — which is precisely why it is not evidence about the trained
model.

**Still uncertain.** Nothing blocking. The dropped conv1d means the baby model's
`x/B/C` statistics differ from the real model's; that affects the *magnitude* of
state errors we measure, not the mechanism.

---

## Milestone 3 — verify the exact recurrence ✅

**Implemented.** [`baby_mamba/tests/test_transition.py`](baby_mamba/tests/test_transition.py),
13 tests. 38 tests total in `baby_mamba/tests`; all pass in ~1.5 s.

The ones that carry weight:

* `test_recurrence_is_the_closed_form_sum` — brute-forces
  `h_t = Σ_{s≤t} (Π_{k=s+1..t} a_k)·b_s` with nested Python loops and compares.
  This pins the semantics independently of the implementation.
* `test_a_equal_one_means_pure_accumulation` / `..._zero_means_pure_forgetting` —
  `a≡1` must reduce the scan to a cumsum; `a≡0` must reduce it to `b_t`.
* `test_A_is_one_scalar_per_head_and_negative`, `test_delta_positive_and_z_nonpositive`,
  `test_exact_a_in_unit_interval` — the invariants the polynomial will later break.
* `test_only_a_differs_when_swapping_transition` — swapping the transition must
  leave `delta, A, z, b, x, B, C` bit-identical. **This is the experiment's
  integrity check.** If it ever fails, we have contaminated the study by changing
  something other than `exp`.
* `test_poly_that_is_exp_reproduces_exp` — a degree-12 fit recovers the exact
  model to 1e-4, so any error reported later comes from the low degree, not the
  plumbing.

**What was learned.** The fp32 *coefficient storage* in `PolyExp`, not the
evaluation, is what caps a degree-12 fit at ~1e-5. Irrelevant at degree ≤ 4
(error ~3e-2) but worth knowing before anyone tries degree 12 under FHE.

**Still uncertain.** Nothing.

---

## Milestone 4 — implement polynomial exp ✅

**Implemented.** [`baby_mamba/polynomial.py`](baby_mamba/polynomial.py) and the
[`fit_exp_polynomial.py`](fit_exp_polynomial.py) CLI. Four fitting methods
(`chebyshev`, `lobatto`, `remez`, `lstsq`), a `--pin-zero` constraint, and
`--weight relative`. Taylor is deliberately absent.

Part 3's depth accounting is `PowerSchedule`, which builds the power graph as
*data* and **computes** the depth from it, so the claim is auditable rather than
asserted in a comment. Confirmed: degree 2 → depth 1, degree 3 → depth 2,
degree 4 → depth 2, and `ceil(log2(degree))` for degrees 2–32. Horner would be
depth = degree; we pay one extra multiplication to halve the depth at degree 4.

**What was learned.**

1. **`P(0) ≠ 1` is the dominant error, not max|P − exp|.** Unconstrained degree-4
   Chebyshev on `[-8,0]` gives `P(0) = 0.9663`. Since `a` multiplies the state
   once per step, `0.9663^1024 ≈ 5e-16`: a "3% error" destroys all long-range
   memory. `--pin-zero` removes it for one degree of freedom and zero extra depth.
2. **Every unconstrained low-degree fit goes negative** on `[-8,0]`: 40% of the
   interval at degree 2, 17% at degree 4. `exp` can never be negative, and a
   negative `a` flips the sign of a head's entire state in one step.
3. **The interval dominates the degree.** Degree 4 on `[-2,0]` is >100× more
   accurate than degree 4 on `[-8,0]`. If Part 6 says the real `z` is
   concentrated near 0, this problem is much easier than it looks; if it says `z`
   reaches −20, degree 4 is hopeless. Same code, opposite conclusion.
4. A correctness trap we hit and fixed: fitting `Q` to `(exp(z)−1)/z` under
   minimax is **not** minimax for `P = 1 + z·Q`, because the error in `P` is `|z|`
   times the error in `Q`. Our first pinned-Remez was 8× worse than it looked.
   `--pin-zero` is now rejected for `--method remez` with an explanation, and
   the pinned least-squares path constrains `P` directly.

**Still uncertain.** Whether `--weight relative` helps or hurts once the
polynomial is inside a trained model. It is implemented and untested at the model
level.

---

## Milestone 5 — show polynomial error propagation ✅

**Implemented.** [`baby_mamba/error_propagation.py`](baby_mamba/error_propagation.py).
Sequence lengths 16/64/256/1024 × {exact, P2, P3, P4}; measures relative state
error, max state norm, mean/max transition error, `frac(a<0)`, `frac(a>1)`, and
NaN/Inf. Writes `config.json`, `metrics.json`, `polynomial_coefficients.json`, a
CSV and a 3-panel PNG to `runs/part4_error_propagation/`.

**What was learned.**

1. **Pointwise error is flat; state error compounds.** Degree 4: mean `|Δa|`
   stays at 2.3e-02 across all lengths, while the relative error of the final
   state grows 6.2e-02 → 4.3e-01 from L=16 to L=1024 (**7×**). Degree 3 amplifies
   4.2×, degree 2 2.6×. Lower-degree polynomials amplify *less* only because they
   are already saturated.
2. **`--pin-zero` is the best lever found so far.** With Lobatto + `pin_zero`,
   degree 4 at L=1024 drops from 0.43 to **0.13** relative state error — 3.4×
   better, at identical FHE depth. Its pointwise max error is slightly *worse*
   (4.0e-02 vs 3.4e-02), so a pointwise-only evaluation would have rejected it.
   That is the clearest evidence that Part 4 is necessary.
3. **Out-of-interval is catastrophic, not graceful.** `--delta-scale 6` pushes
   `z` to −24.75. Degree 3 then produces `min a = −45.8` — one timestep
   multiplying a head's whole state by −45.8. Degree 2 stays near −0.06 only
   because it is nearly flat. A polynomial does not decay off-interval; it
   diverges.
4. Relative state error saturates near ~1.0 at long L rather than growing without
   bound, because with `a` typically well below 1 the state is dominated by recent
   tokens. The error is bounded but not small — the model is not blowing up, it is
   quietly computing something else.

**Still uncertain.** All of the above is measured on the baby model, whose `z`
range comes from *random* weights. The amplification *mechanism* transfers to the
real model; the *magnitudes* do not. Nothing here justifies a claim about
perplexity.

---

## Milestone 6 — instrument the real 130M transition ✅

**Implemented.** [`real_mamba/`](real_mamba):

* [`reference_ssd.py`](real_mamba/reference_ssd.py) — the SSD rewritten in per-step
  **product** form. `segprod` replaces `segsum`, `cumprod` replaces `cumsum`,
  `suffix_prod_exclusive` replaces `exp(A_cumsum[-1] - A_cumsum)`. Same einsums,
  same chunk decomposition, but the only thing it needs from the transition is
  `a_t`, so a polynomial can go there.
* [`patch.py`](real_mamba/patch.py) — replaces `Mamba2.forward` with a pure-PyTorch
  forward that recomputes the mixer from **that layer's own pretrained weights**.
  `torch.exp` is never patched globally; `A = -exp(A_log)` is left alone.
* [`model.py`](real_mamba/model.py) — two backends, one checkpoint. `official` uses
  `MambaLMHeadModel.from_pretrained` (needs CUDA + Triton); `local` loads the same
  `pytorch_model.bin` into a module tree with byte-identical parameter names, so
  the project runs on a laptop.
* [`nn_ref.py`](real_mamba/nn_ref.py), [`data.py`](real_mamba/data.py),
  [`eval_lm.py`](real_mamba/eval_lm.py), and 31 tests (25 run anywhere, 6 need CUDA).

**What was learned — the correctness anchors.** `ssd_product_form` reproduces the
**official** `ssd_minimal_discrete` to **1e-12** when the transition is `exp`, and a
literal `for t in range(L)` loop to **1e-11** for *any* transition, including
polynomials that go negative, with any chunk size and ragged lengths. Without that
pair of tests nothing downstream would be trustworthy. End to end, the local
backend gives wikitext-2 validation perplexity **17.60** (6 blocks) / **22.24**
(120 blocks) — a plausible pretrained 130M model.

**Still uncertain.** The official-backend parity test
(`test_parity.py::TestOfficialParity`) is written but **has not run here** — this
machine has no NVIDIA GPU and `mamba_ssm` imports Triton at import time. It is
gated on `mamba_ssm` + CUDA and should be the first thing run on the 24 GB box.
Until then, "our reference forward == the fused kernel" rests on the
`ssd_minimal_discrete` equivalence plus the plausible perplexity, not on a direct
comparison.

---

## Milestone 6b — collect real transition statistics ✅ **(this changed the plan)**

**Implemented.** [`collect_transition_stats.py`](collect_transition_stats.py).
Streaming Welford moments, reservoir sampling for percentiles, per layer, per
head, and globally. Records `delta`, `z`, `exp(z)` and `A`, plus
`fraction(z < -2/-4/-6/-8/-10/-12/-16/-20)`.

**What was learned. This is the most consequential measurement in the project.**

1. **`z` reaches −179,766.** Not −8. The fit interval in the brief was off by
   four orders of magnitude, and Part 4 had already shown that outside its
   interval a polynomial *diverges* rather than decaying.
2. **The cause is `A`, not `delta`.** `A` is one scalar per head, and across the
   576 heads of the trained checkpoint `|A|` spans **4.0e−04 to 3.6e+04** — five
   orders of magnitude. 26 heads have `|A| > 100`; layer 4 has one head at
   `A = −36,315`. `delta` itself is unremarkable (softplus output, O(0.001–5)).
3. **So the interval is a per-head quantity**, and 523 of 576 heads are perfectly
   comfortable inside `[-8, 0]`:

   | \|z_min\| | heads |
   |---|---|
   | < 1 | 322 |
   | 1–4 | 171 |
   | 4–8 | 30 |
   | 8–32 | 15 |
   | 32–1000 | 21 |
   | > 1000 | 17 |

4. **A per-head interval is free under FHE**, because `A` is a *weight* and
   weights are plaintext. Same degree, same ct-ct depth, different plaintext
   constants. And narrowing the interval is worth far more than raising the
   degree: degree 4 on `[-4, 0]` has max error 3.7e−03 and **never goes
   negative**, versus 3.4e−02 and 17% negative on `[-8, 0]`.
5. **The extreme heads are a free case, not a hard one.** A head with
   `|A| = 3.6e4` has `exp(z) = 0` to fp32 precision over its whole observed
   range, so its exact optimal degree-4 polynomial is the constant `0` — error
   3e−32, ct-ct depth **0**. `fit_per_head(zero_degenerate=True)` detects this.
6. **The 4% of `z` values below −20 are not a per-head tail**; they are ~4% of
   *heads* that are always far out. That distinction is what makes the per-head
   fix work and a global fix impossible.

**Consequences for the code, all implemented.** `PerHeadPolyExp`,
`fit_per_head`, `head_intervals_from_stats`, `build_per_head_transitions`, and an
`--interval-mode global|per-head` flag on every script. `patch_transition` now
accepts a `{layer_idx: module}` dict. The global mode is kept as the naive
baseline, because its failure is the clearest result we have.

**A bug this exposed.** `numpy`'s `Chebyshev.convert(...).coef` **trims trailing
zeros**, so a degree-4 fit on a very wide interval came back with one
coefficient — and a `PolyExp` built from it would have silently reported
`degree=0, depth=0`, i.e. a wrong FHE cost. Fixed centrally with
`_pad_to_degree`.

**Still uncertain.** The statistics come from 8–32 sequences of wikitext-2. The
extreme heads are a property of the *weights*, so they will not change with more
data, but the `z` percentiles for ordinary heads might shift on a different
domain. `--interval-margin` (default 0.25) is the safety factor, and it has not
been stress-tested against out-of-domain text.

---

## Milestone 7 — evaluate the polynomial replacement with no training ✅

**Implemented.** [`eval_poly_exp.py`](eval_poly_exp.py), with `--sweep` for the
whole table in one command and a `TransitionProbe` that records `min a`, `max a`,
`frac(a<0)`, `frac(a>1)` and `frac(NaN)` during evaluation.

**What was learned.** wikitext-2 validation, 120 × 1024 = 122,880 tokens, fp32:

| transition | interval | ct-ct depth | val loss | perplexity | Δ ppl | min `a` | frac `a<0` | frac `a>1` |
|---|---|---|---|---|---|---|---|---|
| exact | — | n/a | 3.1020 | **22.2425** | — | 0 | 0 | 0 |
| poly2 | `[-8,0]` | 1 | nan | **inf** | — | −0.0595 | 7.5e−04 | 4.1e−03 |
| poly3 | `[-8,0]` | 2 | nan | **inf** | — | −1.7e13 | 5.5e−03 | 0 |
| poly4 | `[-8,0]` | 2 | nan | **inf** | — | −0.0097 | 2.1e−04 | 4.4e−03 |
| poly2 | per-head | **1** | 3.1019 | **22.2392** | **−0.0033** | −0.125 | 1.4e−03 | 0 |
| poly3 | per-head | 2 | 3.1019 | **22.2392** | **−0.0033** | −14.72 | 7.7e−03 | 0 |
| poly4 | per-head | 2 | 3.1020 | **22.2419** | **−0.0006** | −0.194 | 8.2e−03 | 0 |

1. **A depth-1 quadratic is free.** −0.0033 perplexity out of 22.24 (0.015%, and
   negative) with **zero training**. This is a stronger result than the project
   set out to look for: the fine-tuning in Part 9 has almost nothing left to
   recover.
2. **The global interval fails completely** — `inf` at every degree, pinned or
   not. `a` reaches 6.1e17 and 79% of transition values are NaN. `pin_zero` does
   not help because the problem is not accuracy near 0, it is divergence at
   `z = -1.8e5`.
3. **Degree barely matters once the interval is right.** 2, 3 and 4 are all
   within 0.003 perplexity of exact. The interval was the whole game.
4. **`min a = −14.7` for per-head poly3, with no effect on perplexity.** One head
   multiplied its state by −14.7 and the model did not care. This is the least
   comfortable number in the project and it is reported, not clamped.

**A second bug this exposed — and the fix that mattered.** In the raw `z` basis
the per-head coefficients span **1e−45 to 1** (some are fp32 subnormals), and
`z⁴` at `z = -1.8e5` is 1e21. `PerHeadPolyExp` now stores a *scale-normalised*
form, `P(z) = Σ c_k (z/s_h)^k` with `s_h = |xmin_h|` plaintext, fitting
`t ↦ exp(s_h·t)` on `t ∈ [-1, xmax/s_h]`. Mathematically identical (`1/s_h` is a
ct-pt multiply, no extra depth), but every coefficient becomes O(1) or exactly 0.
It also improved the measured `min a` from −1.00 to −0.19 and `frac(a<0)` from
1.2% to 0.8% for poly4, purely from better fp32 conditioning. **Milestone 9 could
not train at all without it.**

**Still uncertain.** One eval set, one domain, one sequence length (1024). A
2048- or 8192-token evaluation would exercise the compounding Part 4 warned about
much harder. Part 8 checks state norms up to 2048 but not perplexity.

---

## Milestone 8 — stability ✅

**Implemented.** [`stability_check.py`](stability_check.py) — per-layer `a`
diagnostics and SSM state norms at sequence lengths 128 / 512 / 1024 / 2048, with
an explicit FLAG column. **No `torch.clamp` anywhere**, and
`test_no_clamp_anywhere` scans the source tree to keep it that way (a `clamp_min`
guarding a division or a variance is allowed and must be commented as such; the
one fidelity clamp that reproduces upstream's `dt_limit` is marked
`ALLOW-CLAMP`).

**What was learned** (L = 512, 1 sequence; full table in
`runs/part8_stability/stability.csv`):

| transition | depth | min `a` | max `a` | frac `a<0` | frac NaN | max ‖h‖ | verdict |
|---|---|---|---|---|---|---|---|
| exact | — | 0 | 1.000 | 0 | 0 | 796.9 | ok |
| poly2 global | 1 | −0.059 | 3.8e8 | 7e−04 | **0.79** | 307 | FLAG |
| poly4 global | 2 | −0.010 | **6.1e17** | 2e−04 | **0.79** | 418 | FLAG |
| poly2 per-head | 1 | −0.997 | 1.000 | 0.012 | 0 | 793.3 | ok |
| poly3 per-head | 2 | −0.999 | 1.000 | 0.011 | 0 | 796.8 | ok |
| poly4 per-head | 2 | −1.000 | 1.000 | 0.011 | 0 | 796.8 | ok |

1. Per-head state norms match the exact model **to three significant figures**
   (796.8 vs 796.9), and do not grow with `L`. The compounding explosion Part 4
   demonstrated on the baby model does not occur here, because the per-head
   interval keeps `|a| ≤ 1`.
2. `frac(a > 1) = 0` for every per-head configuration. That is the property that
   actually guarantees non-explosion, and it holds.
3. The global-interval failure is not subtle: 79% of transition values are NaN.
   `max‖h‖` for those rows is *small* only because NaN propagation kills the norm
   statistic, which is why the NaN column has to be read first.

**Still uncertain.** `frac(a<0) ≈ 1%` is stable across lengths but unexplained at
the level of "which heads, on which tokens, and why is the network insensitive to
it". A per-head breakdown is saved in `metrics.json` and has not been analysed.

---

## Milestone 9 — small fine-tuning ✅ (implemented; runs are budget-limited here)

**Implemented.** [`finetune_poly_exp.py`](finetune_poly_exp.py) and
[`real_mamba/train_utils.py`](real_mamba/train_utils.py): three progressive modes
(A = polynomial frozen, only `dt_bias`/`A_log`/`D`; B = + coefficients; C = + LoRA
on `in_proj`/`out_proj`), `--token-budget` on the CLI, AdamW with cosine decay and
warmup, gradient accumulation, a separate learning rate for the polynomial,
bf16/fp16/fp32, `training_log.csv`, and a post-training transition probe so the
reported `frac(a<0)` is the trained value rather than the fit-time one.

**What was learned.**

1. **MODE A touches 1,728 of 128,989,632 parameters (0.0013%)** and is enough to
   move perplexity. A 20k-token smoke test took per-head poly2 from 17.6169 to
   **17.4374** — *below* the untouched exact model's 17.5990.
2. **That overshoot is a trap, and the script now says so.** Training on wikitext
   and evaluating on wikitext improves perplexity for reasons unrelated to the
   polynomial. The only valid comparison is the polynomial run's *ending* against
   an **exact-exp control** trained identically. `--transition exact --mode A` is
   now supported precisely for that, and the script prints a warning whenever the
   "recovered fraction" exceeds 1.0.
3. **Divergence is a result, not a crash.** Both training scripts abort with a
   diagnosis (lr too high for `A_log`, since `A = -exp(A_log)` moves
   exponentially; or `--poly-lr` too high; or a global interval that was already
   diverging) instead of writing NaN into `metrics.json`. MODE B/C default
   learning rates were lowered from 3e−3 to 1e−3 after the first divergence.
4. **There is very little left to recover.** Part 7 closed the gap to −0.003
   perplexity with no training at all, so Part 9's honest role here is to confirm
   that the lightweight modes do not *hurt*, and that MODE C and a full
   fine-tune are unnecessary — which satisfies the brief's instruction not to
   full-fine-tune unless the light versions clearly fail.

5. **The 100k-token runs, with the control.** wikitext-2 train -> wikitext-2
   validation (48 x 1024 tokens):

   | run | trainable | start ppl | ending ppl | frac(a<0) after |
   |---|---|---|---|---|
   | exact-exp control, MODE A | 1,728 | 21.5102 | **21.019528** | 0 |
   | poly2 per-head, MODE A | 1,728 | 21.4971 | **21.019501** | 0.0012 |
   | poly2 per-head, MODE B | 3,456 | 21.4971 | 21.2087 | 0.0236 |

   **The polynomial model and the exact model end at the same perplexity to five
   decimals** (the polynomial is 0.000027 lower). Both improved by ~0.49, and the
   control proves all of that is domain adaptation, not exp-recovery. After the
   same light fine-tuning, the depth-1 quadratic costs nothing measurable.

6. **MODE B is worse than MODE A, with a visible mechanism.** Training the
   coefficients raised `frac(a < 0)` from 0.0012 to **0.0236** (19x) and cost
   0.19 perplexity. Gradient descent at this budget found something worse than
   the Chebyshev fit it started from. So MODE C and the full fine-tune were never
   run: the brief says not to full-fine-tune unless the light versions clearly
   fail, and the *lightest* one won outright.

**Still uncertain / not done.** The 1M- and 5M-token budgets have **not** been
run: this machine is CPU-only and a 1M-token run is ~55 min at the observed rate
(~330 s per 100k tokens including three evaluations). MODE C has not been run at
all. `training_hours` and `peak_gpu_memory_gb` are recorded but are `nan` on CPU,
so **no GPU memory number in this repo is measured** — they will populate on the
24 GB box. Whether MODE B's degradation persists at a larger budget, or is just
under-training, is open.

---

## Milestone 10 — knowledge distillation ✅ (implemented, smoke-tested only)

**Implemented.** [`distill_poly_exp.py`](distill_poly_exp.py). Output-logit KD
with configurable temperature, `lambda_lm` / `lambda_kd` / `lambda_transition`,
and the optional `MSE(a_student, a_teacher)` gate-matching term. Hidden-state
matching is deliberately absent.

**What was learned.**

1. **Teacher and student share one set of weights.** They differ only in the
   transition, so we load **one** model and run the teacher pass with `ExactExp`
   under `torch.no_grad()` and the student pass with the polynomial. Halves
   memory on the 24 GB target and makes it impossible for the two to differ in
   anything but the transition. Cost: one extra forward per micro-batch.
2. **The first distillation run diverged in four steps**, and the cause was
   Milestone 7's coefficient-scale problem, not the loss. A single learning rate
   cannot train coefficients spanning 1e45. The fix was the scale-normalised
   basis; after that, the same command at `--lr 3e-4` is stable
   (teacher 17.7734, student 17.7475 → 17.7710 on a 4-block eval).
3. `lambda_transition` on `MSE(a_student, a_teacher)` produces a small, healthy
   loss term (~0.011 at the start) and is cheap, because `a` is only
   `B·L·24` values per layer.

**Still uncertain.** Everything quantitative. Only an 8,192-token smoke test has
run. Whether KD beats plain LM loss at equal budget is untested, and with a −0.003
perplexity gap it may not be answerable on this model at all — a harsher setting
(longer contexts, or degree 2 on a deliberately too-wide interval) would be a
better testbed for the KD machinery.


---

## Milestone 6c — the parity gate, verified on real GPU hardware ✅

**2026-09-19, CRC `qa-rtx6k-019` (Quadro RTX 6000, sm_75), job 1457354.**
`PRE-FLIGHT PASSED`, 73 tests passed / 3 skipped.

**The result Milestone 6 said was missing.** `TestOfficialSSDParity` compares
`ssd_product_form` (our per-step product form, transition = `exp`) against the
official `mamba_chunk_scan_combined` Triton kernel on identical inputs:

| comparison | relative error |
|---|---|
| chunk 64 / 128 / 256 | 2.974e-04 / 2.966e-04 / 2.964e-04 |
| with the `D` skip term | 2.879e-04 |
| sequence length 2048 | 2.886e-04 |
| our chunk 32/64/128/256 vs their chunk 128 | 2.789e-04 (all four identical) |

That is the magnitude of fp16 kernel arithmetic against an fp32 reference, not of
a semantic difference — and a time-reversed control in the same test is >10x
worse, so the tolerance cannot be passing a wrong answer. **The cumsum-to-product
reformulation is verified against the real kernel.**

Also verified: the official backend loads the checkpoint (129.0M params, 0.727 GB
on GPU) and reproduces the laptop's evaluation — poly4/per-head gives ppl
**18.9538** on GPU/official vs **18.9532** on CPU/local, a 3e-5 difference.

**What it took to get there — four failed submissions, each a real finding.**

1. **1455182** — a test bug of mine. `TestRealCheckpoint` used `backend="auto"`,
   which falls back to local on the laptop but picks *official* on the cluster,
   and the official model **cannot run on CPU at all**: `Block.forward` calls the
   Triton `layer_norm_fn` even when the mixer is patched. Pinned that class to
   `backend="local"`.
2. **1455191** — `causal_conv1d` is not installed, and **both** branches of
   `Mamba2.forward` need it: the fused path calls
   `causal_conv1d_cuda.causal_conv1d_fwd` directly, and the `use_mem_eff_path=False`
   fallback is **broken upstream in mamba_ssm 2.2.2** (`self.dconv`, a typo for
   `self.d_conv`, plus the slice `[:, -(d_conv-1):]` where current upstream has
   `[:, :-(d_conv-1)]`). So the whole-mixer comparison is impossible without that
   package. Replaced it with `TestOfficialSSDParity`, which talks to the scan
   kernel directly and needs no conv extension — a better-targeted test anyway,
   since the scan is the only thing this project replaces.
3. **1457312 / diagnostics 1457328, 1457331** — the new tests all died with
   `IndexError: map::at` inside Triton's IR translation. Two diagnostic jobs
   isolated it: plain Triton kernels compile fine, `layer_norm_fn` compiles fine,
   but **`mamba_chunk_scan_combined` compiles in float16 ONLY** on sm_75. fp32 and
   bf16 both fail. Turing has fp16 tensor cores and no bf16 ones, and there is no
   valid MMA layout for the fp32 `tl.dot` path.
   **`torch.cuda.is_bf16_supported()` returns `True` on this card anyway** — it
   reports driver support, not tensor-core support. Added
   `real_mamba.model.recommended_dtype`, and every script now defaults to
   `--dtype auto`: bf16 only from sm_80 up, fp16 below. The previous
   `--dtype bfloat16` default would have silently picked an unusable dtype on the
   only GPU we have.
4. **1457342** — one leftover test still used the fp32 convention
   (`expected scalar type Float but found Half`). Fixed.

**Still uncertain.** The three `TestOfficialParity` whole-mixer tests remain
skipped pending `causal_conv1d`. Installing it into the shared `pdpo` env would
change which conv path `mamba_ssm` takes for the DP-GRPO project's running jobs,
so it should go into an isolated `--target` directory instead, and that has not
been done. The scan-level parity above covers what this project actually changes;
the conv, RMSNorm and projections are untouched code paths.


---

## Milestone 7-10 rerun on the GPU — the numbers that were missing ✅

**2026-09-19, CRC `qa-rtx6k-019` (Quadro RTX 6000, 22.2 GB), OFFICIAL `mamba-ssm`
backend, fp32.** Jobs 1458346 (eval), 1458347 (fine-tune), 1458454 (mode-C
control), 1458438 (distillation). Raw rows in `results_summary_gpu.csv`, run
directories under `runs/gpu/`.

### Part 7, full validation set (244 x 1024 = 249,856 tokens)

| transition | interval | depth | perplexity | Δ vs exact | frac a<0 |
|---|---|---|---|---|---|
| exact | — | n/a | **23.6843** | — | 0 |
| poly2 / poly3 / poly4 | global `[-8,0]` | 1 / 2 / 2 | **inf** | diverges | — |
| poly2 | per-head | **1** | 23.7194 | **+0.0351** | 0.00043 |
| poly3 | per-head | 2 | 23.6839 | **−0.0004** | 0.00635 |
| poly4 | per-head | 2 | 23.6846 | **+0.0003** | 0.00844 |

Reproduces the laptop result on the official backend and the full eval set. The
absolute perplexity is higher than the laptop's 22.24 only because that was 120
blocks and this is all 244.

**Real resource numbers at last** (every one in this repo was `nan` before):
peak **2.79 GB** for the whole evaluation at batch 4 x 1024, and **16–22 s** per
244-block sweep versus ~5 minutes on the laptop CPU.

Part 6 on 64 blocks (37.7M samples of z) sharpens the picture: observed
`z_min = −263,700`, `fraction(z < −8) = 0.048`.

### Part 8, stability to L=2048

Per-head, every degree, every length: `max a = 1.000000`, **`frac(a > 1) = 0`**,
no NaN, and `max‖h‖` tracking exact within ~1% (e.g. 975.8 vs 975.4 at L=2048 for
poly4). Global-interval rows: 79% NaN and `a` up to **2.4e18**.

### Part 9 at 1M tokens — with a control per mode

| run | trainable | start ppl | **end ppl** | peak GPU | time |
|---|---|---|---|---|---|
| **exact control, MODE A** | 1,728 | 22.2432 | **21.1932** | 10.97 GB | 209 s |
| poly2 per-head, MODE A | 1,728 | 22.2702 | **21.1758** | 10.98 GB | 207 s |
| poly2 per-head, MODE B | 3,456 | 22.2702 | 21.7048 | 10.98 GB | 187 s |
| **exact control, MODE C** | 1,235,136 | 22.2432 | **16.8255** | 11.56 GB | — |
| poly2 per-head, MODE C | 1,236,864 | 22.2702 | **16.7826** | 11.57 GB | 202 s |

1. **At both adaptation levels the polynomial matches its control, slightly ahead.**
   MODE A: 21.1758 vs 21.1932. MODE C: 16.7826 vs 16.8255. The ~1.05 and ~5.42
   perplexity drops are **domain adaptation, not exp-recovery** — the controls
   move by the same amount. After identical light fine-tuning, a depth-1
   quadratic costs nothing this evaluation can measure. This is the 100k-token
   laptop result replicated at 10x the budget with a proper control.
2. **MODE B is worse than MODE A again, with the same mechanism.** 21.7048 vs
   21.1758, and `frac(a<0)` rises 0.00091 → 0.02302 (25x). Gradient descent on
   the coefficients finds something worse than the Chebyshev fit it starts from,
   at 100k and at 1M tokens alike. Replicated, not a fluke.
3. **A control gap that had to be fixed mid-flight.** The first fine-tune job ran
   only a MODE-A control, which would have credited the polynomial for LoRA's
   5.4-perplexity drop in MODE C. `cluster/job_finetune.sh` now runs a control per
   mode. (MODE B cannot have one: its extra parameters *are* the polynomial
   coefficients, so it is compared against MODE A.)

### Part 10, distillation at 1M tokens

| config | teacher | student before | student after | Δ vs teacher |
|---|---|---|---|---|
| `λ_kd=1, T=2, λ_transition=0` | 22.2432 | 22.2702 | **22.3564** | +0.1132 |
| `λ_kd=1, T=2, λ_transition=1` | 22.2432 | 22.2702 | **22.3564** | +0.1132 |

1. **Distillation is worse than plain fine-tuning here** — 22.3564 against MODE
   A's 21.1758 at the same budget. The KD term is 4.45 of a 7.62 total loss, so
   it dominates, and it pulls the student toward a teacher that is itself only
   22.24. With a starting gap of +0.027 perplexity there is nothing for KD to
   recover, and plenty for it to give up.
2. **Transition matching at `λ_transition=1` changed nothing measurable.** Both
   rows are identical to four decimals, and `lm`/`kd` are bit-identical at every
   logged step. Not a bug: a standalone gradient check confirms the term produces
   real gradients on 5 of 6 transition parameters. It is simply **0.2% of the
   loss** (`tr ≈ 0.016` of 7.84), so its gradient is swamped. It would need `λ` of
   order 100–1000 to have any influence, and that has not been tried.

**Bugs this run exposed, all fixed.**

* `cluster/env.sh` still did `conda create` + `pip install torch`. On a `$HOME`
  with 5.5 GB free it built 3.6 GB of a ~7 GB env, **filled the filesystem to 0
  bytes**, and killed the job with `OSError: [Errno 28]`. A full `$HOME` also
  endangers every other job running at the time. The partial env and the 1.6 GB
  pip cache it created were removed (recovering to 4.4 GB free); `env.sh` now
  **verifies and refuses**, and never installs. The three job scripts that still
  sourced the old version were pointed at it.
* `--dtype bfloat16` was still the default in `job_finetune.sh` / `job_distill.sh`.
  Changed to explicit fp32: `auto` resolves to fp16 on sm_75, and these scripts
  have no `GradScaler`, which is exactly how a 1,728-parameter MODE-A run
  silently learns nothing. We are not memory-bound — peak was 11 GB of 22 GB.
* `--transition exact --mode C` crashed (`--trainable-poly makes no sense with
  --transition exact`) because MODE B/C auto-enable that flag. The exact control
  must be able to run MODE C, so the auto-enable is now skipped for `exact`.

### A stability regression that only shows up AFTER training

Checked 2026-09-20 from the post-training transition probes. Before any training,
every per-head configuration had **`frac(a > 1) = 0`** — the property that
guarantees the recurrence cannot expand. After 1M tokens:

| run | `frac(a < 0)` | **`frac(a > 1)`** |
|---|---|---|
| poly2 per-head, MODE A | 0.00091 | **0.00000** |
| poly2 per-head, MODE B | 0.02302 | **0.04414** |
| poly2 per-head, MODE C | 0.02777 | **0.02112** |

**MODE A stays inside the stable region. MODE B and MODE C leave it.**

The mechanism is the same in both: MODE B/C train `A_log` and `dt_bias` (and in B,
the coefficients), which are exactly the things that produce `z`. Move them and
`z` drifts away from the interval each head's polynomial was fitted on — and
outside its interval a polynomial diverges rather than decaying (Part 4). The
`--interval-margin 0.25` absorbs the MODE-A drift and not the MODE-B/C drift.

This matters for how MODE C should be read. Its perplexity is the best number in
the whole project (16.7826), and its gate is the second-worst behaved. Perplexity
on 120 blocks of in-domain text did not notice a decay factor exceeding 1 on 2% of
tokens; a longer context or a different domain might. **The best-looking result
has the worst-behaved transition**, which is precisely the pairing Part 8 exists
to catch.

Cheap fix worth trying, not yet tried: re-measure `z` after fine-tuning and re-fit
the per-head intervals (a 0.04 s fit plus one forward pass), or simply widen
`--interval-margin` for the modes that move `A_log`.

**Still uncertain.** The 5M-token budget (needs wikitext-103 fetched, and `$HOME`
is at 96%). Whether MODE B's degradation is under-training or real. Whether a
much larger `λ_transition` helps. Whether re-fitting after training removes the
`a > 1` regression. The three whole-mixer `TestOfficialParity` tests remain
skipped pending an isolated `causal_conv1d` install.


---

## Milestone 11 — the generalisation campaign (E1-E6) ✅

**2026-09-20, CRC. Jobs 1460570 (E2), 1460582 (E1), 1460583 (E3-E6).**
Written up in [`FINDINGS.md`](FINDINGS.md); raw rows in `runs/gpu/e1_cross_domain/`,
`runs/gpu/e2_scale/`, `runs/gpu/e3456_robustness/`.

**Why this was needed.** Everything before this rested on one model, one corpus,
one sequence length, one seed — and, worst, the per-head intervals were *measured
on the same corpus they were evaluated on*. That is the first thing a reviewer
attacks, and it would have been fair.

**What changed as a result: the defensible claim is degree 4, not degree 2.**
In-domain, poly2 (depth 1) looked free. Four independent tests disagree:

| test | poly2 | poly4 |
|---|---|---|
| held-out domain (Pile / LAMBADA) | +0.089 / +0.202 | **+0.013 / +0.011** |
| sequence length 8192 | −0.90 | **−0.012** |
| paired bootstrap, 95% CI | **significantly different** from exact | not distinguishable |
| margin sensitivity, 0 → 1.0 | 0.44 ppl spread | **0.007 ppl spread** |

poly4 passes all four for one extra ct-ct multiplication and **no extra depth**.

**What held up everywhere.**
1. The five-orders-of-magnitude `|A|` spread is universal: 130m, 370m, 780m and
   1.3b all show it, and the global `[-8,0]` interval gives `inf` at every scale.
   This is a property of trained Mamba-2, not of one checkpoint.
2. The intervals transfer. Fitted on wikitext-103 and never refitted, only
   **1.6e−04** of tokens fall outside them on The Pile — a corpus of web text,
   code and papers.
3. LAMBADA zero-shot accuracy drops by **0.0013** (one example in 800), identically
   for degrees 2, 3 and 4. Perplexity-independent evidence that the model still works.
4. `frac(a > 1)` stays at or below 2e−06 everywhere without training.

**A finding worth its own line.** Smaller interval margins are *better*
(margin 0 beats margin 0.25 for every degree). Widening the interval to be "safe"
trades approximation accuracy for coverage, and at degree 2 that trade is
expensive. The right default is a small margin plus a high enough degree, not a
wide margin.

**Odd degrees are less safe than their depth suggests.** poly3 has the same ct-ct
depth as poly4 but reached `a = −130.7` on The Pile. It never cost accuracy in our
measurements, but an unbounded negative tail is not something to ship on the
strength of "it did not matter in our tests".

**Bugs fixed during the campaign.** wikitext-103's train split is two parquet
shards; `_read_parquet_text` hardcoded `-of-00001` and failed with a
`LocalEntryNotFoundError` that reads like a network problem. Shard count is now
probed. Also hit the documented `ssh host '...$VAR...'` trap twice more — the
remote login shell ate a `$(qsub ...)` command substitution.

**Still uncertain.** No FHE implementation exists; "depth 2" remains a proxy with
no CKKS parameters, noise budget or latency behind it. `frac(a<0) ≈ 0.008` is
unexplained. The MODE B/C post-fine-tuning `frac(a>1)` regression still has no
re-fit experiment. English only, one architecture family.

---

## Milestone: STEP 1' Phase 0 + Phase 0b + Phase 1 — the eval floor, and Path B's cause of death

**Implemented.** `eval/noise_floor.py` (unpaired spread across disjoint token
shards, no block straddling), `eval/paired_significance.py` (per-shard paired
difference, the correct error bar for an operator swap), and
`norm/collect_norm_stats.py` (the distribution of the mean-square argument `v`
at every RMSNorm instance). Both paths' scaffolding exists — `norm/const_norm.py`
(Path A, learned constant divisor, zero levels) and `norm/newton_norm.py`
(Path B, prescaled Newton inverse sqrt) — with one weight set serving as both
teacher and student.

**Learned.**

*The floor, and which floor.* Unpaired 2σ is 5.50 ppl at L=512 and 4.14 at
L=2048 (16 shards, GPU). Paired, the same swap is measured 88× tighter: the exp
gate is +0.0001 with 2·SEM = 0.0019, and squared-softplus is +0.4663 with
2·SEM = 0.0562 — **significant**, where the unpaired floor would have cleared
it. This is not a refinement, it is the difference between a true and a false
conclusion, and `eval/noise_floor.py` had been printing the false one for all
four gates. Any gate verdict in this project must come from the paired test.

*Path B is dead, on a criterion fixed before the measurement.* Newton's method
for `1/sqrt(v)` converges only for `v/s ∈ [0.25, 2.0]` — an 8× window — so a
static per-layer prescale needs the **within-layer** spread of `v` to fit in 8×.
Measured over 24 `norm_gated` instances: median `p99/p1` = **19.66** at L=512
and 19.54 at L=2048; worst layer **6268** and 6745; `max/min` ≈ 1e7. The median
layer is 2.5× too wide and the worst ~780×. No training budget fixes a
divergent iteration. The nine planned Path B configurations were cancelled
rather than run — the point of stating the kill condition in advance.

*Path A's length question resolved favourably.* Median `v` moves by 0.971 from
L=512 to L=2048, worst layer 1.132, `p99/p1` shift 1.076. One learned constant
can serve both lengths; per-length constants are not needed. But the same
within-layer `p99/p1` of ~20 means a constant divisor faces a ~4.4× swing in
`1/sqrt(v)` inside the median layer, and ~79× in the worst — so Path A is not
free either, and its cost has to be read off perplexity, not off this table.

**Bugs these logs exposed.** (1) `norm_pre` and `norm_f` collected **zero**
samples — 25 of 49 instances — and the report skipped empty sites silently, so
the table looked complete. `fused_add_norm=True` routes both through
`layer_norm_fn` with `self.norm.weight`, so the module is never called and a
forward pre-hook cannot fire (`block.py:57`, `mixer_seq_simple.py:208`). The
script already handled this for `norm_gated`; I assumed the other two were
ordinary module calls. It now un-fuses for the collection pass and shouts
`NO SAMPLES COLLECTED`. **Path A stage A1 targets `norm_pre`, so Phase 1 is only
24/49 done and A1 cannot be initialised from measurement until it is rerun.**
(2) The floor job died after its unpaired stage on a missing 24 MB
`gate_stats.json`, excluded by the root `.gitignore`'s blanket `*.json`; the job
now regenerates it.

**Still uncertain.** Path A is unrun at every stage. Whether a constant divisor
survives a within-layer 20× spread of `v` is the open question, and the honest
prior from that number is "probably not without help" — which would leave the
normalisation operators unreplaced and is a legitimate outcome of this step.
Nothing is yet known about `norm_pre` or `norm_f` distributions. No CKKS
parameters back the level counts.

**A process failure worth recording.** Four jobs died and I diagnosed the cause
twice without reading a single log — once blaming `env.sh`, once a missing
`logs/` directory. The logs show the first was right in substance and wrong in
mechanism (an unbound-variable abort under `set -u`, not a bad interpreter
path), and the second was simply false: `logs/` existed and jobs wrote into it.
The first "verification" of that second claim was `git ls-files` run in a
directory that is not a git repository; it exited 128 with no output and I read
empty output as "nothing tracked". `qacct` is unavailable on this cell, so job
logs are the only record — read them before theorising.
