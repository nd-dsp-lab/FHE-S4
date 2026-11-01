# Replacing the *other* non-polynomial gates

Follow-on to the exp work. Status after the first pass: **exp turned out to be the
easy one.**

| gate | instances | status | best Δ perplexity | depth |
|---|---|---|---|---|
| `exp(A·Δ)` | 1 / layer | **solved** (previous work) | **+0.0002** | 2 |
| `softplus` → `Δ` | 1 / layer | **works, but a real regression** | **+0.59** (significant) | 2 |
| `SiLU` in the gated norm | 1 / layer | **not solved** | +4.55 (significant) | 2 |
| `SiLU` after conv1d | 1 / layer | **not solved — unstable** | +5.85 (significant), see below | 2 |
| fused `softplus+exp` | 1 / layer | **not solved** (§7b) | +134 | **2** (vs 4) |
| `RMSNorm` 1/√· | **2 / layer** + `norm_f` | **not attempted** | — | 6–15 est. |

Measured on `state-spaces/mamba2-130m`, wikitext-2, degree 4, per-channel
intervals, no training. Baseline (everything exact) 18.1368.

> **Update 2026-10-28 (§9):** segmented lookup tables (ROSETTA-style, evaluated in
> TFHE rather than as CKKS polynomials) replace every gate in this table at once
> for **+0.017 / +0.016** paired perplexity (L = 512 / 2048), and the RMSNorm
> 1/√· table has been run encrypted in OpenFHE at 128-bit security: 15.2 bits
> relative with two Newton steps. The polynomial rows above are kept as the record
> of what failed and why.

---

## 1. Depth is a per-layer budget, and it is the thing to minimise

The 130M has 24 blocks **in series**, so whatever one block costs is paid 24
times. Counting only the verified parts — our depth-2 exp gate, the depth-10
prefix-product tree at L=1024, ~2 for the einsums — a block is already ~14
levels, i.e. **~336 for the network**. A CKKS setup typically affords 10–30
levels before a bootstrap, so bootstrapping here is **structural, not an
optimisation**.

That fixes the currency: a gate costing 2 instead of 6 saves **96 levels
network-wide**. It also means **RMSNorm is the biggest single prize**, at an
estimated 6–15 levels and *two instances per layer* — plausibly more than
everything else combined.

Per-layer critical path is `max(SiLU_conv, softplus → exp) + SiLU_norm`. Note
softplus and exp compose **in series**, which is why fusing them is worth
testing (`gates.py:FusedDtGate`, implemented, not yet evaluated).

## 2. Measure first — again

`collect_gate_stats.py` records each gate's real input range before anything is
fitted. Results (6 × 1024 tokens):

| gate input | min | p50 | max | per-channel narrowing |
|---|---|---|---|---|
| `dt_raw` | −13.9 | −0.09 | 14.0 | 2.9× |
| `softplus_in` | −13.7 | −1.07 | 13.3 | 2.8× |
| `silu_conv_in` | −26.6 | −0.09 | 36.9 | **26.5×** |
| `silu_norm_in` | −17.7 | −0.30 | 34.5 | 6.6× |
| `rmsnorm_meansq` | 0.0019 | 40.0 | 11,260 | **6.1e+06× spread** |

One global interval gives degree-4 SiLU a max error of **2.29** — larger than
most of the values it is approximating. Per-channel is again the difference
between usable and useless.

## 3. softplus: the failure was a SIGN error, not an accuracy error

A plain per-head degree-4 fit to softplus gives **infinite perplexity**. The
reason is not the 3e−2 error:

> **415 of 576 heads produce a negative `Δ` somewhere inside their own fitted
> interval** (median 11% of the interval).

And `Δ < 0` is catastrophic, not inaccurate:

```
Δ < 0  →  z = A·Δ > 0  →  a = exp(z) > 1  →  the recurrence EXPANDS
```

Exact softplus can never return a negative, so no extra degree fixes a *form*
that can.

**The fix: `Δ = Q(x)²`**, with `Q` fitted to `√softplus(x)`. A square cannot be
negative, so the invariant holds for every input — including inputs never
measured and inputs outside the fitted interval. **Structural, not statistical.**

| | negative-`Δ` heads | Δ perplexity | depth |
|---|---|---|---|
| direct degree-4 polynomial | **415 / 576** | **inf** | 2 |
| `Q(x)²`, `Q` degree 2 | **0 / 576** | **+0.278** | **2** |

Same depth (effective degree 4 either way), and the guarantee is free. Its
pointwise error is *worse* (median 0.25 vs 0.033) and it is the one that works —
the same lesson as `pin_zero` in the exp work: pointwise error is a poor
predictor of what a recurrence does with it.

## 4. SiLU after conv1d is fragile, and margin is not the lever

I hypothesised the failure was tokens falling outside the fitted interval (77% of
channels have an observed `max − p99.9` tail exceeding a 0.15 margin), and
widened the intervals. **That made it much worse.** The full sweep:

| interval margin | SiLU-conv ppl | SiLU-norm ppl |
|---|---|---|
| 0.00 | **2.2e+42** | **19.29** (+1.69) |
| 0.05 | **21.55** (+3.95) | 19.82 |
| 0.15 | 61.74 | 21.39 |
| 0.60 | 6.3e+15 | 48.69 |

SiLU-norm is monotone — tighter is better, best +1.69. **SiLU-conv is
non-monotone and swings over 40 orders of magnitude**, with a narrow optimum
near 0.05. Too tight and it diverges off-interval; too wide and the fit degrades.
That is the signature of a genuinely fragile approximation, not a mistuned knob.

Plausible reason it is the worst of the four: the conv output is split into
`x`, `B` **and** `C`, so its error corrupts both what is written into the state
and what is read out of it, on every timestep.

## 5. Higher degree actively backfires

| degree | depth | median max err | worst max err |
|---|---|---|---|
| 4 | 2 | 6.5e−06 | 5.1e−02 |
| 6 | 3 | 1.0e−07 | 1.2e−02 |
| 8 | 3 | 6.0e−08 | **1.73** |
| 12 | 4 | 6.0e−08 | **2.2e+12** |

Beyond degree ~6 the fit explodes on the narrowest channels — fp32 coefficient
conditioning, the same hazard `_pad_to_degree` and the scale normalisation exist
to manage. **Degree 4–6 is the usable band**; "just raise the degree" is not
available.

## 6. RMSNorm: why it is deferred, not forgotten

`1/√(mean(x²)+ε)` needs **both** a reciprocal and a square root, and its
argument spans **6.1 × 10⁶** on real data (0.0019 → 11,260). No low-degree
polynomial covers six orders of magnitude. It needs range reduction (factor out a
plaintext power of two per block? the exponent is data-dependent, so this is not
obvious) plus Newton–Raphson or Goldschmidt iteration, at ~2–3 multiplies per
iteration.

It is also the **most valuable** target, because it appears **twice per layer**
(the block pre-norm and the mixer's gated norm) plus once at the end. It deserves
its own effort rather than being bolted onto this pass.

## 7. Where this leaves the overall claim

Exact baseline 18.1368. Best per gate, individually, no training:

- exp **+0.0002** ✓
- softplus **+0.59** — a real regression, not noise ⚠
- SiLU-norm **+4.55** — significant ✗
- SiLU-conv **+5.85** at a knife-edge margin ✗

> **These verdicts were revised on 2026-09-22.** They previously read +0.278,
> +1.69 and +3.95 and were described as within the evaluation noise. That
> comparison used the UNPAIRED floor (2σ ≈ 5.5), which is the spread of
> perplexity across *different text* and far too lenient for an operator swap
> where both models see the same tokens with the same weights. Measured
> properly, paired, at 16 shards and both lengths, all three are SIGNIFICANT.
> Only the exp gate survives: |Δppl| < 0.0015 at both lengths. See the paired
> table in PROGRESS.md.
- all four together: **inf** — the errors compound

So the honest position: **two of four solved, one marginal, one unsolved, one
untouched.** The exp result does not generalise to the other gates for free, and
the reason differs per gate — a sign invariant for softplus, numerical fragility
for SiLU-conv, dynamic range for RMSNorm.

## 7b. The fused softplus+exp route: halves depth, fails on the same kind of invariant

`FusedDtGate` approximates the whole per-head chain
`x -> exp(A_h * softplus(x + b_h))` with ONE polynomial, so that path costs
**depth 2 instead of 2+2 = 4**. It works mechanically and the depth saving is
real. It also costs **+134 perplexity** (151.41 vs a 16.92 baseline).

The error does scale with `|A|` as predicted, since large `|A|` makes the fused
function nearly a step in `x`:

| \|A\| bucket | heads | median max err | worst |
|---|---|---|---|
| < 0.5 | 377 | 0.0085 | 0.078 |
| 0.5 – 2 | 95 | 0.0106 | 0.290 |
| 2 – 8 | 57 | 0.0216 | 0.116 |
| 8 – 100 | 21 | **0.114** | 0.193 |
| > 100 | 26 | 0.031 | 0.150 |

But the **decisive** problem is not accuracy, it is the invariant:

> **401 of 576 heads produce `a > 1`**, and 74 produce `a < 0`.

`a > 1` is the expanding recurrence again. The separate route never had this
problem because it inherits it from `exp` itself: `exp(z)` for `z <= 0` is
automatically in `(0, 1]`, and the narrow per-head intervals plus `pin_zero` kept
`frac(a>1)` at exactly 0. Fusing throws that structure away and asks a raw
polynomial to respect a **two-sided** bound it has no reason to respect.

Fixing it needs a form that cannot leave `[0, 1]`. `Q(x)^2` gives `>= 0` only;
`1 - S(x)^2` gives `<= 1` only; a reciprocal gives both but costs a division. No
non-constant polynomial is bounded on all of R, so any such guarantee is
interval-local — which is exactly why interval coverage dominates everything
here. **The fused factorisation is not obviously salvageable at depth 2.**

## 7c. The pattern, across all four gates

This is the through-line and it is worth stating on its own:

| gate | what actually broke it | the fix |
|---|---|---|
| `exp` | `P(0) = 0.966 != 1` -> spurious decay compounding to `0.966^1024` | pin `P(0) = 1` |
| `softplus` | `Delta < 0` on 415/576 heads -> `a > 1` -> expansion | `Delta = Q(x)^2`, non-negative by construction |
| fused `softplus+exp` | `a > 1` on 401/576 heads -> expansion | unsolved; needs a two-sided bound |
| `SiLU` after conv1d | no invariant identified yet; numerically fragile | unsolved |

**In every solved case the binding constraint was a structural invariant, not
pointwise accuracy** — and in two of three, the fix that worked had *worse*
pointwise error than the version that failed. Pointwise error has been a poor
predictor at every single step of this project. That suggests the productive
question for SiLU is not "how do I fit it better" but "what property must it
satisfy that a polynomial does not".

## 8. Next, in priority order

1. **SiLU-conv needs a different form, not a different margin.** The softplus fix
   came from finding the *structural* property that had to hold. What is the
   equivalent for SiLU? It is bounded below by −0.2785 and asymptotically linear
   above — a form like `x·σ̃(x)` with `σ̃` a bounded rational/squared
   approximation may respect that where a raw polynomial does not.
2. ~~Evaluate `FusedDtGate`~~ **Done — see §7b. It halves depth and costs +134
   perplexity, because it violates `a <= 1` on 401/576 heads.** Reviving it needs
   a two-sided-bounded form, which is an open design question.
3. **More measurement data.** All of the above rests on 6 × 1024 tokens. The
   interval tails are exactly what the fragility is sensitive to.
4. **RMSNorm**, as its own project.
5. **Fine-tuning.** Everything here is zero-training. The exp work showed MODE A
   recovers essentially nothing because there was no gap; here there are real
   gaps of +0.3 to +4, so fine-tuning has something to actually recover. This is
   the most likely route to making SiLU viable.

---

## 9. Segmented lookup tables: the gates the polynomials could not do (2026-10-27/28)

Prompted by ROSETTA (Yu et al., CCS '26, arXiv 2609.16915): hybrid CKKS/TFHE, where
a non-polynomial function is a TFHE lookup of `N_seg` segments x 2048 entries with
linear interpolation, and CKKS keeps the linear algebra. Two questions: does a
table of that shape preserve the model in plaintext, and what precision does it
actually get encrypted?

Code is on the `seglut-sim` branch (CRC: `/groups/tjung/jzhao7/FHEMAMBA-seglut`):
`real_mamba/seglut.py` (the table, plus RMSNorm / gated-norm modules whose
1/√v is a table), `eval/seglut_paired.py` (same shards and verdict rule as the
paired test above: 16 shards, significant if |mean| > 2·SEM),
`cluster/job_seglut*.sh`, results in `eval/seglut/*.json`. The encrypted
benchmark is `fhe/seglut_invsqrt/` (built and run on tjws-03).

### 9a. Plaintext tables: what matters is the edge of the table, not its precision

4 segments, 2048 entries each, linear interpolation. Boundaries are geometric for
1/√v (its input spans 6e6) and uniform otherwise; per-channel domains for SiLU /
softplus / fused dt come from `runs/gate_stats/gate_stats.json` (the same file the
polynomials were fitted on), widened 5%. Table precision alone is 17-28 bits for
every table, so any loss below comes from somewhere else.

Paired Δppl, L = 512 / 2048 (exact 27.82 / 21.54; n.s. = not significant):

| gate | clamp | **extend** (1/√v ranges calibrated on *train*) | wrap | clamp + 12-bit abs noise | extend + 12-bit abs noise |
|---|---|---|---|---|---|
| 1/√v, all 49 norms | −0.0000 n.s. / +0.0002 n.s. | +0.0001 n.s. / +0.0001 n.s. | +inf / +inf | +0.049 / +0.042 | +0.049 / +0.042 |
| fused `exp(A·softplus)` | +0.0003 n.s. / +0.0002 n.s. | +0.0001 n.s. / +0.0001 | +0.63 / +2.18 | +0.0002 n.s. / +0.0004 n.s. | −0.0000 n.s. / +0.0003 n.s. |
| softplus alone | −0.0004 n.s. / +0.0008 n.s. | **+inf / +inf** | +inf / +inf | +0.065 / +0.043 | +inf / +inf |
| SiLU after conv | +0.35 / +0.28 | **+0.011 / +0.012** | +22.8 / +17.0 | +0.35 / +0.28 | +0.011 / +0.012 |
| SiLU in gated norm | +0.14 / +0.10 | **+0.005 / +0.003** | +21.6 / +15.4 | +0.14 / +0.10 | +0.005 / +0.003 |
| **all of the above** | +0.58 / +0.45 | **+0.017 / +0.016** | +inf / +inf | +0.63 / +0.49 | +0.067 / +0.058 |

What the columns do with inputs outside the table (0.01-0.1% of them):
**clamp** holds the edge value, **extend** continues the end segments' lines,
**wrap** is what a negacyclic blind rotation literally computes (the sign flips).

1. **wrap kills everything.** A fraction of a percent of off-table inputs is enough.
   The encrypted protocol must clamp the table index.
2. **SiLU's whole loss under clamp was the clamp.** SiLU(x) ≈ x for large x, so
   holding the edge value throws away signal. Linear tails cut it 25-50x.
3. **The decay must come from the fused table, never softplus followed by exp.**
   Softplus alone diverges under extend (the lower tail goes negative, Δ < 0,
   a > 1) and under output-scaled noise; inside `all` the fused table supplies the
   decay (bounded in [0, 1]) and softplus only weights the input, so it is harmless.
   So: **fused dt + softplus clamped, SiLU extended, 1/√v either.**
4. Noise at 12 bits absolute costs +0.05 through the norms (`norm_f` is most of it).

Remaining gap: the SiLU / softplus / dt domains are still from validation
statistics; only the 1/√v ranges have been moved to train.

### 9b. Encrypted 1/√v in OpenFHE 1.2.1, 128-bit

`fhe/seglut_invsqrt/bench.cpp`, stock OpenFHE (nothing modified or installed),
CKKS ring 2^17 / depth 27, FHEW STD128 (n = 1305, N = 2048, q = 4096). 128
values sampled across the 4 segments of [0.0013, 16638]:

1. **segment index**: 3 comparisons, each CKKS→FHEW of (v − τ_k), large-precision
   `EvalSign`, FHEW→CKKS. One sign cannot resolve a 6e6 range, so each comparison
   has its own scale and is trusted only up to 1.2·τ_{k+1}; the masks are built so
   an untrusted comparison is always multiplied by a trusted 0.
2. **tables**, construct-then-select: per segment, an affine map to table
   positions, CKKS→FHEW, a manual modulus switch to q, and `EvalFunc` with a
   hand-built **negacyclic 2048-entry table** (one bootstrap). OpenFHE's own
   `GenerateLUTviaFunction` allows only q/256 = 8 entries.
3. **exact interpolation**: the index the bootstrap reads is ~9 positions from the
   true one, but that offset is a public-coefficient linear function of the key,
   so it is returned to CKKS exactly (to 0.003 positions) and the interpolation
   uses the index that was actually read.
4. **select** with mask products, after sharpening each comparison bit with
   x²(3 − 2x). Without it, a 2.5e-4 bit error times segment 0's outputs (up to
   ~45) leaked into segment 3's outputs (~0.008): up to 70% error.
5. **Newton** on the selected output, y ← y(3 − v·y²)/2, CKKS only, 4 levels.

| | rel. bits | worst segment | max rel. error | ROSETTA bits (−log2 RMSE) |
|---|---|---|---|---|
| table + interpolation | 5.5 | 5.3 | 9.5% | 3.8 |
| + 1 Newton step | 9.2 | 8.7 | 1.3% | 7.3 |
| + 2 Newton steps | **15.2** | 14.4 | 0.03% | 13.5 |

(TOY parameters, n = 32: 8.4 → 15.4 → 22.6.) Comparisons: 0 wrong out of 289.

**Why the table alone stops near 5 bits.** Every FHEW result is modulus-switched
back to q = 4096, which adds noise of √(n/18) ≈ 8.5 units (measured 7.5-9.7 in
every segment; 1.3 at TOY's n = 32, matching the same formula). FHEW→CKKS decodes
through a sine, so outputs must stay within ±0.2q ≈ ±820 units: ~7 bits of each
table's output range, minus ~2 more because 1/√v varies 7.7x inside a 60x-wide
segment. ROSETTA avoids the final modulus switch by packing the blind-rotation
output straight into CKKS, which OpenFHE does not expose.

**Why Newton is now alive.** §6 deferred RMSNorm and Newton (Path B) died because
no starting guess holds over v's 6e6 range. The table is that guess: ~5 bits
relative everywhere, from which each step roughly squares the error.

**How many bits are enough.** `--noise-scale rel` (new; error proportional to the
output, which is the shape the benchmark measured) on all 49 norms, small-scale
(8 shards x 2 blocks, L = 512, CPU): 5.2 bits **+0.25** (significant), 8.1 bits
+0.004 n.s., 12 bits +0.001 n.s. One Newton step is about at the threshold; two
clear it. (The older `range` mode is no proxy: it applies the largest output's
noise to every element and diverges at 8 bits.)

**Cost**, single core per value: 3 `EvalSign` at 3.4 s + 8 `EvalFunc` at 0.38 s
= 13.1 s of FHEW work, 77% of it the comparisons. Wall clock on 20 threads,
128 values: 753 s, of which FHEW→CKKS (15 batched calls) is 510 s. Key generation
48 s, peak memory 24 GB. Per token, the 49 norms are ~640 core-seconds of FHEW
work before conversions.

### 9c. RMSNorm through the benchmarked pipeline, out to L = 8192 (2026-10-30/31)

§9a's 1/√v is the ideal table (17+ bits, clamped). This replaces it with the
operator §9b actually built, simulated in plaintext by `real_mamba/ss_invsqrt.py`
(branch `prenorm-tfhe`, `--invsqrt-impl openfhe`): exact segment comparisons, a
read position off by N(0, 9.7) positions, phase noise of the same size, sine
decoding, a **negacyclic** read past either end of the table (it returns the
entry reflected about the table's midpoint, ~12x too large at v = 2τ_S), then
Newton and a 2^-15.3 CKKS noise floor. It reproduces the benchmark: 5.5 / 9.7 /
15.2 relative bits for 0 / 1 / 2 Newton steps against 5.5 / 9.2 / 15.2 measured
(`real_mamba/tests/test_ss_invsqrt.py`).

Staged pre-norms (24) → gated norms (24) → `norm_f` → all 49, 16 shards of
wikitext2 validation (30 / 7 / 3 / 1 blocks per shard at L = 512 / 2048 / 4096 /
8192), table range calibrated on wikitext2 *train*. `cluster/job_prenorm_tfhe.sh`,
job 1658582, results in `eval/prenorm_tfhe/*.json`. Paired Δppl (exact 27.82 /
21.54 / 20.73 / 22.66), all 49 norms:

| table range, Newton | L = 512 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| 6 x 1024 tokens, margin 0.05, 2 steps (as benchmarked) | +inf (7/16) | +inf (7/16) | +inf (6/16) | +inf (14/16) |
| **6 x 8192 tokens, margin 0.5, 2 steps** | **+0.0010 n.s.** | **+0.0000 n.s.** | **−0.0055 n.s.** | **+0.0081** (±0.0042) |
| 6 x 8192 tokens, margin 0.5, 1 step | +inf (1/16) | +0.0011 n.s. | +inf (1/16) | +0.041 |

(k/16 = shards that went to inf.) By stage, as benchmarked: pre-norms alone
|Δ| ≤ 0.0003 n.s. and `norm_f` alone 0.0000 at every length; gated norms alone
+0.021 / +0.016 / +0.015 / **+2.7** (±3.1). With the 8k range and 2 steps every
stage is within ±0.006 up to L = 4096, and the L = 8192 cost is all gated norms.

1. **A read past the table is fatal, not a small error.** As benchmarked, only
   0.02-0.15% of reads fall past the table, but Newton diverges from the reflected
   entry, the residual overflows fp16, and every later norm sees inf (up to 88% of
   `norm_f`'s inputs at L = 8192). §9a's ideal table, clamped, lost only +0.0001
   on the same inputs. A 2-shard CPU run had shown +0.006 here: too few shards to
   hit one.
2. **Gated-norm v grows with sequence length.** Calibrated on 1024-token blocks,
   the range misses it at 8192 (top 7.2e3; at 8192 tokens 1.4e4). **Calibrate at
   the longest length that will run.** Pre-norm v barely moves (8.4e3 → 1.0e4).
3. **Margin buys range only up to a few hundred x per segment.** A segment wider
   than that puts its low end too many read-noise positions apart for one guess:
   margin 1.0 (~1,800x per segment) sent the gated norms to +inf in a smoke run,
   and at margin 0.5 the worst gated table is already down to 9.9 relative bits
   after 2 steps (7.6 after 1; the median is 15.3 / 12.4). More range means more segments,
   i.e. more 3.4 s comparisons.
4. **One Newton step is not enough.** Pre-norms alone cost a small but
   significant +0.0003-0.0011, and the 7.6-bit tables let a handful of reads past
   the table (≤ 3e-4) blow up a shard.
5. **The margin is thin even in the passing row**: one read past the table in
   ~6e6 at L = 512, harmless that time. The table range comes from 6 train blocks;
   other text or longer contexts could fall outside it.

So: **4 segments + 2 Newton steps replaces all 49 RMSNorms at no measurable cost
to L = 4096 and +0.008 (0.04%) at L = 8192**, provided no input ever reaches the
negacyclic wrap. That has to be enforced, not calibrated.

### 9d. Next

1. **Range guard for 1/√v.** Two more comparisons (v < τ_0, v > τ_S) that clamp
   the table position to its ends before the lookup, ~7 s per norm. It turns a
   fatal read into §9a's clamp (+0.0001), and would let the margin and the
   per-segment span shrink back. Simulate it, rerun the 1024-token/0.05 row, then
   add it to `bench.cpp`.
2. **Cheaper segment selection.** Three large-precision signs are 77% of the FHEW
   cost; fewer, smaller-range comparisons (or more segments with small-precision
   signs) are the obvious lever.
3. **Batch the FHEW→CKKS conversions** (15 calls of 128 values now) and measure
   how the cost scales with values per call.
4. Move SiLU / softplus / dt domains to train-split statistics, then run the whole
   `all` configuration encrypted — SiLU is 98.5% of the lookups per token
   (43,008 + 36,864 of ~81,000), so its cost decides whether this is practical.
5. Depth: the table + 2 Newton steps use 24 of 27 levels, so in a full block this sits
   right after a CKKS bootstrap.
