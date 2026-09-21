# Replacing the *other* non-polynomial gates

Follow-on to the exp work. Status after the first pass: **exp turned out to be the
easy one.**

| gate | instances | status | best Δ perplexity | depth |
|---|---|---|---|---|
| `exp(A·Δ)` | 1 / layer | **solved** (previous work) | **+0.0002** | 2 |
| `softplus` → `Δ` | 1 / layer | **solved, this pass** | **+0.278** | 2 |
| `SiLU` in the gated norm | 1 / layer | marginal | +1.69 | 2 |
| `SiLU` after conv1d | 1 / layer | **not solved — unstable** | +3.95, but see below | 2 |
| `RMSNorm` 1/√· | **2 / layer** + `norm_f` | **not attempted** | — | 6–15 est. |

Measured on `state-spaces/mamba2-130m`, wikitext-2, degree 4, per-channel
intervals, no training. Baseline (everything exact) 18.1368.

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
- softplus **+0.278** ✓
- SiLU-norm **+1.69** ⚠
- SiLU-conv **+3.95** at a knife-edge margin ✗
- all four together: **inf** — the errors compound

So the honest position: **two of four solved, one marginal, one unsolved, one
untouched.** The exp result does not generalise to the other gates for free, and
the reason differs per gate — a sign invariant for softplus, numerical fragility
for SiLU-conv, dynamic range for RMSNorm.

## 8. Next, in priority order

1. **SiLU-conv needs a different form, not a different margin.** The softplus fix
   came from finding the *structural* property that had to hold. What is the
   equivalent for SiLU? It is bounded below by −0.2785 and asymptotically linear
   above — a form like `x·σ̃(x)` with `σ̃` a bounded rational/squared
   approximation may respect that where a raw polynomial does not.
2. **Evaluate `FusedDtGate`** (implemented, untested): one per-head polynomial
   for `x → exp(A·softplus(x+b))`, halving that path's depth from 4 to 2 and
   removing the |A|-amplification of softplus error.
3. **More measurement data.** All of the above rests on 6 × 1024 tokens. The
   interval tails are exactly what the fragility is sensitive to.
4. **RMSNorm**, as its own project.
5. **Fine-tuning.** Everything here is zero-training. The exp work showed MODE A
   recovers essentially nothing because there was no gap; here there are real
   gaps of +0.3 to +4, so fine-tuning has something to actually recover. This is
   the most likely route to making SiLU viable.
