# baby_mamba — the Mamba-2 transition, small enough to read

This folder is Parts 1–4 of the project. It contains **no** language model, **no**
pretrained weights and **no** FHE. It exists so that you understand the one line
we are attacking before we attack it in a real 130M-parameter model.

## The idea in four sentences

**`A` controls how quickly memory decays.**
**`delta` is input-dependent.**
**`exp(A * delta)` becomes a number controlling how much previous state survives.**
**Since input is private in FHE, this transition value becomes encrypted too.**

That last sentence is the whole reason this project exists. If `A * delta` were a
fixed constant we could precompute `exp` of it and never think about it again.
It is not: `delta` comes from the token, so under FHE `A*delta` is a ciphertext,
so `exp` has to be evaluated *homomorphically* — and CKKS cannot do `exp`. It can
only add and multiply. So we need a polynomial.

## Run it

```bash
python -m baby_mamba.demo_transition
```

One command. It prints `u`, `A`, `delta`, `z = A*delta`, `a = exp(z)`, and the
resulting recurrent states, for a model with `batch=2, seqlen=8, nheads=4,
headdim=2, d_state=2`. Every number on screen came from
[`transition.py`](transition.py), which you can read in five minutes.

Then see what a polynomial does to it:

```bash
python -m baby_mamba.demo_transition --poly 4
```

## The files

| file | what it is |
|---|---|
| [`transition.py`](transition.py) | the transition + recurrence in plain PyTorch, with the upstream line number next to each step |
| [`polynomial.py`](polynomial.py) | polynomial fitting (Chebyshev / Lobatto / Remez / weighted least squares) **and** the FHE depth accounting |
| [`demo_transition.py`](demo_transition.py) | Part 1: print everything |
| [`error_propagation.py`](error_propagation.py) | Part 4: show that a small gate error is not a small model error |
| [`tests/`](tests) | 38 tests. `python -m pytest baby_mamba/tests -q` |

## What is faithful, and what is simplified

Faithful (checked against the real source — see
[`../PART0_SOURCE_INSPECTION.md`](../PART0_SOURCE_INSPECTION.md)):

* `A = -exp(A_log)`, shape `(nheads,)` — **one scalar per head**, not one per
  state dimension. This surprises people.
* `delta = softplus(proj(u) + dt_bias)`, shape `(batch, seqlen, nheads)`, with
  `dt_bias` initialised exactly as upstream does it (softplus-inverse of a
  log-uniform sample in `[0.001, 0.1]`).
* `z = A * delta ≤ 0` and, for the real model, `a = exp(z) ∈ (0, 1]`.
* `h_t = a_t·h_{t-1} + b_t` with `b_t = delta_t · B_t ⊗ x_t`, then
  `y_t = C_t·h_t + D·x_t`. These mirror `mamba2.py:313-319` line for line.
* The packed `[z, x, B, C, dt]` input projection layout.

Simplified on purpose:

* **No depthwise causal conv1d before the SSM.** Upstream runs a 4-wide causal
  conv + SiLU over `[x, B, C]`. It changes *what* `x/B/C` are, not *how* the
  transition works. The real-model path in [`../real_mamba/`](../real_mamba) does
  include it.
* **No RMSNormGated, no output projection, no residual stream.** Same reason.
* **A serial `for t in range(seqlen)` loop.** Upstream never does this; it is
  chunked and fused. Serial is 100× slower and 100× clearer.

Untouched, deliberately, per the project scope: **SiLU, RMSNorm, softplus, and
the scan structure itself.**

## Part 2 — fitting the polynomial

```bash
python fit_exp_polynomial.py --degree 4 --xmin -8 --xmax 0 --compare
```

Four methods, none of them Taylor (Taylor's error grows like `|z|^(n+1)`, which
is exactly wrong on a wide interval):

* `chebyshev` — interpolation at Chebyshev roots. Near-minimax, never fails. **Default.**
* `lobatto` — interpolation at Chebyshev–Lobatto nodes, which include the
  endpoints, so with `--xmax 0` you get `P(0) = 1` exactly.
* `remez` — true minimax, our own Remez exchange. Best max-error. Raises rather
  than returning garbage.
* `lstsq` — weighted least squares. Best RMSE, and the only one that supports
  `--weight relative`.

Measured on `[-8, 0]` (this is the math, not our code, so the numbers are stable):

| degree | ct-ct depth | max abs err | RMSE | `P(0)` | frac `P(z) < 0` |
|---|---|---|---|---|---|
| 2 | 1 | 2.78e-01 | 8.39e-02 | 0.7219 | 0.403 |
| 3 | 2 | 1.04e-01 | 3.65e-02 | 0.8959 | 0.298 |
| 4 | 2 | 3.37e-02 | 1.31e-02 | 0.9663 | 0.169 |

Two things in that table should bother you, and they are the reason Part 4 exists:

1. **`P(0) = 0.9663`, not 1.** `exp(0) = 1` means "remember the previous state
   perfectly". A degree-4 fit instead multiplies by `0.9663` every step.
   `0.9663^1024 ≈ 5e-16`. All long-range memory, gone — from an error that looks
   like 3%. Fix it with `--pin-zero`.
2. **`P(z) < 0` on 17% of the interval.** A negative decay factor flips the sign
   of the entire head state, every step. `exp` can never do that.

## Part 3 — why the evaluation order matters

`P4(z) = c0 + c1·z + c2·z² + c3·z³ + c4·z⁴`, built as:

```
z2 = z * z        # ciphertext × ciphertext, depth 1
z3 = z2 * z       # ciphertext × ciphertext, depth 2
z4 = z2 * z2      # ciphertext × ciphertext, depth 2
P4 = c0 + c1*z + c2*z2 + c3*z3 + c4*z4      # ciphertext × PLAINTEXT, no extra depth
```

**Not Horner.** Horner uses one fewer multiplication but they are sequential, so
its depth is 4 instead of 2. Under CKKS, depth is what forces bigger parameters
or a bootstrap; the raw multiplication count barely matters. `PowerSchedule` in
[`polynomial.py`](polynomial.py) *computes* the depth from the graph rather than
asserting it in a comment, and the tests check `ceil(log2(degree))` for degrees
2–32.

| degree | ct-ct mults | ct-ct depth | Horner depth |
|---|---|---|---|
| 2 | 1 | **1** | 2 |
| 3 | 2 | **2** | 3 |
| 4 | 3 | **2** | 4 |

Multiplying by `c_i` is ciphertext-times-plaintext: cheap, and it does not
consume a fresh ct-ct level, so it is not counted.

## Part 4 — the part that actually matters

```bash
python -m baby_mamba.error_propagation
python -m baby_mamba.error_propagation --method lobatto --pin-zero
python -m baby_mamba.error_propagation --delta-scale 6
```

> The polynomial does not merely approximate one activation.
> Its output controls how memory is repeatedly multiplied through time.
> Therefore long-sequence stability matters more than pointwise
> approximation error alone.

Measured, degree 4, Chebyshev on `[-8, 0]`, `delta_scale = 1`:

| L | mean \|Δa\| | relative error of final state |
|---|---|---|
| 16 | 2.3e-02 | 6.2e-02 |
| 64 | 2.3e-02 | 4.9e-01 |
| 256 | 2.3e-02 | 4.6e-01 |
| 1024 | 2.3e-02 | 4.3e-01 |

The gate error is flat at 2.3%. The state error is **7× larger at L=1024 than at
L=16** and lands near 50%. Adding `--pin-zero` (`P(0) = 1` enforced) cuts the
L=1024 state error from 0.43 to **0.13** — a 3.4× improvement bought with one
degree of freedom and zero extra FHE depth. That is the single best lever we
found in Part 4.

And the failure mode you must not walk into — `--delta-scale 6` pushes `z` down
to `-24.75`, outside the `[-8, 0]` fit interval:

| poly | min `a` at L=1024 | frac `a < 0` |
|---|---|---|
| P2 | -0.06 | 0.059 |
| P3 | **-45.8** | 0.099 |
| P4 | -0.01 | 0.011 |

A polynomial does not decay outside its fit interval — it **diverges**. `min a =
-45.8` means one timestep multiplied that head's entire state by −45.8. This is
why Part 6 (measure the real `z` distribution on the real model) has to happen
before Part 7, and not after.

## Success and failure for this folder

**Success:** the demo runs and you can point at the line where `a` is computed;
all 38 tests pass; you can explain why `P(0) ≠ 1` is worse than it looks.

**Failure:** you can only say "we replaced exp with a polynomial and the error is
3%". That number is true and almost irrelevant.
