# STEP 1 — the usable CKKS epoch after bootstrapping

**Status (2026-10-06): measured.** The EPOCH search covers N = 2^16 and 2^17 on
OpenFHE 1.2.1 (tjws-03) and 1.6.0, which agree row for row. The recommended
configuration was then **bootstrapped for real at N = 2^17** on tjws-03: the EPOCH
was confirmed by consuming levels, and precision, repeated-bootstrap decay and key
memory were measured. Results are in `epoch_results.json`.

## Result 0 — the verification run (N = 2^17, Δ = 2^59, {3,3}, 128-bit, OpenFHE 1.2.1)

This is the headline. Everything else in this file is context for it.

| | uniform secret | sparse secret |
|---|---|---|
| bootstrap consumes | 20 levels | 16 levels |
| **EPOCH, declared** | **23** | **27** |
| **EPOCH, consumed** (multiplies that still decrypt correctly) | **23** ✓ | **27** ✓ |
| **precision after 1 bootstrap**, −log2(max abs err) | **8.0 bits** | **15.3 bits** |
| precision, −log2(mean abs err) | 11.4 bits | 18.2 bits |
| precision after bootstraps 1→5 | 8.0, 8.2, 8.1, 8.3, 8.1 | 15.3, 15.2, 15.2, 15.5, 15.3 |
| precision across all EPOCH levels | flat, 8.0–8.4 | flat, 14.9–15.5 |
| log2(QP) / 128-bit bound | 3497 / 3523 (margin 26) | 3497 / 3523 (margin 26) |
| **evaluation keys** (serialized) | **55.9 GB** (55.6 rotation + 0.35 relin) | 55.9 GB |
| **peak host memory** | **93.6 GB** | **94.7 GB** |
| one bootstrap, setup, keygen *(CPU-only, 20 threads, non-transferable)* | 96 s, 22 s, 59 s | 88 s, 24 s, 65 s |

Three findings, in order of how much they change the plan:

1. **Keys do not fit the GPU target.** 55.9 GB of evaluation keys against a 24 GB
   card, before any rotation keys the model's own matrix-vector products will need.
   Almost all of it (55.6 GB) is the bootstrapping rotation keys.
2. **Precision is much lower at real N than at toy N.** At N = 2^12 the same
   settings gave 14.6 / 22.8 bits; at 2^17 they give **8.0 / 15.3**. The maximum
   is taken over 65,536 slots instead of 2,048, and bootstrap error grows with N.
   **8 bits** is a max error of 2^−8 ≈ 0.004 on values in [−1, 1]. Whether the model
   tolerates that is unmeasured.
3. **Neither the EPOCH nor the precision decays.** Declared and consumed EPOCH agree
   exactly. Precision is flat across 5 consecutive bootstraps and across every
   level of the EPOCH. A 24-layer model that refreshes 24+ times loses nothing to
   repetition: the cost of a bootstrap is a fixed precision floor, not a drift.

## What the EPOCH is, in one paragraph

A CKKS ciphertext can survive only a fixed number of multiplications ("levels").
Bootstrapping refreshes it, but bootstrapping uses up levels itself. The **EPOCH**
is what's left over: how many multiplications you get between two refreshes, at
128-bit security. Every depth estimate in this project — 2 levels for the exp gate,
~10 for the scan, ~14–31 for a block — only matters relative to this number.

## Result 1 — EPOCH by configuration (128-bit, HEStd_128_classic, FLEXIBLEAUTO)

`boot` = levels the bootstrap itself consumes. `QP / bound` = log2 of the
total modulus against the HE-standard maximum for that ring dimension.

| N | scale bits | secret | level budget | boot | **EPOCH** | log2 QP / bound | margin |
|---|---|---|---|---|---|---|---|
| 2^16 | 40 | sparse | {3,3} | 16 | **14** | 1722 / 1747 | 25 |
| 2^16 | 40 | sparse | {4,4} | 18 | **12** | 1722 / 1747 | 25 |
| 2^16 | 40 | uniform | {3,3} | 20 | **10** | 1722 / 1747 | 25 |
| 2^16 | 40 | uniform | {4,4} | 22 | **8** | 1722 / 1747 | 25 |
| 2^16 | 50 | sparse | {3,3} | 16 | **8** | 1732 / 1747 | 15 |
| 2^16 | 50 | sparse | {4,4} | 18 | **6** | 1732 / 1747 | 15 |
| 2^16 | 50 | uniform | {3,3} | 20 | **4** | 1732 / 1747 | 15 |
| 2^16 | 50 | uniform | {4,4} | 22 | **2** | 1732 / 1747 | 15 |
| 2^16 | 59 | sparse | {3,3} | 16 | **4** | 1661 / 1747 | 86 |
| 2^16 | 59 | sparse | {4,4} | 18 | **2** | 1661 / 1747 | 86 |
| 2^16 | 59 | uniform | {3,3} | 20 | does not fit | - / - | - |
| 2^16 | 59 | uniform | {4,4} | 22 | does not fit | - / - | - |
| 2^17 | 40 | sparse | {3,3} | 16 | **48** | 3501 / 3523 | 22 |
| 2^17 | 40 | sparse | {4,4} | 18 | **46** | 3501 / 3523 | 22 |
| 2^17 | 40 | uniform | {3,3} | 20 | **44** | 3501 / 3523 | 22 |
| 2^17 | 40 | uniform | {4,4} | 22 | **42** | 3501 / 3523 | 22 |
| 2^17 | 50 | sparse | {3,3} | 16 | **34** | 3452 / 3523 | 71 |
| 2^17 | 50 | sparse | {4,4} | 18 | **32** | 3452 / 3523 | 71 |
| 2^17 | 50 | uniform | {3,3} | 20 | **30** | 3452 / 3523 | 71 |
| 2^17 | 50 | uniform | {4,4} | 22 | **28** | 3452 / 3523 | 71 |
| 2^17 | 59 | sparse | {3,3} | 16 | **27** | 3497 / 3523 | 26 |
| 2^17 | 59 | sparse | {4,4} | 18 | **25** | 3497 / 3523 | 26 |
| 2^17 | 59 | uniform | {3,3} | 20 | **23** | 3497 / 3523 | 26 |
| 2^17 | 59 | uniform | {4,4} | 22 | **21** | 3497 / 3523 | 26 |

`firstMod = scale + 1` throughout, as in every OpenFHE bootstrapping example.
At N = 2^16 a uniform secret does not fit at all: the library reports that even one
level after bootstrapping needs N = 2^17.

## Result 2 — precision at a toy ring dimension (N = 2^12, INSECURE, smoke only)

**Superseded by Result 0 for N = 2^17. Kept because it shows the trend across scales,
and because it is too optimistic by 7 bits — a warning about toy-N measurements.**

Full slot count, values uniform in [−1, 1], −log2(max abs error):

| scale | uniform secret | sparse secret |
|---|---|---|
| 59 | **14.6 bits** | **22.8 bits** |
| 50 | 5.2 bits | 10.8 bits |
| 40 | decode fails | 0.8 bits |

A fresh encryption, before any bootstrap, decrypts to ~34–43 bits, so the loss is
the bootstrap, not the measurement. This has to be re-measured at N = 2^17.

**Precision does not decay across refreshes** (sparse, scale 59, N = 2^12):
22.8 → 22.6 → 22.8 → 22.8 → 22.5 bits over 5 consecutive bootstraps with nothing
in between. It also holds at ~22.8 bits through all 10 multiplies of the epoch.
That matters for a 24-layer model that refreshes at least 24 times — but it is
one configuration at a toy ring dimension, and needs repeating at 2^17.

## The tradeoff, stated plainly

**Only Δ = 2^59 gives usable precision after bootstrapping — and 59 bits per level
is also what makes the EPOCH small.** The large EPOCHs in the table (42–48 at
scale 40) are bought with bootstraps whose output is noise. Reporting them as the
answer would be exactly the "tune parameters to make the EPOCH look large"
the brief forbids.

## Decision gate

| | EPOCH at Δ = 2^59 | case |
|---|---|---|
| N = 2^16, any secret | ≤ 4 | **< 15 → ruled out for this model** |
| N = 2^17, uniform secret | 21–23 | **15–26 → refresh must bracket the scan** |
| N = 2^17, sparse secret | 25–27 | 15–26, with {3,3} landing exactly on ~27 |

**We are in the middle case, now measured: N = 2^17, EPOCH = 23 (uniform) or 27
(sparse).** With the scan costing log2(L) ≈ 10–11 levels (Sklansky, L = 1024–2048),
the rest of a block gets EPOCH − scan ≈ **12–13 levels (uniform) or 16–17 (sparse)**
per refresh.

**Recommended configuration:** N = 2^17, Δ = 2^59, level budget {3,3}, **uniform
secret** — EPOCH 23, 8.0 bits. Sparse buys 4 levels and 7 bits, but its 128-bit
security is not established by the check OpenFHE performs (first caveat below).

**The decision gate is passed on levels and not yet passed on precision or memory.**
8 bits may be too few, and 56 GB of keys is 2.3× the GPU. Both have known levers,
listed below; neither has been tried.

## Levers, untried

* **Precision: iterative bootstrapping.** OpenFHE's `EvalBootstrap(ct, 2, precision)`
  runs a second correction pass and roughly doubles precision (per OpenFHE's
  `iterative-ckks-bootstrapping` example), at the cost of one level and about 2×
  bootstrap time. Uniform 8 → ~16 bits for EPOCH 23 → 22, if it holds at 2^17.
* **Precision requirement: measure it, don't guess.** Inject noise at 2^−8 and 2^−15
  at every refresh point in the plaintext model and read the paired Δppl. That says
  whether 8 bits is already enough.
* **Key memory: a larger level budget.** {4,4} or {5,5} spends more levels per
  bootstrap on the linear transforms and needs fewer rotation keys. That trades
  EPOCH for memory, and the size of the trade is unmeasured.
* **Key memory: fewer slots.** Bootstrapping keys scale with the slot count being
  refreshed; a sparsely packed ciphertext needs fewer.

## What is still uncertain

1. **Sparse-secret security.** OpenFHE checks sparse secrets against the same
   HE-standard table as uniform ones. That table assumes a uniform ternary secret,
   so this check does not establish 128-bit security for a sparse secret. The
   sparse rows' extra 4 levels and 8 bits come with an open security question.
2. **One configuration verified.** {3,3} only, both secrets. {4,4} would trade
   ~2 levels for less key memory; not yet measured.
3. **Library version.** The EPOCH search agrees exactly between 1.2.1 and 1.6.0.
   The verification run is 1.2.1 only. Precision differed slightly between versions
   in smoke tests (6.8 vs 5.2 bits at scale 50, uniform, N = 2^12).
4. **GPU library.** Phantom has no bootstrapping; DESILO is commercial. CPU timings
   say nothing about GPU latency, and none are used in any conclusion here.

## How to run

```bash
cmake -S fhe -B fhe/build -DOpenFHE_DIR=/usr/local/lib/OpenFHE   # or your install prefix
cmake --build fhe/build -j
python3 fhe/run_epoch_sweep.py --smoke            # ~10 s, insecure: does the pipeline work?
python3 fhe/run_epoch_sweep.py --search-only --logn 16 17   # minutes: Result 1
python3 fhe/run_epoch_sweep.py --logn 17 --scales 59 --budgets 3,3   # the verification run, needs ~100+ GB RAM
```

## Defects found while building this — worth knowing before trusting any number

* **OpenFHE silently returned garbage** at scale 50 with a uniform secret and
  `firstMod = 60`: −2.4 bits, no exception raised. Every precision number here
  is therefore checked against the input, never inferred from a successful call.
* A fixed `firstMod = 60` is wrong below scale 59: at scale 40 the q0/Δ gap
  exceeds the bootstrap's correction factor and the library throws.
* The library accepts one more multiply than the declared EPOCH (it moves into
  −1 levels remaining), and that extra ciphertext cannot be decrypted. Both counts
  are recorded; only the "correct" one is an EPOCH. Checked at N = 2^12, sparse,
  scale 59: declared 10, accepted 11, decrypting correctly 10.
