# STEP 1 — the usable CKKS epoch after bootstrapping

**Status (2026-10-06): the EPOCH search is done at N = 2^16 and 2^17. Precision has
only been measured at a small, insecure N = 2^12. The 2^17 verification run — a
real bootstrap, an empirical level count, precision after 1 and 5 bootstraps, and
key memory — is still pending.** Until that run exists, every EPOCH below is
*declared by the library*, not yet *consumed*.

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

## Result 2 — precision after one bootstrap (N = 2^12, INSECURE, smoke only)

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

**We are in the middle case: N = 2^17, EPOCH ≈ 21–27.** With the scan costing
log2(L) ≈ 10–11 levels (Sklansky, L = 1024–2048), the rest of a block gets
EPOCH − scan ≈ **10–17 levels** per refresh, depending on the secret distribution.

**Recommended configuration (provisional):** N = 2^17, Δ = 2^59, level budget
{3,3}. Uniform secret if the security argument has to be the HE standard
(EPOCH 23, ~15 bits); sparse if a separate security analysis is accepted
(EPOCH 27, ~23 bits). See the first caveat below.

## What is still uncertain

1. **Sparse-secret security.** OpenFHE checks sparse secrets against the same
   HE-standard table as uniform ones. That table assumes a uniform ternary secret,
   so this check does not establish 128-bit security for a sparse secret. The
   sparse rows' extra 4 levels and 8 bits come with an open security question.
2. **Nothing at N = 2^17 has been bootstrapped yet.** The EPOCH is declared, not
   consumed; precision and key memory at 2^17 are unknown. Key memory decides
   whether this fits the 24 GB GPU target at all.
3. **Library version.** Measured with OpenFHE 1.6.0. The workstation install may be
   older, and bootstrapping depth has changed between versions.
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
