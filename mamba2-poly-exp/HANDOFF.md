# Handoff: FHE-friendly Mamba-2 transition

Everything you need to pick this up. Read this first, then
[`FINDINGS.md`](FINDINGS.md) for the evidence, then
[`PART0_SOURCE_INSPECTION.md`](PART0_SOURCE_INSPECTION.md) for why the code is
shaped the way it is.

---

## 1. The one-sentence version

Mamba-2 gates its recurrent memory with `a = exp(A·Δ)`. Fully Homomorphic
Encryption can only add and multiply, so `exp` has to go. **We replaced it with a
degree-4 polynomial — one polynomial per attention head — and the pretrained model
is statistically indistinguishable from the original, with no retraining.**

---

## 2. The models

All checkpoints are the official `state-spaces/mamba2-*` weights, unmodified.
Architecture values below are derived from each `config.json` plus the `Mamba2`
defaults (`expand=2, headdim=64, d_state=128, ngroups=1, d_conv=4, chunk_size=256`);
the parameter counts reproduce the published sizes.

| model | params | `d_model` | layers | `d_inner` | heads/layer | `headdim` | `d_state` |
|---|---|---|---|---|---|---|---|
| **mamba2-130m** ← primary | 129.0 M | 768 | 24 | 1536 | 24 | 64 | 128 |
| mamba2-370m | 368.3 M | 1024 | 48 | 2048 | 32 | 64 | 128 |
| mamba2-780m | 780.2 M | 1536 | 48 | 3072 | 48 | 64 | 128 |
| mamba2-1.3b | 1343.8 M | 2048 | 48 | 4096 | 64 | 64 | 128 |

Tokenizer: `EleutherAI/gpt-neox-20b`, vocab 50,277 padded to 50,288.
Most work is on **130m**; 370m/780m/1.3b were used to show the result is not
scale-specific. 2.7b exists and was not tried.

---

## 3. What exactly was replaced

### The target

Inside every Mamba-2 mixer, per token and per head:

```
dt_raw          one slice of the packed input projection      (B, L, nheads)
Δ  = softplus(dt_raw + dt_bias)              > 0              (B, L, nheads)
A  = -exp(A_log)                             < 0              (nheads,)
z  = A · Δ                                   ≤ 0              (B, L, nheads)
a  = exp(z)                                  ∈ (0, 1]         (B, L, nheads)   ← REPLACED
h_t = a_t · h_{t-1} + b_t                                     (B, nheads, headdim, d_state)
```

**`a = exp(z)` is the only thing replaced.** Reference: `Mamba2.step()` at
`third_party/mamba/mamba_ssm/modules/mamba2.py:313-319`, the one place upstream
writes it in this form.

### How small the target is

| model | `A` scalars (the whole decay parameterisation) | as % of params | `dt` slice / projected channels |
|---|---|---|---|
| 130m | 576 | 0.00045% | 24 of 3,352 |
| 370m | 1,536 | 0.00042% | 32 of 4,384 |
| 780m | 2,304 | 0.00030% | 48 of 6,448 |
| 1.3b | 3,072 | 0.00023% | 64 of 8,512 |

`A` is **one scalar per head** — not one per state dimension. This surprises
people and it is the key to the whole result.

### The replacement

```
a = P_h(z) = c₀ + c₁·t + c₂·t² + c₃·t³ + c₄·t⁴          where t = z / s_h
```

* **per head `h`**: its own coefficients `c₀..c₄` and its own scale `s_h = |z_min,h|`
* fitted offline by **Chebyshev interpolation** on that head's own measured range of `z`
* `P_h(0) = 1` pinned exactly (Chebyshev–Lobatto nodes include the endpoint)
* heads whose `exp(z)` underflows over their whole range get the exact constant `0`

Evaluated **not** by Horner, but as explicit powers so the encrypted depth is minimal:

```
z2 = t · t        ciphertext × ciphertext,  depth 1
z3 = t2 · t       ciphertext × ciphertext,  depth 2
z4 = t2 · t2      ciphertext × ciphertext,  depth 2
P  = c₀ + c₁t + c₂z2 + c₃z3 + c₄z4          ciphertext × PLAINTEXT, no extra depth
```

**3 ct-ct multiplications, sequential depth 2.** Horner would be depth 4. Depth is
the FHE currency; multiplication count is not.

### Why per head, and why it is free

`|A|` spans 5–7 orders of magnitude across heads *within the same model*
(130m: 4.0e−04 … 3.6e+04), so `z` per head ranges from `[-0.004, 0]` to
`[-1.8e5, -0.47]`. **A single global interval gives infinite perplexity at every
model scale.** But `A` is a *weight*, so under FHE it is plaintext — per-head
coefficients are plaintext constants. Same degree, same circuit, same depth.
Fitting all 576 polynomials takes 0.04 s on a laptop.

### What was NOT touched

`SiLU` · `RMSNorm` / `RMSNormGated` · `softplus` · the depthwise causal conv1d ·
the `D` skip · the chunk decomposition · the residual stream · the embedding ·
the LM head · **every weight value** · and `A = -exp(A_log)` itself (an `exp` of a
*parameter* — plaintext under FHE, therefore free). `torch.exp` is never patched
globally.

### One structural change you must understand

The production Triton kernel **never computes `a` per timestep**. It computes
`cumsum(A·Δ)` and exponentiates *differences of prefix sums*, exploiting
`exp(Σz) = Π exp(z)`. That identity is false for any polynomial. So the scan was
rewritten in **per-step product form** — every `cumsum` became a `cumprod`, every
`exp(segsum(·))` a `segprod(·)` (`real_mamba/reference_ssd.py`).

This is verified, not assumed: **relative error 2.9e−04 against the official
`mamba_chunk_scan_combined` kernel**, and 1e−12 against the official pure-PyTorch
`ssd_minimal_discrete`.

---

## 4. The result

Degree 4, per-head, depth 2, **no training**:

| test | result |
|---|---|
| 4 model scales, 130m → 1.3b | Δ perplexity ≤ 0.006 |
| held-out domains (The Pile, LAMBADA), intervals fitted on wikitext-103 only | +0.013 / +0.011 |
| sequence length to 8192 | −0.012 |
| LAMBADA zero-shot accuracy | −0.0013 (1 example in 800) |
| paired bootstrap, 95% CI | **[−0.000065, +0.00015] nats — indistinguishable from `exp`** |
| margin hyperparameter, 0 → 1.0 | 0.007 perplexity spread |
| a single **global** interval | **inf at every scale** |

Degree 2 (depth 1) works in-domain but is *not* defensible out of it.

---

## 5. Code map

| path | what |
|---|---|
| `baby_mamba/` | teaching sandbox: the transition in ~200 readable lines, polynomial fitting, FHE depth accounting, error-propagation demo |
| `real_mamba/reference_ssd.py` | **the core**: per-step product-form SSD with a swappable `z → a` |
| `real_mamba/patch.py` | surgical replacement of `Mamba2.forward`, reusing the layer's own weights |
| `real_mamba/transitions.py` | `--transition exact\|poly2\|poly3\|poly4`, `--interval-mode global\|per-head` |
| `real_mamba/model.py` | checkpoint loading; two backends (official `mamba-ssm`, and a pure-PyTorch fallback that reads the same weights) |
| `collect_transition_stats.py` | **run this first** — measures `z` per head |
| `eval_poly_exp.py` · `stability_check.py` | zero-training evaluation and stability |
| `cross_domain_eval.py` · `robustness_suite.py` | the generalisation campaign |
| `finetune_poly_exp.py` · `distill_poly_exp.py` | adaptation (modes A/B/C, and KD) |
| `cluster/` | CRC/SGE jobs. See `cluster/README.md` |

76 tests: `python -m pytest baby_mamba/tests real_mamba/tests -q` (63 pass
anywhere, 13 need CUDA).

---

## 6. Running it

```bash
pip install -r requirements.txt
python -m pytest baby_mamba/tests real_mamba/tests -q     # 63 pass, ~2 s
./run_all.sh                                              # Stages 0-8, ~15 min on a laptop CPU
```

**Order matters: measure before you fit.**

```bash
python collect_transition_stats.py --blocks 64            # measure z per head
python eval_poly_exp.py --sweep --blocks 120              # then evaluate
```

On CRC (set `$FHEMAMBA_ROOT` per `cluster/README.md`; queue `gpu@@jung_gpu`, 1 RTX 6000):

```bash
qsub cluster/job_preflight.sh      # ALWAYS first: proves the port matches the official kernel
qsub cluster/job_e1_crossdomain.sh # the load-bearing generalisation test
qsub cluster/job_e2_scale.sh
qsub cluster/job_e3456_robustness.sh
```

Resource envelope, measured: evaluation peaks at **2.79 GB** and 16–22 s for 244
blocks; fine-tuning at **11–11.6 GB** and ~200 s per 1M tokens. Comfortable on one
24 GB card.

---

## 7. Traps that will cost you a day

1. **`use_mem_eff_path=False` does NOT un-fuse the model.** Both branches call
   Triton. Even `softplus` is inside the kernel. You cannot see `z` from Python on
   any default path.
2. **The RTX 6000 (sm_75) compiles the mamba SSD kernel in fp16 ONLY.** fp32 and
   bf16 both die with `IndexError: map::at` inside Triton. And
   `torch.cuda.is_bf16_supported()` returns `True` on that card anyway — it reports
   driver support, not tensor-core support. Use `real_mamba.model.recommended_dtype`.
3. **`causal_conv1d` is not installed on CRC**, and *both* branches of
   `Mamba2.forward` need it — the `use_mem_eff_path=False` fallback is broken
   upstream in `mamba_ssm` 2.2.2 (`self.dconv` typo). Whole-mixer parity tests are
   skipped for this reason; scan-level parity is not affected.
4. **Never install from inside a batch job.** A `conda create` in a job script once
   filled `$HOME` to 0 bytes, which endangers every other job running. `cluster/env.sh`
   now verifies and refuses.
5. **`ssh host '...$VAR...'` is expanded by the remote LOGIN shell** before your
   `bash -c` sees it. This silently copied `hub/` into `hub/`. Write a script file
   and `scp` it.
6. **`numpy`'s `Chebyshev.convert().coef` trims trailing zeros**, so a wide-interval
   fit returns fewer coefficients and a naive `PolyExp` would report the wrong
   degree *and the wrong FHE depth*. Handled by `_pad_to_degree`; do not remove it.
7. **Per-head coefficients span 1e45 in the raw `z` basis.** They are stored
   scale-normalised for that reason. Un-normalising them will break training and
   overflow fp32.

---

## 8. Open items, in priority order

1. **Re-fit after fine-tuning.** Training `A_log`/`dt_bias` moves `z` off the
   intervals the polynomials were fitted on. MODE A survives (`frac(a>1)=0`); MODE B
   and C push `frac(a>1)` to 4.4% and 2.1%. **The best perplexity we recorded
   (16.78, MODE C) has the second-worst-behaved gate.** The fix — re-measure, re-fit
   — is ~20 GPU-minutes and untested. Do this first.
2. **Real CKKS numbers.** "Depth 2" is a proxy from the evaluation graph. No
   parameters, noise budget, bootstrap placement or latency has been computed. This
   is the natural next project and the biggest gap in the story.
3. **`frac(a < 0) ≈ 0.008`.** One token in 120 gets a negative decay, impossible for
   `exp`. It does not move perplexity, state norms or LAMBADA accuracy, and we have
   no mechanistic account of why.
4. Odd degrees have an unbounded negative tail (poly3 hit `a = −130.7` on The Pile,
   at the same depth as poly4). Prefer degree 4.
5. Breadth: non-English, mamba2-2.7b, other architectures.
6. Replacing the *other* nonlinearities — `softplus`, `SiLU`, `RMSNorm` — was
   explicitly out of scope and is untouched.

---

## 9. Provenance

Upstream reference: `github.com/state-spaces/mamba`, commit `e9594ce1` (2026-07-23),
trimmed read-only copy in `third_party/mamba/`. Note CRC has `mamba_ssm` **2.2.2**
installed, which is older and has the `self.dconv` bug mentioned above.

`PROGRESS.md` is the full chronological log — what was implemented, what was
learned, what stayed uncertain, per milestone, including the bugs and false starts.
