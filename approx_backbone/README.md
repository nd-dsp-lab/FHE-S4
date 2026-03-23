# Approximate S4D Backbone (FHE Handoff)

This folder is the handoff package for implementing OpenFHE inference on the
adding-problem S4D backbone.

## 1) Recommended backbone

Use:
- `gelu_mode=poly6`
- `glu_mode=linear_gate`

Best reported result:
- `test_mse=6.4206e-05`
- `test_acc=99.9%`
- `pre_gate_range=[-249.43, 248.67]`

## 2) What is in this folder

- `model.py`: approximate model + `forward_features()` stage tensors
- `eval_backbone.py`: quick accuracy/range check
- `checkpoints/`: packaged checkpoints
- `__init__.py`: package exports

Quick check:

```bash
python eval_approx_backbone.py
```

or

```bash
python -m approx_backbone.eval_backbone
```

## 3) What was approximated

### GELU
- Original `nn.GELU()` replaced by selectable approximation.
- Recommended: `poly6` (best accuracy + HE-friendly polynomial).

### GLU gate
Split is unchanged:

```python
a, b = torch.chunk(pre_gate, 2, dim=channel_dim)
```

Original GLU gate:

```python
gate = sigmoid(b)
out = a * gate
```

Approx gate used here:

```python
gate = clamp(0.25 * b + 0.5, 0, 1)
out = a * gate
```

## 4) Why this matters for HE

- `0.25*b + 0.5`: easy in CKKS (ct-pt linear ops).
- `a * gate`: one ct-ct multiply.
- `clamp`: hard in CKKS (piecewise/if-else-like operation).

So this gate is typically easier than sigmoid GLU in HE, but exact clamp still
needs a strategy.

## 5) Stage mapping for OpenFHE

Use `model.s4d.forward_features()` outputs as stage references:
- `kernel` -> Toeplitz coefficients
- `conv` -> Toeplitz convolution result
- `skip` -> add `D` skip term
- `post_gelu` -> GELU output
- `pre_gate` -> output of `Conv1d(d_model, 2*d_model, 1)`
- `gated` -> post-gate tensor
- `pooled` -> sequence mean

Then apply decoder (`weight`, `bias`) for final scalar.

## 6) Integration paths for the gate

### Path A (recommended first): Hybrid gate
1. Run HE up to `pre_gate`.
2. Decrypt.
3. Plaintext gate: `clamp(0.25*b + 0.5, 0, 1)`, then `a*gate`.
4. Re-encrypt if needed.

Pros: easiest and exact to training-time gate.
Cons: not fully end-to-end HE.

### Path B: Full-HE gate with polynomial surrogate
1. Compute `g_lin = 0.25*b + 0.5` in HE.
2. Approximate clamp with polynomial `g_hat = p(g_lin)`.
3. Compute `a * g_hat` in HE.

Pros: fully encrypted gate.
Cons: approximation error + more depth; may require retraining with same gate polynomial.

### Path C: Sigmoid-style GLU in HE
Approximate sigmoid and use `a * sigmoid_approx(b)`.

Usually harder/more expensive than Path B for CKKS.

## 7) Concrete execution plan

### Phase 0: Toeplitz sanity

```bash
python export_for_openfhe.py --seq_len 32 --d_model 4 --toeplitz_K 16 --out toeplitz_test_data.json
cmake -S . -B build -DOpenFHE_DIR=/usr/local/lib/OpenFHE
cmake --build build -j
./build/openfhe_rotate_sum_protocol toeplitz_test_data.json
./build/openfhe_dense_toeplitz_baseline toeplitz_test_data.json
```

Proceed only if errors are small (toy run around `1e-6` or better).

### Phase 1: Export full-backbone reference

Create `export_approx_backbone_for_openfhe.py` and export one deterministic sample:
- metadata: `seq_len`, `d_model`, `seed`
- per-channel input vectors
- `kernel`, `D`
- refs: `conv`, `skip`, `post_gelu`, `pre_gate`, `gated`, `pooled`
- decoder params + final output

Save: `approx_backbone_test_data.json`.

### Phase 2: C++ pipeline executable

Create `openfhe_approx_backbone_pipeline.cpp` with:
- `--stage conv|skip|post_gelu|pre_gate|gated|final`
- one ciphertext per channel
- timing print: context/encrypt/op/decrypt
- max-abs-diff check vs exported reference

### Phase 3: Implement stages in order

1. `conv` (structured Toeplitz rotate-sum)
2. `skip`
3. `post_gelu` (poly6)
4. `pre_gate` (1x1 linear map)
5. gate path (A or B)
6. `pooled` + decoder

Do not move forward unless current stage matches reference.

### Phase 4: End-to-end milestones

- **Milestone M1 (hybrid end-to-end):** Path A gate.
- **Milestone M2 (full HE):** Path B gate.

## 8) Minimal command checklist

```bash
python eval_approx_backbone.py
python export_for_openfhe.py --seq_len 32 --d_model 4 --toeplitz_K 16 --out toeplitz_test_data.json
cmake -S . -B build -DOpenFHE_DIR=/usr/local/lib/OpenFHE
cmake --build build -j
./build/openfhe_rotate_sum_protocol toeplitz_test_data.json
./build/openfhe_dense_toeplitz_baseline toeplitz_test_data.json
python export_approx_backbone_for_openfhe.py --seq_len 32 --d_model 4 --out approx_backbone_test_data.json
./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage conv
./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage skip
./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage post_gelu
./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage pre_gate
./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage final --mode hybrid_gate
# optional full-HE gate:
./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage final --mode full_he_gate
```

## 9) Bottom line

This backbone is the current best handoff target:
- strong accuracy in plaintext (`poly6 GELU + linear_gate GLU`)
- clear stage boundaries for HE insertion
- existing Toeplitz scripts already provide the export/verify template
# Approximate S4D Backbone Handoff

This directory is the handoff package for collaborators who will add the FHE/OpenFHE part later.

The goal is to provide:
- a **working approximate model backbone**
- a **minimal test script** that prints test accuracy/MSE
- a clear explanation of **what is approximated** and **where OpenFHE/C++ logic should be inserted**

Everything here is plain PyTorch/Numpy. No TenSEAL is required.

## Recommended Backbone

Use:
- `gelu_mode = poly6`
- `glu_mode = linear_gate`

This is the best-performing approximate backbone found so far.

From `gelu_retrain_results.json`:
- `poly6 + linear_gate`
  - `test_mse = 6.4206e-05`
  - `test_acc = 99.9%`
  - `pre_gate_range = [-249.43, 248.67]`

## Files In This Directory

- `model.py`
  - standalone approximate adding-problem model
  - contains the approximate S4D block and helper dataset/eval utilities
- `eval_backbone.py`
  - tiny script to verify the checkpoint still gives the expected test accuracy
- `__init__.py`
  - package exports
- `checkpoints/`
  - packaged approximate backbone checkpoints for sharing/push

## Checkpoint Bundle

The current checkpoint bundle is stored in:

- `approx_backbone/checkpoints/`

Included checkpoint files:

- `s4d_best_exact.pt`
- `s4d_best_linear_gate.pt`
- `s4d_best_poly4_gate.pt`
- `s4d_best_poly6_gate.pt`
- `s4d_best_gelu_exact_glu_linear_gate.pt`
- `s4d_best_gelu_poly4_glu_linear_gate.pt`
- `s4d_best_gelu_poly6_glu_linear_gate.pt`

## What Is Approximated

### GELU

The original `nn.GELU()` is replaced by a selectable approximation.

Recommended mode:
- `poly6`

Why:
- it retrained successfully with near-baseline accuracy
- it is polynomial, which is much more natural for CKKS evaluation

### GLU

The original `Conv1d(d_model, 2*d_model, 1) + GLU(dim=-2)` structure is preserved, but the gate is selectable.

Recommended mode:
- `linear_gate`

This computes:

```python
a, b = torch.chunk(pre_gate, 2, dim=channel_dim)
gate = clamp(0.25 * b + 0.5, 0, 1)
output = a * gate
```

Why:
- retrained `linear_gate` performed very well
- retrained polynomial gate variants were much worse

### GLU approximation: what exactly changed

The channel split behavior is unchanged:

```python
a, b = torch.chunk(pre_gate, 2, dim=channel_dim)
```

What changed is the gate function applied to `b`.

Original GLU gate:

```python
gate = sigmoid(b)
out = a * gate
```

Approximate linear gate used here:

```python
gate = clamp(0.25 * b + 0.5, 0, 1)
out = a * gate
```

So this is still a multiplicative gate (`a * gate`), but the gate itself is an
affine transform plus clamping instead of sigmoid.

### What this means for FHE implementation

In CKKS/OpenFHE:

- `0.25 * b + 0.5` is easy and cheap (ciphertext-plaintext arithmetic).
- `a * gate` is one ciphertext-ciphertext multiply (depth increase).
- `clamp(., 0, 1)` is the difficult part because it is piecewise and behaves
  like an if/else comparison, which CKKS does not natively support in leveled mode.

Practical implication:
- this gate is still usually easier than full sigmoid GLU in HE,
- but exact clamp is not "free" in HE.

### FHE integration paths for the gate

Use one of the following paths explicitly:

#### Path A: Hybrid gate (fastest path to end-to-end)

1. Run HE pipeline up to `pre_gate`.
2. Decrypt `pre_gate`.
3. In plaintext: split `(a, b)`, compute exact `clamp(0.25*b + 0.5, 0, 1)`, then `a * gate`.
4. Re-encrypt if later stages must continue in HE.

Pros:
- easiest to implement
- exact match to trained gate
- best first milestone for end-to-end correctness

Cons:
- not fully homomorphic end-to-end (contains decrypt/re-encrypt boundary)

#### Path B: Full-HE gate via polynomial saturation

1. Compute `g_lin = 0.25*b + 0.5` in HE.
2. Replace clamp with polynomial surrogate: `g_hat = p(g_lin)` where `p` approximates clip on target range.
3. Compute `out = a * g_hat` in HE.

Pros:
- fully encrypted gate path

Cons:
- approximation error from replacing clamp
- additional depth/noise budget usage
- often requires retraining with the same polynomial gate for stability

#### Path C: Keep original sigmoid GLU in HE (not recommended first)

1. Approximate sigmoid with polynomial over a wide pre-gate range.
2. Use `out = a * sigmoid_approx(b)` in HE.

Pros:
- closer to original architecture

Cons:
- usually harder/more expensive than Path B for CKKS
- higher risk of instability over large activation ranges

### Recommended order for collaborators

1. Implement Path A first (hybrid gate) and validate full pipeline.
2. Then implement Path B (full-HE polynomial gate).
3. Only attempt Path C if there is a specific reason to preserve sigmoid behavior.

## Important Caveat

This is an **approximation-friendly backbone**, but it is not automatically a full CKKS-ready nonlinear stack.

- `poly6 GELU`: good FHE candidate
- `linear_gate GLU`: strong empirically, but `clamp` is not directly CKKS-friendly

So this package should be understood as:
- a **high-accuracy approximate backbone**
- not yet a finished end-to-end HE implementation

## Quick Accuracy Check

From the repo root:

```bash
python eval_approx_backbone.py
```

Or use module execution:

```bash
python -m approx_backbone.eval_backbone
```

Or specify a checkpoint manually:

```bash
python eval_approx_backbone.py --checkpoint s4d_best_gelu_poly6_glu_linear_gate.pt
```

Or point directly to the packaged checkpoint directory:

```bash
python eval_approx_backbone.py --checkpoint approx_backbone/checkpoints/s4d_best_gelu_poly6_glu_linear_gate.pt
```

Expected behavior:
- prints checkpoint path
- prints test MSE
- prints test accuracy
- prints pre-gate range

## Model API

### Load the model

```python
from approx_backbone import load_approx_backbone

model = load_approx_backbone()
```

### Full adding-problem forward

```python
out = model(x)
```

where `x` has shape:

```python
(batch, seq_len, 2)
```

### Get intermediate tensors

```python
with torch.no_grad():
    feats = model.s4d.forward_features(u)
```

where `u` has shape:

```python
(batch, d_model, seq_len)
```

Returned tensors include:
- `kernel`
- `conv`
- `skip`
- `post_gelu`
- `pre_gate`
- `gated`
- `pooled`

## OpenFHE / C++ Integration Guidance

This is the most important section for the FHE team.

### Stage mapping

The natural implementation map is:

1. `kernel`
   - export the per-channel S4D kernel coefficients
   - these define the causal Toeplitz convolution

2. `conv`
   - replace with OpenFHE Toeplitz evaluation
   - recommended implementation: rotate-sum / structured Toeplitz
   - baseline implementation: dense Toeplitz linear transform

3. `skip`
   - keep as ciphertext + plaintext vector multiply/add
   - no need for a new nonlinear here

4. `post_gelu`
   - replace `poly6 GELU` with ciphertext polynomial evaluation
   - this is the cleanest nonlinear candidate currently available

5. `pre_gate`
   - this is the tensor after the `Conv1d(d_model, 2*d_model, 1)` and before the GLU-style split
   - split it into:
     - `a` = first half of channels
     - `b` = second half of channels

6. `gated`
   - current plaintext backbone uses `a * clamp(0.25*b + 0.5, 0, 1)`
   - in OpenFHE, this exact clamp is not directly cheap in CKKS
   - collaborators will need to decide whether to:
     - approximate the clamp polynomially,
     - replace it with another HE-safe gate,
     - or keep this stage hybrid/plaintext

7. `pooled`
   - mean reduction can be done with ciphertext slot-sum / EvalSum style logic

8. `decoder`
   - final linear layer is straightforward ciphertext-plaintext linear algebra

### Suggested C++ execution plan

For a first OpenFHE prototype:

1. encode/encrypt input
2. run Toeplitz convolution in HE
3. apply skip in HE
4. apply `poly6 GELU` in HE
5. stop here and compare with plaintext `post_gelu`

For a second milestone:

1. keep the above
2. add the `1x1 Conv1d` in HE
3. stop at `pre_gate`

For the final nonlinear decision:

1. test whether the linear gate should stay hybrid/plaintext
2. or replace it with a polynomial/saturating surrogate that retrains well enough

### Reuse the Toeplitz export-and-verify pattern for the full backbone

The existing Toeplitz workflow already gives a clean template for how to move
PyTorch tensors into OpenFHE C++ and verify correctness stage by stage.

Reference scripts:
- `export_for_openfhe.py`
- `openfhe_rotate_sum_protocol.cpp`
- `openfhe_dense_toeplitz_baseline.cpp`
- `run_inference_fhe_shell_new.py`

What that Toeplitz template currently does:
1. export deterministic per-channel inputs + kernel coefficients to JSON
2. run a structured OpenFHE operator (rotate-sum Toeplitz)
3. run a dense OpenFHE baseline for comparison
4. decrypt and compare each channel against a plaintext reference
5. print timing slices: context / encrypt / op / decrypt / reference

Use this exact pattern for the whole approximate backbone:
1. **Export once from PyTorch**: save all constants and reference tensors for one
   deterministic sample using `forward_features()`.
2. **Implement one C++ stage at a time**: each stage takes encrypted input plus
   plaintext constants and returns encrypted output.
3. **Verify after each stage**: decrypt and compare to exported plaintext target.
4. **Only then compose stages** into a longer encrypted pipeline.

### Minimal export contract for a full-backbone JSON

Create a new exporter (for example `export_approx_backbone_for_openfhe.py`) with
the same style as `export_for_openfhe.py`, but include all stages:

- metadata:
  - `seq_len`, `d_model`, `d_state`, `seed`
- input:
  - per-channel `x` (shape `(L,)` for each channel)
- S4D linear constants:
  - per-channel Toeplitz coefficients (`kernel`)
  - per-channel skip scalar `D[c]`
- references from `forward_features()`:
  - `conv`
  - `skip`
  - `post_gelu`
  - `pre_gate`
  - `gated`
  - `pooled`
- output head:
  - `decoder_weight`, `decoder_bias`
  - final expected scalar output

Keep one deterministic example first (`batch_size=1`) so C++ debugging is
simple and reproducible.

### C++ stage plan that mirrors the Python backbone

Follow this progression (do not jump directly to full forward):

1. **Stage A: Toeplitz only**
   - reuse rotate-sum implementation style from `openfhe_rotate_sum_protocol.cpp`
   - keep one ciphertext per channel
   - compare to exported `conv`
2. **Stage B: Toeplitz + skip**
   - add `D[c] * x_c` in HE
   - compare to exported `skip`
3. **Stage C: + poly6 GELU**
   - evaluate GELU polynomial in HE
   - compare to exported `post_gelu`
4. **Stage D: + 1x1 Conv head**
   - implement channel mixing linear map in HE
   - compare to exported `pre_gate`
5. **Stage E: gate decision**
   - either keep gate hybrid (decrypt before gate),
   - or replace clamp with an HE-safe approximation
   - compare to exported `gated`
6. **Stage F: pooling + decoder**
   - implement reduction and final linear map
   - compare to exported final output

### Why this is the safest path

- It matches what already worked for Toeplitz.
- It isolates errors to a single stage.
- It gives timing transparency for each operation.
- It lets collaborators swap implementations (structured vs dense) while keeping
  the same exported test vectors and acceptance checks.

## Concrete Execution Plan (HE Pipeline + End-to-End)

This section is a direct implementation checklist. Follow in order.

### Phase 0: Environment sanity

1. Build and run the existing Toeplitz references first.
2. Confirm you can run:
   - `python export_for_openfhe.py --seq_len 32 --d_model 4 --toeplitz_K 16 --out toeplitz_test_data.json`
   - `cmake -S . -B build -DOpenFHE_DIR=/usr/local/lib/OpenFHE`
   - `cmake --build build -j`
   - `./build/openfhe_rotate_sum_protocol toeplitz_test_data.json`
3. Proceed only if max error is small (about `1e-6` or better for this toy size).

### Phase 1: Create a full-backbone export artifact

Goal: produce one deterministic JSON file containing all constants + stage references.

1. Create `export_approx_backbone_for_openfhe.py`.
2. Load model from `approx_backbone/model.py` with recommended modes:
   - `gelu_mode=poly6`
   - `glu_mode=linear_gate`
3. Set fixed seed and generate one sample (`batch_size=1`).
4. Run `model.s4d.forward_features(u)` and export:
   - metadata: `seq_len`, `d_model`, `seed`
   - per-channel input vectors
   - Toeplitz coeffs (`kernel`)
   - skip params (`D`)
   - stage references: `conv`, `skip`, `post_gelu`, `pre_gate`, `gated`, `pooled`
   - decoder params and final scalar output
5. Save as `approx_backbone_test_data.json`.

Acceptance check:
- JSON loads in C++ and has all keys above.

### Phase 2: Implement a single C++ stage runner

Goal: one executable that can run one stage at a time and compare to reference.

1. Create `openfhe_approx_backbone_pipeline.cpp`.
2. Add CLI flag:
   - `--stage conv|skip|post_gelu|pre_gate|gated|final`
3. Keep one ciphertext per channel.
4. Print timing breakdown:
   - context creation
   - encryption
   - homomorphic op
   - decryption
   - max abs error vs exported reference
5. Reuse rotation-key pattern from `openfhe_rotate_sum_protocol.cpp`.

Acceptance checks by stage:
- `conv`: matches exported `conv`
- `skip`: matches exported `skip`
- `post_gelu`: matches exported `post_gelu`
- `pre_gate`: matches exported `pre_gate`

Do not move to next stage until current one matches.

### Phase 3: HE linear backbone (no gate yet)

Goal: complete all linear + polynomial-friendly stages in HE.

Implement in this order:
1. **Toeplitz conv** (structured rotate-sum)
2. **Skip add** (`+ D[c] * x_c`)
3. **poly6 GELU**
4. **1x1 Conv head** (channel mixing linear transform)

Run command pattern:
- `./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage pre_gate`

Acceptance check:
- `pre_gate` decrypted output matches exported `pre_gate` within a small tolerance.

### Phase 4: End-to-end path A (hybrid gate, fastest to working)

Goal: deliver end-to-end inference quickly while keeping most heavy work in HE.

1. Run HE until `pre_gate`.
2. Decrypt `pre_gate`.
3. In plaintext apply:
   - split into `(a, b)`
   - `gate = clamp(0.25*b + 0.5, 0, 1)`
   - `gated = a * gate`
4. Re-encrypt if needed for pooling/decoder in HE, or finish in plaintext.

Acceptance checks:
- compare `gated` vs exported `gated`
- compare final scalar vs exported final output

This is the recommended first full pipeline milestone.

### Phase 5: End-to-end path B (full HE gate)

Goal: remove hybrid step and keep all operations encrypted.

1. Replace clamp gate with an HE-safe polynomial/saturating surrogate.
2. Evaluate surrogate in ciphertext on `b`.
3. Multiply ciphertexts: `a * gate_approx`.
4. Re-validate against exported references and final output.

Acceptance checks:
- per-stage error remains controlled
- final prediction error vs plaintext baseline is acceptable on a small evaluation set

Note:
- This phase is harder than all prior phases and may require retraining with the
  exact gate approximation used in HE.

### Phase 6: Scale-up and benchmark protocol variants

Goal: report reproducible performance and correctness.

1. Sweep sequence lengths (`L=32, 64, 128`) at fixed small `d_model`.
2. Measure structured vs dense Toeplitz:
   - structured rotate-sum (`openfhe_rotate_sum_protocol.cpp`)
   - dense baseline (`openfhe_dense_toeplitz_baseline.cpp`)
3. Record for each run:
   - context time
   - encrypt time
   - op time
   - decrypt time
   - max error
4. Keep the same CKKS parameters across comparisons.

### Concrete command checklist

Use this exact order:

1. `python eval_approx_backbone.py`
2. `python export_for_openfhe.py --seq_len 32 --d_model 4 --toeplitz_K 16 --out toeplitz_test_data.json`
3. `cmake -S . -B build -DOpenFHE_DIR=/usr/local/lib/OpenFHE`
4. `cmake --build build -j`
5. `./build/openfhe_rotate_sum_protocol toeplitz_test_data.json`
6. `./build/openfhe_dense_toeplitz_baseline toeplitz_test_data.json`
7. `python export_approx_backbone_for_openfhe.py --seq_len 32 --d_model 4 --out approx_backbone_test_data.json`
8. `./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage conv`
9. `./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage skip`
10. `./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage post_gelu`
11. `./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage pre_gate`
12. `./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage final --mode hybrid_gate`
13. (optional full HE) `./build/openfhe_approx_backbone_pipeline --input approx_backbone_test_data.json --stage final --mode full_he_gate`

If step 12 works and step 13 does not, you still have a valid end-to-end hybrid
pipeline milestone and can continue improving the full-HE gate separately.

### Why this decomposition helps

The FHE team does not need to reconstruct the model graph from scratch.

They can line up the HE implementation with named intermediate tensors:
- `conv`
- `skip`
- `post_gelu`
- `pre_gate`
- `gated`
- `pooled`

That makes plaintext-vs-HE debugging much easier.

## Why This Backbone Was Chosen

Empirically:
- polynomial GLU gate replacements were not robust
- linear gate retrained extremely well
- `poly6 GELU + linear_gate GLU` gave the best approximate backbone result

So this is the cleanest current handoff point for adding FHE logic around a model that still works.
