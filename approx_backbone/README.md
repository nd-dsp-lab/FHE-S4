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
