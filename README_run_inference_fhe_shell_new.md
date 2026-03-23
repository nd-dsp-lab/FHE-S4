# `run_inference_fhe_shell_new.py` Student Guide

This script is an **inference sanity-check harness** for incremental FHE integration with MiniS4D.

It is **not** a training script, and it is **not** full end-to-end encrypted MiniS4D yet.

## What This Script Is For

Use this script to:
- verify plaintext inference works
- verify TenSEAL setup works (encrypt -> tiny op -> decrypt)
- validate a Toeplitz/conv milestone in FHE
- inspect timings at each stage

Do **not** use this script for:
- training
- benchmarking production performance
- claiming full encrypted MiniS4D forward

## Core Modes (Milestones)

The script supports:
- `plain`: plaintext MiniS4D forward baseline
- `fhe_stub`: minimal encrypted pipeline (bias add)
- `fhe_toeplitz`: Toeplitz/conv milestone in FHE, per channel
- `fhe_full`: hybrid milestone (Toeplitz in FHE, rest in plaintext)
- `both`: `plain + fhe_stub` (default)


## Important Guardrails

- Inference only: batch size is fixed to 1 and CPU only.
- Context creation happens once in `main()`, never inside forward or loops.
- Keep dimensions small while debugging (`seq_len=32`, `d_model=4`).
- In Toeplitz modes, Galois keys are enabled because rotations are needed.

## What It Does *Not* Do (Yet)

- It does not implement explicit `rotate()` calls for CKKS vectors (TenSEAL API constraints in many setups).
- It does not run all MiniS4D layers fully in FHE (e.g., GLU / full output stack).
- `fhe_full` is hybrid by design: it decrypts after Toeplitz and finishes the rest in plaintext.

That is the correct intermediate milestone for this project.

## Recommended Run Order

Run in this order and do not skip steps:

1. Plain baseline

```bash
python run_inference_fhe_shell_new.py --mode plain --seq_len 32 --d_model 4
```

2. FHE stub (should run quickly)

```bash
python run_inference_fhe_shell_new.py --mode fhe_stub --seq_len 32 --d_model 4
```

3. FHE Toeplitz milestone (main target)

```bash
python run_inference_fhe_shell_new.py --mode fhe_toeplitz --seq_len 32 --d_model 4 --toeplitz_K 16
```

4. Hybrid forward (optional, after Toeplitz correctness)

```bash
python run_inference_fhe_shell_new.py --mode fhe_full --seq_len 32 --d_model 4 --toeplitz_K 16
```

## How to Read the Code (Section Map)

The script is staged with explicit comments:
- **Stage 1**: plaintext baseline
- **Stage 2**: one-time TenSEAL context setup
- **Stage 3a**: FHE stub pipeline
- **Stage 3b**: per-channel Toeplitz in FHE + plaintext reference checks
- **Stage 3c**: optional hybrid continuation (`fhe_full`)

Edit one stage at a time. Do not refactor everything at once.

## Where Students Should Modify Code

Start changes in:
- `extract_channel_kernel_coeffs(...)` if coefficient extraction mismatches a model variant
- Toeplitz implementation only **after** Toeplitz correctness is validated

Do **not** change early:
- per-channel packing strategy
- context creation placement
- default small dimensions

## Correctness Expectations

For `fhe_toeplitz`, check:
- per-channel max absolute diff is small
- overall max absolute diff is small
- timings are printed for encryption, Toeplitz op, decryption, and plaintext reference

Focus on correctness and clear timing visibility first, not speed.

## Common Failure Patterns

- `fhe_stub` fails or is very slow:
  - likely TenSEAL install or context parameter issue
- `fhe_toeplitz` mismatch is large:
  - likely kernel coefficient extraction or Toeplitz convention mismatch
- dimension mismatch errors:
  - verify `seq_len`, `d_model`, and per-channel coefficient shapes

## Suggested Next Milestone (After This Works)

Once `fhe_toeplitz` is correct and stable:
- move from matrix-based Toeplitz to a true rotate-sum implementation (if your TenSEAL version exposes required APIs)
- keep this as a second-phase optimization, not the first deliverable

---

