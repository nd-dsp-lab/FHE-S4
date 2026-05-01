# OpenFHE S4 Stage Debug Fix

## What Was Broken

The full OpenFHE debug run was producing a large final error even though the Toeplitz + skip stage matched plaintext nearly exactly:

```text
[phase1] skip_max_abs_diff ~= 1e-15
[phase2] post-GELU error ~= 3.024159
[phase3] Global Error ~= 0.835842
```

This made it look like the activation approximation or GLU clamp path was introducing a large error.

## Root Cause

The debug comparisons were not stage-aligned, and the first real mismatch was in the GELU Chebyshev evaluation.

The repeated `3.024159` error was half of the exported GELU constant coefficient:

```text
6.048316 / 2 = 3.024158
```

OpenFHE's `EvalChebyshevSeries` uses the conventional Chebyshev-series form with the constant term interpreted as `c0 / 2`. The PyTorch exporter stores coefficients for:

```text
c0*T0 + c1*T1 + ...
```

So passing the exported coefficients directly to OpenFHE shifted the GELU output by `-c0/2`.

## What Changed

`export_for_openfhe_new.py` now exports explicit references for every stage:

- `post_gelu`
- `pre_gate`
- `gate`
- `gated`
- `pooled`

`openfhe_s4.cpp` now:

- compares decrypted values against the matching stage reference,
- prints per-stage `max_abs_diff` values,
- includes a `phase3_ref` path that re-encrypts exported `gated` and validates mean + decoder independently,
- doubles the Chebyshev constant coefficient before calling `EvalChebyshevSeries`.

## Validation

Run:

```bash
python export_for_openfhe_new.py --seq_len 32 --d_model 4 --toeplitz_K 16 --out forward_pass_data.json
cmake --build build_nd --target openfhe_s4 -j
./build_nd/openfhe_s4 forward_pass_data.json
```

Expected key output after the fix:

```text
[phase2] overall_post_gelu_max_abs_diff=5.241206e-07
[conv] overall_pre_gate_max_abs_diff=4.177383e-07
[glu] overall_gated_max_abs_diff=2.116435e-07
[phase3_ref] Global Error: 1.032426e-08
[phase3] Global Error: 1.161045e-08
```

Conclusion: on the toy config, Toeplitz, GELU, Conv1d, GLU, mean reduction, and decoder are stage-validated. The previous large error was a Chebyshev coefficient convention bug, not GLU clamp error or accumulated FHE noise.
