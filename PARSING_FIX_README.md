# Parsing Fix Notes (`openfhe_s4.cpp`)

## Problem observed

The C++ pipeline crashed during/after Phase 2.5 with either:

- `conv_weight row mismatch: got 4, expected 8`
- segmentation fault when shape checks were bypassed
- occasional `stod` parsing failures in the line-based parser

Root cause was a mismatch between exported JSON layout and what `openfhe_s4.cpp` expected, combined with a fragile manual parser.

---

## What was changed

### 1) Fixed JSON export shape in `export_for_openfhe_new.py`

`output_linear[0]` is `Conv1d(d_model, 2*d_model, kernel_size=1)`, so:

- Weight shape is `(2*d_model, d_model, 1)` and must be exported as a 2D matrix `(2*d_model, d_model)`.
- Bias shape is `(2*d_model,)` and must be exported as a vector.

Updated export:

- `conv_weight = conv_layer.weight[:, :, 0]` (as nested list)
- `conv_bias = conv_layer.bias[:]` (as list)

This resolves the immediate dimension mismatch in C++.

### 2) Replaced manual line parser with `nlohmann::json` in `openfhe_s4.cpp`

The old parser depended on line formatting and bracket-state flags. It could fail when arrays changed formatting or when non-numeric lines were fed into `stod`.

Now `ParseTestData(...)`:

- parses the file as JSON object
- extracts all fields with typed access (`get<int>`, `get<double>`, `get<vector<...>>()`)
- validates `channels` is an array and converts each channel explicitly
- emits clear errors for:
  - invalid JSON syntax
  - schema mismatches / missing keys / wrong types

Backward compatibility:

- If `conv_bias` is still a scalar in an older JSON file, parser accepts it and stores a length-1 vector so failure mode is explicit in later shape checks.

### 3) Added dependency wiring in `CMakeLists.txt`

Added `FetchContent` for `nlohmann_json` and linked target:

- `nlohmann_json::nlohmann_json` (for `openfhe_s4`)

---

## Safety guards added

Before heavy FHE compute, `openfhe_s4.cpp` checks:

- number of channels equals `d_model`
- `conv_weight` row count equals `2*d_model`
- each `conv_weight[i]` has width `d_model`
- `conv_bias` length equals `2*d_model`

If any fail, program exits with clear message instead of undefined behavior.

---

## Verification run

Build and run after changes:

1. Configure/build `openfhe_s4`
2. Re-export `forward_pass_data.json`
3. Execute `openfhe_s4`

Observed:

- No parse crashes
- No `conv_weight` shape mismatch
- No segfault in conv loop
- Pipeline proceeds through `[phase1]`, `[phase2]`, `[conv]`, `[glu]`

Current remaining issue is unrelated to JSON parsing:

- CKKS decode error at final stage (`approximation error too high`)

That is a parameter/noise-budget tuning task, not a parser correctness issue.

