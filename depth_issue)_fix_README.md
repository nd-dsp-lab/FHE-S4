# depth_issue)_fix

## What changed

This branch adds targeted debugging instrumentation and a hybrid GLU clamp path in `openfhe_s4.cpp` to diagnose CKKS decode failures and stabilize the gate range.

### 1) Runtime parameter banner
- Added startup logging for:
  - `multDepth`
  - `scaleModSize`
  - `firstModSize`
  - `ringDim`
  - `batchSize`
- Purpose: confirm the binary is using the expected CKKS parameters at runtime.

### 2) Stage-level decrypt probes
- Added `probeDecrypt(tag, ciphertext, outLen)` helper.
- It decrypts selected intermediate ciphertexts and prints:
  - vector length
  - min/max
  - first two values
- It catches decryption exceptions, prints `tag`, and rethrows.
- Probe points currently include:
  - `phase2/act[c=*]`
  - `conv/pre_gate[i=*]`
  - `glu/gate[c=*]`
  - `glu/gated[c=*]`
  - `phase3/mean[c=*]`
- Purpose: isolate the first failing stage when approximation error is too high.

### 3) Hybrid GLU clamp (decrypt -> clamp -> re-encrypt)
- In the GLU block, after computing `gate = 0.25 * b + 0.5`:
  1. decrypt `gate`
  2. clamp each slot to `[0.0, 1.0]` in plaintext
  3. re-encrypt the clamped gate
  4. continue with `ctxt_gated = a * gate`
- Purpose: force gate range control while debugging and avoid runaway gate values.

## Validation run status

Validated with:
- binary: `build_nd/openfhe_s4`
- input: `forward_pass_data.json`

Observed:
- build succeeds
- run completes (no OpenFHE decryption exception on this sample)
- clamp is visible in probe output (`glu/gate` max capped at `1.0` where applicable)

## Important caveat

The GLU clamp path now crosses a trust boundary (decrypt/re-encrypt). This is not end-to-end encrypted execution for that segment. It is intended for debugging/hybrid experiments, not strict fully-homomorphic deployment behavior.

## Suggested next step

If strict FHE is required, replace this hybrid clamp with an FHE-safe approximation (e.g., polynomial/surrogate clamp) and keep probe hooks for staged verification.
