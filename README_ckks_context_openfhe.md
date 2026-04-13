# OpenFHE CKKS Context: Quick Guide

This note explains what a CKKS context means in OpenFHE and how to increase multiplicative depth.

## 1) What a CKKS context is

A CKKS context is the encryption "compute environment" that defines:
- security level
- precision/scale behavior
- ciphertext size/performance
- how many multiplication layers your circuit can support

In OpenFHE, this is configured through `CCParams<CryptoContextCKKSRNS>`.

## 2) Important parameters

- `SetMultiplicativeDepth(d)`
  - Main depth budget knob.
  - Roughly: max multiplication *layers* (not raw multiply count).

- `SetScalingModSize(s)`
  - Bit-size of scaling primes used during CKKS rescaling.
  - Common values: 40–50 bits.

- `SetFirstModSize(f)`
  - Size of first prime; often a bit larger (e.g., 60).

- `SetRingDim(N)` or security-level-based selection
  - Controls polynomial ring dimension and performance/security tradeoff.

- `SetBatchSize(b)`
  - Number of packed slots used for SIMD-style vector operations.

## 3) What "multiplicative depth" means

Depth is about the longest chain of dependent multiplications.

Example:
- `(a*b) + (c*d)` has depth 1
- `((a*b)*c)` has depth 2

Additions/rotations do not usually consume multiplicative depth the same way multiplication+rescale does, but they still add runtime/noise.

## 4) How to increase depth

Primary method: increase `SetMultiplicativeDepth`.

Example:

```cpp
CCParams<CryptoContextCKKSRNS> params;
params.SetMultiplicativeDepth(5);   // increase from 3 -> 5
params.SetScalingModSize(40);
params.SetFirstModSize(60);
params.SetRingDim(8192);
```

Then regenerate context and keys (you cannot reuse old keys with new params).

## 5) Practical tradeoff

Higher depth gives more circuit capacity, but costs:
- larger ciphertexts
- slower ops
- higher memory use

So increase depth gradually (e.g., 3 -> 4 -> 5) and benchmark each step.

## 6) Rule of thumb for this repo

If you see level/depth-type errors (e.g., level exhaustion / DropLastElement):
- either reduce encrypted stages (hybrid boundary),
- or increase multiplicative depth and retune parameters.

For current FHE-S4 experiments, a stable strategy is:
- keep Toeplitz + skip encrypted,
- decrypt before deep nonlinear/output stack (GLU/head),
- then optionally re-encrypt final output.
