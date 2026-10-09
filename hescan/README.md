# HEScan on OpenFHE

This directory re-creates **HEScan**, the factorized encrypted scan kernel from
[FESC: Remodeling Long-Context Private Inference with Encrypted State-Space Models](https://arxiv.org/abs/2608.17442),
on OpenFHE CKKS (≥ 1.4).

It covers only the HE side of FESC. In the paper, an MPC "factor builder" produces one
compact packet per token, and the skip and gate paths stay in MPC. Here a random packet
stands in for the MPC output.

## What is computed

FESC Eq. 2–3, for one block of `L` tokens:

```
h_k = br(a_k) ⊙ h_{k-1} + x_k ⊗_g B_k        h_k ∈ R^{E×d_s},  E = H·P
m_k = Σ_{d_s}( h_k ⊙ br(C_k) )               m_k ∈ R^{H×P}
```

Only the packet `(x_k ∈ R^E, a_k ∈ R^H, B_k, C_k ∈ R^{G×d_s})` enters, and only `m_k`
leaves. Expanded state never crosses the boundary.

Each token is an affine map `T_k: h ↦ A_k ⊙ h + U_k` with `A_k = br(a_k)` and
`U_k = br(x_k) ⊙ br(B_k)`. The scan composes these maps (early, then late):

```
A_{late∘early} = A_late ⊙ A_early                 (Eq. 5: decay term)
U_{late∘early} = A_late ⊙ U_early + U_late        (Eq. 6: state term)
```

> **Caveat.** arxiv.org was unreachable from the environment this was written in. Eq. 2–3,
> the packing and the "log-depth, linear-work parallel-summary scan" claim come from the
> paper's abstract and search snippets. The two composition rules above are the standard
> affine-scan operator and are my reconstruction of Eq. 5–6. Check them against the PDF.

## Mapping onto OpenFHE

| HEScan needs | OpenFHE | Where |
|---|---|---|
| ct-ct multiply + relinearize (Eq. 5, Eq. 6) | `EvalMult` | `HEScan::Compose` |
| Rescale / level alignment across scan nodes | `FLEXIBLEAUTO` (default) or `FIXEDAUTO` (`--scaling fixed`) | automatic. In Eq. 6, `U` sits one level below `A` and OpenFHE aligns them. |
| `br` broadcasts | `EvalFastRotationPrecompute` + `EvalFastRotation` (hoisted) | `HEScan::Replicate` |
| `Σ_{d_s}` reduction tree | `EvalRotate` (+1, +2, …, +d_s/2) | `HEScan::ReduceDs` |
| Chain `[60, 40×D, 60]`, scale 2^40 | `SetFirstModSize(60)`, `SetScalingModSize(40)`, `SetMultiplicativeDepth(D)` | `hescan_demo.cpp`. The trailing ~60-bit moduli are OpenFHE's HYBRID key-switching moduli. |
| N = 2^15 / 2^16 | `SetRingDim` (`--logn 15/16`) | 128-bit classic security by default |
| Complex pairing + one conjugation | `SetCKKSDataType(COMPLEX)`, `EvalAutomorphismKeyGen` / `EvalAutomorphism` with index `2N−1` | `HEScan::Unpack`, `conj_check.cpp` |

`EvalSum` is not used for `Σ_{d_s}`. It sums the whole batch, but HEScan needs a
*segmented* sum over each `d_s`-block. That takes log2(d_s) rotate-and-add steps, and
the result lands in the first slot of each block.

### Conjugation: verified first

`hescan_conj_check` tests the risky pieces on 1.4.0 before anything else. All of these pass
with error around 1e-9:

* `EvalAutomorphism(ct, 2N−1)` equals slot-wise complex conjugation.
* The one-conjugation split `u = (z + z̄)/2`, `v = −i/2 · (z − z̄)` recovers both packed chunks.
* Re-packing `u² + i·v²` after real arithmetic works.
* Hoisted `EvalFastRotation` works with negative (right-shift) indices.

There is no dedicated public `Conjugate` call on `CryptoContext`; the automorphism route is
the one to use. Multiplying by the complex constant `−i/2` costs one level, like any scalar
multiply.

## Packing

A state chunk holds `s_state` slots and `c_state = s_state/d_s` channels. Slot `(c, n)`
is `c·d_s + n`, global channel `e = j·c_state + c`, and there are `K_s = ⌈E·d_s/s_state⌉`
chunks.

Packet ciphertexts arrive as *seeds*: zeros everywhere except one representative per
broadcast run. `br` fills each run with hoisted rotations of that one ciphertext. Each
stage does one digit decomposition and up to `radix−1` rotations (`--radix`), for
log_radix(count) stages.

| seed | non-zero at | `br` stride, count |
|---|---|---|
| `x` | `c·d_s` ← `x_e` | 1, `d_s` |
| `a` | `c·d_s` for `c % runA == 0` ← `a_head(e)` | 1, `runA·d_s` |
| `B`, `C` | `c·d_s + n` for `c % runG == 0` ← `B_{g(e),n}` | `d_s`, `runG` |

Here `runA = min(P, c_state)` and `runG = min(E/G, c_state)`. `d_s`, `P`, `s_state` and
`E/G` must be powers of two. Seeds have zeros outside their runs, so `br` uses rotations only
and consumes no levels.

**Complex pairing.** Chunks `2q` and `2q+1` cross the boundary as one ciphertext
`z = u + i·v`, which halves the boundary ciphertexts (4·⌈K_s/2⌉ in and ⌈K_s/2⌉ out per
token). One conjugation per incoming ciphertext splits it. The outputs are re-paired as
`mask⊙m_{2q} + (i·mask)⊙m_{2q+1}`, and that plaintext multiply doubles as the output mask.
The split costs one level. Dropping it (`--complex 0`) saves that level but doubles the
boundary traffic.

## Depth

`Config::Depth()` runs the scan schedule on level counters, so the parameter set always
matches the code. In every run the levels actually consumed equalled the predicted D.

```
D = [1 split] + 1 (x⊙B) + scan + [1 h_0] + 1 (⊙C) + [1 mask/re-pair]
scan:  brent-kung 2⌈log2 L⌉−1  (O(L) work)   hillis-steele ⌈log2 L⌉  (O(L log L))   sequential L−1
```

For L = 16 with complex pairing: BK D = 10, HS D = 8, sequential D = 19.

## Results (4 vCPU, OpenFHE 1.4.0, OpenMP)

Default shape: H=4, P=4, G=2, d_s=8, L=16, s_state=64, so K_s=2 (one complex pair).
Brent–Kung scan, FLEXIBLEAUTO, 128-bit security, random packet with `a ∈ (0.5, 1)`.

| N | D | log2 Q | HEScan time | split+br / scan / contract | max abs error vs Eq. 2–3 |
|---|---|---|---|---|---|
| 2^15 | 10 | 461 | 28.7 s | 18.0 / 5.3 / 2.5 s | 4.0e-8 |
| 2^16 | 10 | 461 | 59.7 s | 42.9 / 11.1 / 5.7 s | 5.3e-8 |

A larger shape at N=2^16 (H=8, P=8, G=2, d_s=16, L=32, s_state=256, so K_s=4) gives
D=12 with 12/12 levels consumed. HEScan takes 327 s (248 / 50 / 29 s), the max error is
2.2e-7, and peak RSS is 10.2 GB. The same run first hit OOM on a 15 GB box, before
`Run` was changed to split and broadcast token by token and free intermediates
immediately.

`run_tests.sh` sweeps the scan algorithms, complex on/off, FIXEDAUTO, `h_0`, radix 4/8,
non-power-of-two L (7, 13), odd K_s, a single group, and chunks narrower than a head. All
cases pass with error ≤ 4e-7. FIXEDAUTO's error is about 10× FLEXIBLEAUTO's, as expected.

## Build and run

```bash
# OpenFHE >= 1.4 (COMPLEX CKKS data type)
git clone --branch v1.4.0 https://github.com/openfheorg/openfhe-development.git
cmake -S openfhe-development -B ofhe-build -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build ofhe-build -j && sudo cmake --install ofhe-build

cd hescan && mkdir -p build && cd build
cmake .. [-DOpenFHE_DIR=<prefix>/lib/OpenFHE] && make -j
./hescan_conj_check
./hescan_demo                        # N=2^15, 128-bit
./hescan_demo --logn 16 --scan hs    # see the header of hescan_demo.cpp for all flags
../run_tests.sh                      # toy-N correctness sweep
```

`hescan/` is a standalone CMake project. The top-level `CMakeLists.txt` hard-codes
`/usr/local/include/openfhe` for older demos, so it is left untouched.

## Not done yet

* **Long sequences / streaming.** One block of L tokens is scanned at a time. `--h0 1`
  carries an encrypted initial state into the block, but every carried block consumes
  levels, so streaming many blocks needs bootstrapping or an MPC refresh of `h`. The
  memory-resident schedule from the paper is not implemented.
* **Wasted decay products.** All decay products are computed, even those only needed when
  `h_0` is present (the last prefix's `A`). Skipping the dead ones would save some
  multiplies.
* **Fixed split constants.** The 1/2 and −i/2 split factors on `x` and `B` could be folded
  into the output mask, saving the split level for the `U` path.
* **No GPU kernels and no MPC side.**
