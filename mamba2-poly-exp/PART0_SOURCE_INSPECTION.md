# Part 0 — What the official Mamba-2 source actually does

Everything below was read out of the real source, not recalled from the paper.

**Source of truth**

```
repo    https://github.com/state-spaces/mamba
commit  e9594ce1c732d97440f0332fdc43170a2294dbfa
date    2026-07-23
```

A trimmed read-only copy lives in [`third_party/mamba/`](third_party/mamba) so the
line numbers quoted here keep working. All quoted line numbers refer to that commit.

---

## 0.1 Where the continuous / discrete transition is constructed

There are three pieces, and they are built in three different places.

**(a) The continuous decay `A` — one scalar per head.**

`third_party/mamba/mamba_ssm/modules/mamba2.py:134-136` stores `A` in log form:

```python
A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
A_log = torch.log(A).to(dtype=dtype)
self.A_log = nn.Parameter(A_log)
```

and `mamba2.py:182` reconstructs it every forward pass:

```python
A = -torch.exp(self.A_log.float())     # (nheads,)
```

So `A` is always **strictly negative**, and `A_init_range=(1, 16)` means
`A ∈ [-16, -1]` at initialisation. This is a *parameter* `exp`, not a
*data-dependent* `exp` — it is computed once per forward on a tensor of 24
numbers, and under FHE it would be plaintext. **It is not our target.**

**(b) The timestep `delta` (called `dt` in the code) — input-dependent.**

`dt` is one of five slices of a single input projection.
`mamba2.py:96`:

```python
d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
```

and `mamba2.py:211-214` splits it as `[z0, x0, z, xBC, dt]`, so the last
`nheads` channels of the projection are the raw `dt`. It then becomes a positive
timestep via a bias and a softplus — `mamba2.py:313`:

```python
dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))   # (batch, nheads)
```

`dt_bias` is initialised (`mamba2.py:118-127`) as the softplus-inverse of a
log-uniform sample in `[dt_min, dt_max] = [0.001, 0.1]`, so `dt` starts *small*.
That initialisation matters for us: it is the first hint that `z = A*dt` starts
near zero, not near `-8`.

**(c) The discretisation itself.** This is where the story splits, see 0.2.

---

## 0.2 Where `z = A * delta` and `a = exp(z)` are computed

There is **no single line** in the production path that computes `a = exp(A*dt)`.
Here is every place the quantity exists, and in what form.

### (i) Sequential form — literally `exp(A*dt)`

`mamba_ssm/modules/mamba2.py:313-314`, inside `Mamba2.step()` (single-token
decoding, used only when the Triton `selective_state_update` kernel is absent):

```python
dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
dA = torch.exp(dt * A)                                  # (batch, nheads)
...
ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
```

This is exactly the user's notation: `z = A*dt`, `a = exp(z)`,
`h_t = a_t·h_{t-1} + b_t`. It is the cleanest statement of the semantics in the
whole repo — but it is **decode-only and single-timestep**.

`mamba_ssm/ops/selective_scan_interface.py:162,175` has the same thing over a
whole sequence (this is the Mamba-**1** reference scan, which Mamba-2 arguments
can be routed into via `ssd_selective_scan`):

```python
deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
...
for i in range(u.shape[2]):
    x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
```

### (ii) Chunked / production form — `exp` of *cumulative sums*, never of one step

`mamba_ssm/ops/triton/ssd_chunk_state.py:84-86` is where `A*dt` is formed:

```python
dA = dt * A[:, None]
dA_cs = tl.cumsum(dA, axis=1)
tl.store(dA_cs_ptrs, dA_cs, ...)
```

The kernel stores `dA_cumsum`, **not** `exp(A*dt)`. Every subsequent `exp` is an
`exp` of a *difference of cumulative sums*, e.g.
`ssd_chunk_state.py:248`, `ssd_chunk_state.py:354`, `ssd_combined.py:153,205`:

```python
scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_k), 0.0)) * dt_k
```

The pure-PyTorch `ssd_minimal.py` shows the same structure in readable form —
`ssd_minimal.py:54,59,67,73`:

```python
L              = torch.exp(segsum(A))                              # A here is already A*dt
decay_states   = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
decay_chunk    = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))
state_decay_out= torch.exp(A_cumsum)
```

Note `ssd_minimal_discrete` is *called* with `A*dt` already multiplied in —
`ssd_minimal.py:103`:

```python
y_min, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, chunk_size)
```

So in the argument list of `ssd_minimal_discrete`, the thing named `A` is our `z`.

### ⚠️ The discrepancy you asked me to flag

Your brief writes the target as

```
z = A * delta
a = exp(z)
h_t = a_t * h_{t-1} + b_t
```

That is a *correct description of the model's semantics* and it is literally what
`mamba2.py:314` computes. But the code you actually run does **not** evaluate
`exp` once per timestep. It evaluates `exp` on **prefix sums** of `z`, exploiting

```
exp(z_{j+1} + ... + z_i)  ==  exp(z_{j+1}) · ... · exp(z_i)
```

so that a length-`T` chunk needs one cumsum and a matrix of `exp`s instead of a
serial product. **This identity is the thing a polynomial breaks.** For any
polynomial `P`,

```
P(z_{j+1} + ... + z_i)  ≠  P(z_{j+1}) · ... · P(z_i)
```

So "replace `exp` with `P`" is ambiguous, and the two readings are *different
models*:

| reading | meaning | FHE relevance |
|---|---|---|
| **per-step** (what we do) | `a_t = P(z_t)`, then multiply the `a_t` together along the sequence | correct. FHE must produce a decay factor from an encrypted `z_t`; the depth cost is `depth(P)` once, plus the depth of the multiplicative accumulation |
| **prefix-sum** | `P(cumsum(z))` | wrong for us. It assumes you can cheaply sum `z` over time *and* that one `P` call covers a whole prefix, i.e. it needs `P` to be accurate on `[-L·|z|, 0]`, an interval that grows with sequence length. Useless at `L=2048` |

**This project takes the per-step reading**, and therefore uses a recurrence /
chunked-product formulation rather than the cumsum formulation. When `P = exp`
the two agree exactly, and `real_mamba/tests` asserts that numerically against
`ssd_minimal_discrete`. See [`real_mamba/reference_ssd.py`](real_mamba/reference_ssd.py).

---

## 0.3 Tensor shapes

Symbols, as the code names them, for `state-spaces/mamba2-130m`
(`config.json`: `d_model=768`, `n_layer=24`, `d_intermediate=0`,
`vocab_size=50277`, `ssm_cfg={"layer": "Mamba2"}`), combined with the `Mamba2`
defaults at `mamba2.py:38-61` (`d_state=128`, `headdim=64`, `expand=2`,
`ngroups=1`, `chunk_size=256`):

```
B  batch
L  seqlen
d_model  = 768
d_inner  = expand * d_model = 1536      (= d_ssm, since d_intermediate = 0)
headdim  = 64
nheads   = d_inner // headdim = 24
d_state  = 128
ngroups  = 1
```

| quantity | shape | dtype | where |
|---|---|---|---|
| `u` (block input) | `(B, L, 768)` | bf16/fp16/fp32 | `mamba2.py:176` |
| `zxbcdt` | `(B, L, 2*1536 + 2*1*128 + 24) = (B, L, 3352)` | model dtype | `mamba2.py:181` |
| `dt` raw slice | `(B, L, 24)` | model dtype | `mamba2.py:211` |
| `dt` after softplus (`delta`) | `(B, L, 24)` | fp32 in kernel | `mamba2.py:313` / kernel `ssd_chunk_state.py:72-80` |
| `A` | `(24,)` | **fp32 always** (`.float()` at `mamba2.py:182`) | `mamba2.py:182` |
| `z = A * dt` | `(B, L, 24)` conceptually | fp32 | kernel: `ssd_chunk_state.py:84` |
| `dA_cumsum` (what is actually stored) | `(B, 24, nchunks, 256)` | fp32 | `ssd_chunk_state.py:724-725` |
| `a = exp(z)` | `(B, L, 24)` conceptually | fp32 | only materialised at `mamba2.py:314` as `(B, 24)` for one step |
| `x` | `(B, L, 24, 64)` | model dtype | `mamba2.py:245` |
| `B`, `C` | `(B, L, 1, 128)` | model dtype | `mamba2.py:248-249` |
| `D` | `(24,)` | model dtype | `mamba2.py:139` |
| SSM state `h` | `(B, 24, 64, 128)` | | `mamba2.py:352-353` |

**The single most important shape fact:** `A` is `(nheads,)` — one scalar per
head — so `z` and `a` carry only `B·L·24` distinct values, **not** one per
state-dimension. Per token that is 24 numbers out of 3352 projected channels.
The transition we are attacking is a genuinely tiny, low-dimensional object.
(Mamba-1's `selective_scan_ref` broadcasts it out to `(B, 1536, L, 128)`, which is
`682×` redundant — do not let that mislead you about the real cost.)

**`dt_limit` / clamping.** Default `dt_limit=(0.0, inf)` (`mamba2.py:55`), in
which case the clamp is skipped entirely at the Python level
(`mamba2.py:183`). Inside the Triton kernel a clamp is applied unconditionally
(`ssd_chunk_state.py:80`, `dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)`) but
with `(0.0, inf)` it is a no-op. The pretrained 130M checkpoint uses the default,
so **there is no `dt` clamp in the model we are studying** — `z`'s range is set
by the data, which is why Part 6 exists.

---

## 0.4 Does the production implementation hide this behind fused kernels?

**Yes, completely.** Both branches of `Mamba2.forward` are fused:

| path | condition | entry point | pure PyTorch? |
|---|---|---|---|
| fully fused | `use_mem_eff_path=True` (the default) and no `inference_params` | `mamba_split_conv1d_scan_combined` (`mamba2.py:186`) | no — Triton. Also fuses conv1d, SiLU, softplus, RMSNormGated and `out_proj` |
| "unfused" | `use_mem_eff_path=False` | `mamba_chunk_scan_combined` (`mamba2.py:245`) | **still no** — also Triton. Only conv1d/norm/projections come out of the kernel |
| decode step | `inference_params` set, `seqlen_offset > 0` | `Mamba2.step` → `selective_state_update` (Triton) or the `torch.exp` fallback at `mamba2.py:314` | only the fallback branch |

So setting `use_mem_eff_path=False` is **not** enough to see `exp(A*dt)`; people
assume it is. `softplus` itself is inside the kernel
(`ssd_chunk_state.py:77`), so even `delta` is not observable from Python on the
default path.

Pure-PyTorch code that exists in the repo:

* `ssd_minimal.py:ssd_minimal_discrete` — pure PyTorch, but cumsum-form (see 0.2 ii),
  and its docstring types `A: (batch, length, n_heads)` meaning `A*dt`.
* `ssd_combined.py:ssd_chunk_scan_combined_ref` (line 687) — calls `chunk_state_ref`
  etc., pure PyTorch, still cumsum-form.
* `ssd_combined.py:ssd_selective_scan` (line 728) — routes Mamba-2 args into
  `selective_scan_fn`, which is `SelectiveScanFn.apply` and **requires the
  `selective_scan_cuda` extension** (`selective_scan_interface.py:26-31` raises if
  missing). The *pure* version is `selective_scan_ref`, but `ssd_selective_scan`
  does not call it.

---

## 0.5 Which path is easiest to modify safely — and what we chose

Candidates, and why they were rejected or kept:

1. **Edit the Triton kernels.** Rejected. The per-step `a` does not exist there;
   we would have to restructure the chunk algebra inside three kernels plus their
   backward passes. High risk, unreadable for undergraduates.
2. **`Mamba2.step()` fallback (`mamba2.py:314`).** Tempting — it is the one exact
   match — but it is single-token decode only. We cannot measure perplexity on a
   validation set through it at reasonable speed, and it never runs when the
   Triton kernel is installed.
3. **`ssd_minimal_discrete`.** Good teaching object, wrong algebra for us
   (cumsum-form), so it cannot express `a_t = P(z_t)`.
4. **Our own pure-PyTorch SSD in per-step product form, monkey-patched over
   `Mamba2.forward`, reusing the layer's own parameters. ← chosen.**

Choice 4 means:

* `in_proj`, `conv1d`, `SiLU`, `softplus`, `RMSNormGated`, `D` skip and `out_proj`
  are all recomputed from **the pretrained layer's own weights**, so nothing but
  the SSM inner loop changes;
* the only swappable object is a `Transition` module mapping `z → a`
  (`ExactExp` / `PolyExp2` / `PolyExp3` / `PolyExp4`);
* we never touch `torch.exp` globally, and `A = -exp(A_log)` (a plaintext
  parameter `exp`) is deliberately left alone;
* `SiLU`, `RMSNorm` and `softplus` are untouched, as instructed — they are
  separate future work.

The correctness obligation this creates: **with `ExactExp`, our forward must
reproduce the official fused forward.** That equivalence is a test, not an
assumption — `real_mamba/tests/test_parity.py`.

---

## 0.6 Consequences for the polynomial we need

Two facts from this inspection already constrain Part 2, before we fit anything:

1. `z = A·dt` with `A ∈ [-16, -1]`-ish and `dt = softplus(·)` initialised small.
   `z` is **negative and unbounded below** — there is no clamp (0.3). So the fit
   interval is an empirical question, answered in Part 6, not a guess. `[-8, 0]`
   is a starting hypothesis only.
2. `a` is used as a *repeated multiplier*. `exp(z) ∈ (0, 1]` for `z ≤ 0`, which is
   what keeps the recurrence contractive. A polynomial has no such guarantee: it
   can go negative, and it diverges outside its fit interval. Both failures are
   amplified by `L` multiplications. Hence Parts 4 and 8 measure
   `frac(a < 0)`, `frac(a > 1)` and state norms, not just pointwise error.
