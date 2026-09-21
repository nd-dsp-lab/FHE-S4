# third_party/mamba — read-only reference copy

This is a trimmed copy of the official implementation:

    https://github.com/state-spaces/mamba
    commit e9594ce1c732d97440f0332fdc43170a2294dbfa  (2026-07-23)

It is here so that you can read the real source while following this project,
and so that the line numbers quoted in `PART0_SOURCE_INSPECTION.md` stay valid
even if upstream moves. **We never import from this copy.** The CUDA kernel
sources (`csrc/`), benchmarks and assets were deleted to keep the checkout small.

If you install `mamba-ssm` on the GPU machine, that installed package is what
gets imported — not this folder.
