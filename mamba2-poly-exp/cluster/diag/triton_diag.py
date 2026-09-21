"""Does Triton work at all on this node, or only the mamba kernels?

Must live in a real file: @triton.jit reads its own source with
inspect.getsourcelines(), which fails for a script piped in on stdin.
"""
import subprocess

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

print("triton", triton.__version__, "| torch", torch.__version__)
p = torch.cuda.get_device_properties(0)
print(f"device {p.name} sm_{p.major}{p.minor} {p.total_memory/1024**3:.1f} GB")
print("driver:", subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip())
print("ptxas :", subprocess.run(["bash", "-lc", "which ptxas && ptxas --version | tail -2"],
                                capture_output=True, text=True).stdout.strip() or "not on PATH")


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)


def report(label, fn):
    print(f"[{label}] ...", end=" ", flush=True)
    try:
        r = fn()
        torch.cuda.synchronize()
        print("OK" if r is None else f"OK {r}")
        return True
    except Exception as e:                                   # noqa: BLE001
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


x = torch.randn(4096, device="cuda")
y = torch.randn(4096, device="cuda")
o = torch.empty_like(x)

report("1 trivial triton kernel, default stages",
       lambda: add_kernel[(16,)](x, y, o, 4096, BLOCK=256))
report("2 same kernel, num_stages=4 num_warps=8 (Ampere-style pipelining)",
       lambda: add_kernel[(16,)](x, y, o, 4096, BLOCK=256, num_stages=4, num_warps=8))


def mamba_scan():
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    B, L, H, P, N = 1, 128, 4, 32, 32
    xx = torch.randn(B, L, H, P, device="cuda")
    dt = F.softplus(torch.randn(B, L, H, device="cuda") - 2)
    A = -torch.exp(torch.rand(H, device="cuda"))
    Bm = torch.randn(B, L, 1, N, device="cuda")
    Cm = torch.randn(B, L, 1, N, device="cuda")
    return tuple(mamba_chunk_scan_combined(xx, dt, A, Bm, Cm, chunk_size=64,
                                           D=None, z=None).shape)


def mamba_layernorm():
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
    h = torch.randn(2, 64, 768, device="cuda")
    w = torch.ones(768, device="cuda")
    layer_norm_fn(h, w, None, residual=None, prenorm=True, is_rms_norm=True)


ok_scan = report("3 mamba chunk-scan kernel", mamba_scan)
ok_ln = report("4 mamba layer_norm_fn (every Block uses it)", mamba_layernorm)

print()
print("VERDICT:", "mamba Triton kernels usable on this node"
      if (ok_scan and ok_ln) else
      "mamba Triton kernels NOT usable in this env on sm_75")
