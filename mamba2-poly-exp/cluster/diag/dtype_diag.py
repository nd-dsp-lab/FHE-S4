"""Can the mamba SSD kernel compile on sm_75 in ANY dtype?

Turing (sm_75) has fp16 tensor cores but NO bf16 support at all. The chunk-scan
kernel uses tl.dot, so the input dtype decides which MMA layout Triton tries to
emit -- and `IndexError: map::at` is a layout-map lookup failing. If fp16 works
and fp32 does not, that is the whole story.
"""
import torch
import torch.nn.functional as F

p = torch.cuda.get_device_properties(0)
print(f"device {p.name} sm_{p.major}{p.minor}")
print("torch says bf16 supported:", torch.cuda.is_bf16_supported())

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def try_scan(dtype, chunk, headdim=32, dstate=32, seqlen=128):
    B, H = 1, 4
    d = dict(device="cuda", dtype=dtype)
    x = torch.randn(B, seqlen, H, headdim, **d)
    dt = F.softplus(torch.randn(B, seqlen, H, device="cuda", dtype=torch.float32) - 2)
    A = -torch.exp(torch.rand(H, device="cuda", dtype=torch.float32))
    Bm = torch.randn(B, seqlen, 1, dstate, **d)
    Cm = torch.randn(B, seqlen, 1, dstate, **d)
    label = f"{str(dtype).replace('torch.',''):9s} chunk={chunk:<4d} headdim={headdim:<3d} dstate={dstate:<3d}"
    try:
        out = mamba_chunk_scan_combined(x, dt, A, Bm, Cm, chunk_size=chunk, D=None, z=None)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out).all())
        print(f"  {label}  OK  shape={tuple(out.shape)} finite={finite}")
        return True
    except Exception as e:                                   # noqa: BLE001
        print(f"  {label}  FAILED: {type(e).__name__}: {str(e)[:60]}")
        return False


print("\n--- dtype sweep ---")
ok = {}
for dt_ in (torch.float16, torch.float32, torch.bfloat16):
    ok[dt_] = any([try_scan(dt_, 64), try_scan(dt_, 128), try_scan(dt_, 256)])

print("\n--- realistic 130M shapes (headdim 64, dstate 128) in the dtype that worked ---")
for dt_, good in ok.items():
    if good:
        try_scan(dt_, 128, headdim=64, dstate=128, seqlen=512)

print("\nVERDICT:", ", ".join(f"{str(k).replace('torch.','')}={'usable' if v else 'NO'}"
                             for k, v in ok.items()))
