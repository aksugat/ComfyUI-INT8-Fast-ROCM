"""
Correctness smoke test for triton_int4_mm.py (RDNA3-tuned autotune version).

Reimplements the quantize/pack logic from int4_quant.py inline (rather than
importing that file directly) so this test has no ComfyUI dependencies and
can run standalone.

RUN:
    H:\\comfyui-rocm\\python_env\\python.exe smoke_test_int4.py

Put this in the same folder as triton_int4_mm.py.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton_int4_mm import triton_int4_mm

_INT4_MAX = 7


def pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def quantize_signed_int4_rowwise(x: torch.Tensor):
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _INT4_MAX
    q = (x / scales).round().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return q, scales.reshape(x.shape[0]).to(torch.float32)


def quantize_signed_int4_rowwise_packed(x: torch.Tensor):
    q, scales = quantize_signed_int4_rowwise(x)
    return pack_int4_row_major(q), scales


def main():
    if not torch.cuda.is_available():
        print("No GPU visible to torch.")
        return

    torch.manual_seed(0)
    device = "cuda"

    print(f"{'M':>6s} {'K':>6s} {'N':>6s} {'max abs err':>14s}")
    print("-" * 40)

    for M, K, N in [(64, 32, 64), (128, 256, 128), (256, 1024, 256), (128, 6144, 128)]:
        a_int8 = torch.randint(-_INT4_MAX, _INT4_MAX + 1, (M, K), device=device, dtype=torch.int8).contiguous()

        w_fp = torch.randn(N, K, device=device, dtype=torch.float32)
        w_packed, w_scale = quantize_signed_int4_rowwise_packed(w_fp)
        w_packed = w_packed.contiguous()

        c_int32 = triton_int4_mm(a_int8, w_packed)

        lo = (w_packed.to(torch.int32) & 0x0F)
        hi = ((w_packed.to(torch.int32) >> 4) & 0x0F)
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        w_unpacked = torch.stack([lo, hi], dim=-1).reshape(N, K)

        ref = (a_int8.to(torch.float64) @ w_unpacked.to(torch.float64).T).round().to(torch.int64)

        err = (c_int32.to(torch.int64) - ref).abs().max().item()
        print(f"{M:>6d} {K:>6d} {N:>6d} {err:>14d}")

    print("\nExpected: max abs err = 0 for every shape, since this is exact")
    print("integer math with no rounding anywhere in the GEMM itself -- any")
    print("nonzero value means a real bug, not a precision/tuning issue.")


if __name__ == "__main__":
    main()
