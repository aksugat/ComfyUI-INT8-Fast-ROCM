"""
Speed benchmark for triton_int4_mm.py (RDNA3-tuned autotune version), at
shapes pulled from a real Krea2 checkpoint (same shapes used in
benchmark_speed.py for the int8 kernel).

Benchmarks the raw GEMM only (int8 activations x packed-int4 weights ->
int32), since the kernel doesn't have a fused dequant epilogue yet.

RUN:
    H:\\comfyui-rocm\\python_env\\python.exe benchmark_speed_int4.py

Put this in the same folder as triton_int4_mm.py.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton_int4_mm import triton_int4_mm
import triton_int4_mm as int4_mod

_INT4_MAX = 7


def pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def quantize_signed_int4_rowwise_packed(x: torch.Tensor):
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scales = absmax / _INT4_MAX
    q = (x / scales).round().clamp_(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return pack_int4_row_major(q), scales.reshape(x.shape[0]).to(torch.float32)


IMG_M = 4096
TXT_M = 512

SHAPES = [
    ("img-stream attn/proj 6144->6144 (count 86)",  IMG_M, 6144, 6144),
    ("img-stream mlp up 6144->16384 (count 56)",     IMG_M, 6144, 16384),
    ("img-stream mlp down 16384->6144 (count 56)",   IMG_M, 16384, 6144),
    ("img-stream proj 6144->1536 (count 56)",        IMG_M, 6144, 1536),
    ("txt-stream attn/proj 2560->2560 (count 20)",   TXT_M, 2560, 2560),
    ("txt-stream mlp up 2560->6912 (count 8)",       TXT_M, 2560, 6912),
    ("txt-stream mlp down 6912->2560 (count 4)",     TXT_M, 6912, 2560),
]

WARMUP_ITERS = 5
TIMED_ITERS = 20


def bench(fn, *args, **kwargs):
    for _ in range(WARMUP_ITERS):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(TIMED_ITERS):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / TIMED_ITERS * 1000.0  # ms per call


def main():
    if not torch.cuda.is_available():
        print("No GPU visible to torch.")
        return

    device = "cuda"
    torch.manual_seed(0)

    print(f"{'Shape':45s} {'time':>12s}")
    print("-" * 60)

    for label, M, K, N in SHAPES:
        a_int8 = torch.randint(-_INT4_MAX, _INT4_MAX + 1, (M, K), device=device, dtype=torch.int8).contiguous()
        w_fp = torch.randn(N, K, device=device, dtype=torch.float32)
        w_packed, _ = quantize_signed_int4_rowwise_packed(w_fp)
        w_packed = w_packed.contiguous()

        try:
            ms = bench(triton_int4_mm, a_int8, w_packed)
            time_str = f"{ms:.3f} ms"
        except Exception as e:
            time_str = f"FAIL: {type(e).__name__}"

        print(f"{label:45s} {time_str:>12s}")
        if hasattr(int4_mod, "triton_int4_mm_kernel") and hasattr(int4_mod.triton_int4_mm_kernel, "best_config"):
            print(f"  -> winning config: {int4_mod.triton_int4_mm_kernel.best_config}")


if __name__ == "__main__":
    main()
