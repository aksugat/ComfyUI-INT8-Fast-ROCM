"""
Speed benchmark: original Triton configs vs. AMD RDNA3-tuned configs vs.
torch._int_mm vs. plain fp16 matmul, at shapes representative of DiT/UNet
linear layers (attention QKV/proj, MLP up/down projections).

SETUP:
  Both kernel files must be in this same folder:
    - int8_fused_kernel.py       (your original file)
    - int8_fused_kernel_amd.py   (the AMD RDNA3-tuned version)

RUN:
    H:\\comfyui-rocm\\python_env\\python.exe benchmark_speed.py

Edit SHAPES below if you know your model's actual hidden_dim / mlp_dim —
current values are pulled directly from a real Krea2 checkpoint via
dump_shapes.py (image stream dim=6144, text stream dim=2560).
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import int8_fused_kernel as orig
import int8_fused_kernel_amd as amd

# (M, K, N) = (tokens, in_features, out_features)
# Real Krea2 layer shapes (from dump_shapes.py against krea2_int8_convrot_row2.safetensors).
# Image-stream token count (M) is a placeholder for a ~1MP-ish generation --
# adjust to your typical resolution. Text-stream M is a placeholder sequence
# length -- adjust to your typical prompt token count.
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


def run_int_mm(x_int8, w_int8, x_scale, w_scale, compute_dtype):
    # Manual per-row-scale int8 matmul via torch._int_mm, mirroring what the
    # fallback path in int8_quant.py does when Triton isn't used.
    acc = torch._int_mm(x_int8, w_int8.T)  # int32
    out = acc.to(torch.float32) * x_scale.reshape(-1, 1) * w_scale.reshape(1, -1)
    return out.to(compute_dtype)


def main():
    if not torch.cuda.is_available():
        print("No GPU visible to torch.")
        return

    device = "cuda"
    compute_dtype = torch.float16
    torch.manual_seed(0)

    print(f"{'Shape':30s} {'orig-triton':>12s} {'amd-triton':>12s} {'int_mm':>12s} {'fp16-ref':>12s}")
    print("-" * 82)

    for label, M, K, N in SHAPES:
        x = torch.randn(M, K, device=device, dtype=compute_dtype)
        w_fp16 = torch.randn(N, K, device=device, dtype=compute_dtype)

        # Shared quantized inputs so all three int8 paths do identical math.
        x_int8, x_scale = orig.triton_quantize_rowwise(x)
        w_int8, w_scale = orig.triton_quantize_rowwise(w_fp16)

        results = {}

        try:
            ms = bench(orig.triton_int8_linear_per_row, x, w_int8, w_scale, compute_dtype=compute_dtype)
            results["orig"] = f"{ms:.3f} ms"
        except Exception as e:
            results["orig"] = f"FAIL: {type(e).__name__}"

        try:
            ms = bench(amd.triton_int8_linear_per_row, x, w_int8, w_scale, compute_dtype=compute_dtype)
            results["amd"] = f"{ms:.3f} ms"
            best = amd._int8_matmul_dequant_per_row_kernel.best_config
            results["amd_config"] = str(best)
        except Exception as e:
            results["amd"] = f"FAIL: {type(e).__name__}"
            results["amd_config"] = "n/a"

        try:
            ms = bench(run_int_mm, x_int8, w_int8, x_scale.reshape(-1), w_scale.reshape(-1), compute_dtype)
            results["int_mm"] = f"{ms:.3f} ms"
        except Exception as e:
            results["int_mm"] = f"FAIL: {type(e).__name__}"

        try:
            ms = bench(torch.nn.functional.linear, x, w_fp16)
            results["fp16"] = f"{ms:.3f} ms"
        except Exception as e:
            results["fp16"] = f"FAIL: {type(e).__name__}"

        print(f"{label:30s} {results['orig']:>12s} {results['amd']:>12s} {results['int_mm']:>12s} {results['fp16']:>12s}")
        print(f"  -> amd winning config: {results['amd_config']}")


if __name__ == "__main__":
    main()
