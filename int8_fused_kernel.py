"""
INT8 W8A8 fused quant + GEMM kernels, tuned for AMD RDNA3 (gfx1101 / RX 7800 XT).

Key differences from an NVIDIA/CDNA-tuned version:
  - RDNA3 executes int8 matmul via WMMA, not MFMA. The `matrix_instr_nonkdim`
    / `kpack` autotune keys from AMD's MI300 (CDNA) tuning guides do NOT apply
    here and are omitted.
  - RDNA3 wavefronts are 32 threads (CDNA is 64), and the 7800 XT has 60 CUs
    (vs. hundreds on Instinct parts). Large 256-wide tiles starve occupancy
    on a 60-CU part, so tiles are kept smaller (<=128) with more configs at
    num_warps=4/8.
  - num_stages is swept at 1-2. Deep software pipelining (3-4 stages) assumes
    register/LDS headroom tuned around CDNA occupancy; on RDNA3 consumer
    parts this tends to hurt more than help, so it's included but not
    defaulted to.
  - `waves_per_eu` is included as an autotune key, which is the AMD-recommended
    ROCm/Triton knob for controlling occupancy on this backend.

BEFORE RELYING ON THIS: run the smoke test at the bottom of this file first.
Whether int8 x int8 -> int32 tl.dot actually lowers to WMMA (vs. silently
falling back to a slow path) depends on your exact Triton + ROCm version.
Recommended minimum: ROCm 6.1+, a Triton build with gfx1101 WMMA support.
"""

import torch
import triton
import triton.language as tl

# =============================================================================
# Kernel 1: Fused Row-wise Quantization (FP16/BF16 -> INT8 + Scale)
# =============================================================================

@triton.jit
def _quantize_rowwise_kernel(
    x_ptr,      # Input pointer (FP16/BF16)
    y_ptr,      # Output pointer (INT8)
    s_ptr,      # Scale pointer (FP32)
    n_elements, # Number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    x_row_ptr = x_ptr + row_idx * n_elements
    y_row_ptr = y_ptr + row_idx * n_elements

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)

    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)

    scale = tl.maximum(max_val / 127.0, 1e-30)

    q_f = x / scale
    q_i = tl.floor(q_f + 0.5)
    q_i = tl.clamp(q_i, -128.0, 127.0)
    q_i = q_i.to(tl.int32)

    tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)
    tl.store(s_ptr + row_idx, scale.to(tl.float32))


def triton_quantize_rowwise(x: torch.Tensor):
    """
    Input: [Batch, Dim] (float16/bfloat16/float32)
    Output: [Batch, Dim] (int8), [Batch, 1] (float32)
    """
    rows, cols = x.shape
    y = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((rows, 1), device=x.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(cols)
    if BLOCK_SIZE < 128:
        BLOCK_SIZE = 128

    # This kernel loads the entire row in one block. If a layer's hidden dim
    # exceeds this, the max-abs reduction (and therefore the scale) would be
    # silently wrong instead of erroring. Fail loudly instead.
    assert cols <= BLOCK_SIZE, (
        f"Row width {cols} > BLOCK_SIZE {BLOCK_SIZE}; this single-block "
        f"quantizer needs a tiled reduction for dims this large."
    )

    grid = (rows,)
    _quantize_rowwise_kernel[grid](x, y, s, cols, BLOCK_SIZE=BLOCK_SIZE)
    return y, s


# =============================================================================
# AMD RDNA3 (gfx1101) autotune configs
# =============================================================================
# Smaller tiles than an NVIDIA/CDNA config list: 60 CUs total, 32-wide waves.
# waves_per_eu is the ROCm/Triton occupancy knob (AMD's tuning docs use this
# in place of CDNA's matrix_instr_nonkdim/kpack, which don't apply to WMMA).
_AMD_RDNA3_INT8_CONFIGS = [
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2},
                  num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 4},
                  num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2},
                  num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2},
                  num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2},
                  num_stages=1, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 1},
                  num_stages=2, num_warps=8),
    triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 4},
                  num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 4},
                  num_stages=1, num_warps=4),
    # Larger BLOCK_K variants: for K-heavy layers (large reduction dim, small
    # N -- e.g. MLP down-projections), fewer/bigger K-loop iterations may
    # beat the smaller-BLOCK_K configs above.
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_SIZE_M': 8, 'waves_per_eu': 2},
                  num_stages=1, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
                  num_stages=1, num_warps=4),
]


# =============================================================================
# Kernel 2: INT8 GEMM + Fused Dequantization Epilogue (scalar weight scale)
# =============================================================================

@triton.autotune(configs=_AMD_RDNA3_INT8_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _int8_matmul_dequant_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    """
    Computes: C = ((A @ B) * (scale_a * scale_b)) + bias
    A: [M, K] int8
    B: [N, K] int8 (read as [K, N] via transposed strides)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale_a = tl.load(a_scale_ptr + offs_am)   # [BLOCK_M]
    scale_b = tl.load(b_scale_ptr)             # scalar

    c = accumulator.to(tl.float32)
    total_scale = scale_a[:, None] * scale_b
    c = c * total_scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)     # [BLOCK_N]
        c = c + bias[None, :]

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale, bias=None, compute_dtype=torch.float16):
    """
    Fused pipeline for W8A8 Linear Layer.
    """
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])

    M, K = x_2d.shape
    N = weight.shape[0]

    x_int8, x_scale = triton_quantize_rowwise(x_2d)

    output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

    if not isinstance(weight_scale, torch.Tensor):
        weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
    else:
        weight_scale = weight_scale.to(x.device, non_blocking=True).reshape(1) if weight_scale.numel() == 1 else weight_scale.to(x.device, non_blocking=True)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else x

    _int8_matmul_dequant_kernel[grid](
        a_ptr=x_int8,
        b_ptr=weight,
        c_ptr=output,
        a_scale_ptr=x_scale,
        b_scale_ptr=weight_scale,
        bias_ptr=bias_ptr,
        M=M, N=N, K=K,
        stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
        stride_bk=weight.stride(1), stride_bn=weight.stride(0),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        HAS_BIAS=has_bias
    )

    return output.reshape(x_shape_orig[:-1] + (N,))


# =============================================================================
# Kernel 3: INT8 GEMM + Fused Dequant with Per-Row Weight Scales
# =============================================================================

@triton.autotune(configs=_AMD_RDNA3_INT8_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _int8_matmul_dequant_per_row_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    """
    Computes: C = ((A @ B) * (scale_a[:, None] * scale_b[None, :])) + bias
    A: [M, K] int8, scale_a: [M, 1] per-row activation scales
    B: [N, K] int8, scale_b: [N, 1] per-row weight scales
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale_a = tl.load(a_scale_ptr + offs_am)   # [BLOCK_M]
    scale_b = tl.load(b_scale_ptr + offs_bn)   # [BLOCK_N]

    c = accumulator.to(tl.float32)
    total_scale = scale_a[:, None] * scale_b[None, :]
    c = c * total_scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)
        c = c + bias[None, :]

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_int8_linear_per_row(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias=None, compute_dtype=torch.float16):
    """
    Fused pipeline for W8A8 Linear Layer with per-row weight quantization.
    weight_scale: [N, 1] per-row scales
    """
    x_shape_orig = x.shape
    x_2d = x.reshape(-1, x_shape_orig[-1])

    M, K = x_2d.shape
    N = weight.shape[0]

    x_int8, x_scale = triton_quantize_rowwise(x_2d)

    output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

    ws = weight_scale.reshape(N).contiguous()

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else x

    _int8_matmul_dequant_per_row_kernel[grid](
        a_ptr=x_int8,
        b_ptr=weight,
        c_ptr=output,
        a_scale_ptr=x_scale,
        b_scale_ptr=ws,
        bias_ptr=bias_ptr,
        M=M, N=N, K=K,
        stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
        stride_bk=weight.stride(1), stride_bn=weight.stride(0),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        HAS_BIAS=has_bias
    )

    return output.reshape(x_shape_orig[:-1] + (N,))


# =============================================================================
# Smoke test — run this FIRST on your machine before wiring into ComfyUI.
# It checks (a) that int8 tl.dot actually runs on your ROCm/Triton build,
# and (b) correctness against a plain PyTorch reference.
# =============================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU visible to torch (check ROCm install / HSA_OVERRIDE_GFX_VERSION "
              "if using an unofficial gfx1101 override).")
    else:
        torch.manual_seed(0)
        device = "cuda"
        M, K, N = 256, 4096, 4096

        x = torch.randn(M, K, device=device, dtype=torch.float16)
        w_fp16 = torch.randn(N, K, device=device, dtype=torch.float16)

        w_int8, w_scale = triton_quantize_rowwise(w_fp16)
        bias = torch.randn(N, device=device, dtype=torch.float16)

        out = triton_int8_linear(x, w_int8, w_scale.squeeze(-1), bias=bias)

        # Reference: dequantize weight, do the matmul in fp32, compare loosely.
        w_deq = w_int8.to(torch.float32) * w_scale  # [N, K] * [N, 1]
        ref = x.to(torch.float32) @ w_deq.T + bias.to(torch.float32)

        err = (out.to(torch.float32) - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"max abs err: {err:.4f}  max rel err: {rel:.4%}")
        print("If this errors out or rel err is large (>~5%), int8 tl.dot is "
              "likely not lowering to WMMA cleanly on this build — check "
              "`triton.__version__` and your ROCm version before debugging tile sizes.")
