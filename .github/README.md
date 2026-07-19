ComfyUI-INT8-Fast-ROCM (AMD RDNA3-Tuned)
Fast INT8 (W8A8) quantized inference for diffusion models in ComfyUI, with Triton kernels specifically tuned for AMD RDNA3 consumer GPUs — validated on an RX 7800 XT (gfx1101).

Forked from patientx/ComfyUI-INT8-Fast-ROCM, itself built on BobJohnson24/ComfyUI-INT8-Fast.

What's different in this fork
The original Triton autotune configs were sized for NVIDIA/CDNA hardware — large tiles (BLOCK_N up to 256), num_warps=8, deep 3-4 stage pipelining. That's a reasonable fit for Instinct-class GPUs with hundreds of compute units and 64-wide wavefronts. It's a poor fit for RDNA3 consumer cards, which have far fewer CUs (60 on the 7800 XT) and 32-wide wavefronts.

This fork adds a second config set tuned specifically for that hardware:

waves_per_eu as an autotune key (the ROCm/Triton occupancy knob for this backend, in place of CDNA's MFMA-specific matrix_instr_nonkdim)
Smaller tiles (≤128) across more configs, num_stages=1-2
Additional BLOCK_K=128 configs for K-heavy layers (large reduction dimension, small output — e.g. MLP down-projections)
Fixed an other=0.0 float-literal-on-int8-load correctness issue
Benchmarks
Measured on an RX 7800 XT (gfx1101), against real layer shapes pulled directly from a quantized Krea2 checkpoint (image stream dim=6144, text stream dim=2560), 3 runs averaged and weighted by how often each shape actually occurs in the model:

Layer type	Original configs	RDNA3-tuned	vs. torch._int_mm
Attention/proj (6144→6144)	5.84 ms	5.51 ms	2.2x faster
MLP up (6144→16384)	15.36 ms	14.15 ms	1.5x faster
MLP down (16384→6144)	6.46 ms	5.72 ms	4% faster
Small proj (6144→1536)	1.73 ms	1.60 ms	int_mm slightly faster
Weighted across a full forward pass: ~5% faster than the original configs, ~16% faster than running the same model in plain fp16.

One honest caveat: at small shapes (≤1536-dim layers, small batch), plain fp16 can beat int8 outright — the quantize/dequant overhead doesn't pay for itself on tiny GEMMs. Worth excluding very small layers from quantization rather than assuming int8 always wins.

Tested configuration
GPU: RX 7800 XT (gfx1101, RDNA3, 60 CUs)
PyTorch: 2.12.0+rocm7.15.0a (nightly)
Triton: triton-windows 3.6.0 and 3.7.1 (both confirmed correct and equivalent in speed — no regression across the upgrade)
OS: Windows, portable/embeddable Python distribution
Known limitations
RDNA1/RDNA2 (gfx10xx) are not supported and will hang the GPU. Neither architecture has WMMA or MFMA matrix-core instructions, which is what tl.dot on int8 inputs compiles to. This isn't a tuning limitation — there's no working instruction path for this kernel to target on that hardware. (ComfyUI core's own native int8 path gates around this explicitly; this fork's kernel currently does not — an arch check before attempting the Triton path would be a good addition.)
Only validated on RDNA3 (gfx1101) so far. RDNA4 (gfx12xx) shares the same 64KB/CU LDS budget and matrix-core instruction support, so the kernel should work, but tile configs haven't been benchmarked there — CU count and occupancy math differ enough (84 CUs on a 7900 XT, 64 on a 9070 XT vs. 60 here) that a fresh autotune pass is worth doing rather than assuming these exact numbers transfer.
Requires Triton ≥3.6 for this fork's kernel (tested). ComfyUI core's own native Triton backend separately requires ≥3.7 to avoid a libdevice.rint crash on HIP — that requirement is about core's path, not this node's kernel.
