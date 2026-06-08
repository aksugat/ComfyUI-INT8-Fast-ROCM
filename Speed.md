# Speed:

Measured on a 3090 at 1024x1024, 26 steps with Flux2 Klein Base 9B.

| Format | Speed (s/it) ↓ | Relative Speedup |
|-------|--------------|------------------|
| bf16 | 2.07 | 1.00× |
| bf16 compile | 2.24 | 0.92× |
| fp8 | 2.06 | 1.00× |
| int8 | 1.64 | 1.26× |
| int8 compile ★| 1.04 | 1.99× |
| gguf8_0 compile | 2.03 | 1.02× |

3090, Qwen Image 2512.

| Format | Speed (s/it) ↓ |
|-------|--------------|
| Nunchaku INT4 Best Quality | 1.21 |
| Nunchaku INT4 with R128 Lora | 1.36 |
| INT8 ConvRot compile | 1.26 |
| INT8 Row compile ★| 1.18 |
| INT8 R128 Lora | No slowdown, except if dynamic. |

I would also like to point out that we beat Nunchaku INT4 on every quality measurement in the [Quality Metrics](Metrics.md)

Additionally, the quality of loras applied with [this nunchaku lora node](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader) appears to be degraded.

Klein 9B, Measured on an 8gb 5060, same settings as the 3090 run:

| Format | Speed (s/it) ↓ | Relative Speedup |
|-------|--------------|------------------|
| fp8 | 3.04 | 1.00× |
| fp8 fast | 3.00 | 1.00× |
| fp8 compile | couldn't get to work | ??× |
| int8 | 2.53 | 1.20× |
| int8 compile ★| 2.25 | 1.35× |

8gb RTX 5060, Anima, Comfy version from 2026-05-02, Pytorch 2.11+CU13.0, latest kitchen triton and everything else

| Format | Speed (it/s) ↑ |
|-------|--------------|
| bf16 | 0.78 |
| INT8 ConvRot | 1.12 |
| INT8 Row | 1.24 |
| INT8 ConvRot Compile | 1.47 |
| MXFP8 | 0.89 |
| MXFP8 --fast | 0.93 |
| MXFP8 + Compile | Still failing. |

Finally have gotten compile with --fast to work with mxfp8, PyTorch 2.13.0.dev20260511+cu132, RTX5060 same as before.

Quality results for this run, can be found here: [Anima Results](Metrics.md#anima-on-a-5060)

| Format | Speed (it/s) ↑ |
|-------|--------------|
| MXFP8 --fast + Compile | 1.37it |
| INT8 ConvRot + Compile | 1.47it |
