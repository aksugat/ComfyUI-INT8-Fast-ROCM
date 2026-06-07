# Comfy INT8 Acceleration

This node speeds up Flux2, Ideogram4, Chroma, Z-Image, Ernie Image in ComfyUI by using INT8 quantization, delivering between 1.5~2x faster inference on my 3090 depending on the model. It should work on any NVIDIA GPU with enough INT8 TOPS. It appears to be faster than FP8 on 40-Series and above as well. 
Works with lora, torch compile.

---

Updates:

2026-24-05:

RAM usage for lora loading is fixed and on par with base comfy.

RAM usage for model loading is fixed.

Only thing that remains is on the fly quantization will create an extra int8 copy in memory, but it is too much of a hassle to work around. Please rely on swap or pre converted models if this is an issue.

Fixed an issue with loading loras on models that include .bias layers (WAN, LTX2.X) which would cause a OOM error.

2026-15-05:

Bringing back stochastic lora. Some loras appear to need it, others don't, try it if your lora is not working and you don't like pre-lora. TLDR is "sometimes it really helps, sometimes its a little worse". See our measurements [here](https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/RAMExp/Metrics.md#some-loras-require-stochastic-lora-to-work).

Attempt at reducing RAM usage

Fixed an issue with Pre-Lora crashing on windows

2026-10-05:

Overhauled the entire lora system. Normal lora loader node works now, no need for specialized lora loaders.

Converted QuaRot to ConvRot, which is a small but free quality gain.

Added Pre-Lora node, which you can connect to the INT8 Model loader to merge loras before utilizing on the fly quantization. 

For more info on quality of convrot, lora approaches see the [Metrics](Metrics.md)

---

Pre-quantized checkpoints were recommended for most architectures, but on-the-fly quantization with ConvRot is better in all cases.
However, ConvRot is also a little slower, so these prequantized models are still useful. Avoid using INT8 Tensorwise models.

**Shoutout to [vistralis](https://huggingface.co/vistralis) for these:** 

| Model | Link |
|-------|------|
| FLUX.2-klein-base-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-9b-INT8-transformer) |
| FLUX.2-klein-base-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-4b-INT8-transformer) |
| FLUX.2-klein-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-9b-INT8-transformer) |
| FLUX.2-klein-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-4b-INT8-transformer) |

**ConvRot:**

| Model | Link |
|-------|------|
| Ideogram-4 | [Download](https://huggingface.co/bertbobson/Ideogram-4-INT8-ConvRot) |
| LTX2.3 10Eros | [Download](https://huggingface.co/bertbobson/LTX2.3-10Eros-INT8-ConvRot) |
| Sulphur2 Base (LTX2.3 Finetune) | [Download](https://huggingface.co/bertbobson/Sulphur-2-base-INT8-ConvRot) |
| Chroma1 HD | [Download](https://huggingface.co/bertbobson/ComfyUI-INT8_ConvRot/blob/main/Chroma1-HD-int8-ConvRot.safetensors) |
| Ernie Image | [Download](https://huggingface.co/bertbobson/ComfyUI-INT8_ConvRot/blob/main/Ernie-Image-Base-int8-convrot.safetensors) |
| Anima Preview 3 | [Download](https://huggingface.co/bertbobson/ComfyUI-INT8_ConvRot/blob/main/anima-preview3-base-int8-ConvRot.safetensors) |
| Flux 2 Klein Base | [Download](https://huggingface.co/bertbobson/ComfyUI-INT8_ConvRot/blob/main/flux-2-klein-base-9b-int8-ConvRot.safetensors) |
| LTX2.3 Dev | [Download](https://huggingface.co/bertbobson/ComfyUI-INT8_ConvRot/blob/main/ltx-2.3-22b-dev-int8-ConvRot.safetensors) |
| LTX2.3 Distilled | [Download](https://huggingface.co/bertbobson/ComfyUI-INT8_ConvRot/blob/main/ltx-2.3-22b-distilled-1.1-int8-ConvRot.safetensors) |
| WAN 2.2 | [Download](https://huggingface.co/bertbobson/ComfyUI-INT8_ConvRot/tree/main) |

**Outdated int8 models:**

| Model | Link |
|-------|------|
| Chroma1-HD² | ~~[Download](https://huggingface.co/bertbobson/Chroma1-HD-INT8Tensorwise)~~ |
| Z-Image-Base¹ | ~~[Download](https://huggingface.co/bertbobson/Z-Image-Base-INT8-QUIP)~~ 
| Z-Image-Turbo² | ~~[Download](https://huggingface.co/bertbobson/Z-Image-Turbo-INT8-Tensorwise)~~ |
| Anima | [Download](https://huggingface.co/bertbobson/Anima-INT8-QUIP) |

¹Z-Image Base weights have been Deprecated in favor of Convrot OTF, which is higher quality.

²Tensorwise models are worse than on the fly quantization since we switched to row-wise INT8


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

## FAQ:

Q: It crashes on my 20-Series GPU!

A: Official triton builds have dropped SM75 support in 3.3. You either have to downgrade triton, build it yourself, or find a build that supports it.


Q: Pre-Lora doesn't work?

A: Pre-Lora is for baking lora into bf16/fp16 weights before on-the-fly quantization. You can not pre-lora an already quantized checkpoint.


# Requirements:
Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)

Triton

Windows untested, but I hear triton-windows exists.

# Credits:

## dxqb for the *entirety* of the INT8 code during the very early versions of this node, it would have been impossible without them:
https://github.com/Nerogar/OneTrainer/pull/1034

If you have a 30-Series GPU, OneTrainer is also the fastest current lora trainer thanks to this. Please go check them out!!

## newgrit1004 for the base ConvRot code we modified into proper ConvRot 
https://github.com/newgrit1004/ComfyUI-ZImage-Triton

## silveroxides for providing a base to hack the INT8 conversion code onto.
https://github.com/silveroxides/convert_to_quant

## Also silveroxides for showing how to properly register new data types to comfy
https://github.com/silveroxides/ComfyUI-QuantOps

## The unholy trinity of AI slopsters I used to glue all this together over the course of multiple months now
