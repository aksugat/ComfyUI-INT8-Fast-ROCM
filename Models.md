
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
