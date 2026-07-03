# ComfyUI-PiD

Compact ComfyUI nodes for **NVIDIA PiD / PixelDiT** using ComfyUI-native `Comfy-Org/PixelDiT` model loading.

<img width="1058" height="604" alt="ComfyUI-PiD workflow screenshot" src="https://github.com/user-attachments/assets/cc5a9da3-94c6-4546-9574-c8387d5dffdb" />

<img width="4096" height="2048" alt="ComfyUI-PiD example output" src="https://github.com/user-attachments/assets/7ccd55ee-e571-4996-9c9c-4b5cecbb4418" />

PiD is a latent-conditioned pixel diffusion decoder/upscaler:

```text
LATENT + caption + sigma -> PiD -> IMAGE
```

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Merserk/ComfyUI-PiD.git
cd ComfyUI-PiD
python -m pip install -r requirements.txt
```

Restart ComfyUI.

Requirements: recent ComfyUI with native PixelDiT/PiD support, Python `>=3.10`, NVIDIA CUDA GPU recommended.

## Models

Most nodes can download required files automatically when `auto_download=true`.

| Use | Source | Local folder |
| --- | --- | --- |
| PiD diffusion + Gemma text encoder | Comfy-Org/PixelDiT **or** `unsloth/gemma-2-2b-it-bnb-4bit` | `.../nvidia_pid/` | Use `text_encoder_variant="unsloth-4bit"` for lower VRAM (requires `bitsandbytes`) |
| Caption Creator | `Qwen/Qwen3.5-0.8B` | `ComfyUI/models/text_encoders/nvidia_pid/qwen35_caption/` |
| Upscale VAEs | Flux/Z-Image, Flux2, SD3 VAE files | `ComfyUI/models/vae/nvidia_pid/` |

Use `model_precision=bf16` for best quality. `fp8` is available only for Flux1-family `2k/2kto4k` and Flux2-family `2k`; Flux2 `2kto4k`, SD3, SDXL, and Qwen-Image must use `bf16`.

## Nodes

| Node | Output | Purpose |
| --- | --- | --- |
| **PiD Decode** | `IMAGE` | One-node PiD decode from latent + caption + sigma. |
| **PiD Text Prompt** | `text`, `caption` | One prompt for normal text encoding and PiD caption input. |
| **PiD Caption Creator** | `text`, `caption` | Creates a caption from an input image with local Qwen. |
| **PiD Empty Latent Image** | `LATENT` | Backbone-aware empty latent with correct channels/downscale. |
| **PiD KSampler Capture** | `final_latent`, `pid_latent`, `pid_sigma` | KSampler-compatible sampler that captures the PiD latent and sigma. |
| **PiD Prepare** | `PID_PREP` | Moves/validates latent data and resolves PiD model assets. |
| **PiD Sample** | `PID_SAMPLES` | Runs native PiD sampling. |
| **PiD Finalize** | `IMAGE` | Converts PiD samples to a ComfyUI image. |
| **PiD Upscale** | `IMAGE` | Image-only tiled PiD upscaler with `2x/4x/6x/8x` output. |

Recommended PiD sampling: `pid_steps=4`, `cfg_scale=1.0`, `scale=0` or `4`.

## Supported Backbones

| Backbone value | PiD family | Checkpoints | Latent | PiD Upscale |
| --- | --- | --- | --- | --- |
| `zimage` | Flux1 | `2k`, `2kto4k` | 16ch / 8x | yes |
| `zimage-turbo` | Flux1 | `2k`, `2kto4k` | 16ch / 8x | yes |
| `flux` | Flux1 | `2k`, `2kto4k` | 16ch / 8x | yes |
| `flux2` | Flux2 | `2k`, `2kto4k` | 128ch / 16x | yes |
| `flux2-klein-4b` | Flux2 | `2k`, `2kto4k` | 128ch / 16x | yes |
| `flux2-klein-9b` | Flux2 | `2k`, `2kto4k` | 128ch / 16x | yes |
| `sd3` | SD3 | `2k`, `2kto4k` | 16ch / 8x | yes |
| `sdxl` | SDXL | `2kto4k` only | 4ch / 8x | no |
| `qwenimage` | Qwen-Image | `2kto4k` only | 16ch / 8x | no |
| `qwenimage-2512` | Qwen-Image | `2kto4k` only | 16ch / 8x | no |

`dinov2` and `siglip` are not supported by the native Comfy-Org PiD model set.

## Output Size Guide

Released PiD checkpoints use native `4x` scale.

| `pid_ckpt_type` | Base latent/image size | Final PiD output | Valid base presets |
| --- | --- | --- | --- |
| `2k` | 512-class | base × 4, e.g. `512x512 -> 2048x2048` | `512x512`, `576x432`, `432x576`, `624x416`, `416x624`, `672x384`, `384x672`, `784x336`, `336x784` |
| `2kto4k` | 1024-class | base × 4, e.g. `1024x1024 -> 4096x4096` | `1024x1024`, `1024x768`, `768x1024`, `1008x672`, `672x1008`, `1024x576`, `576x1024`, `1008x432`, `432x1008` |

Latent size depends on backbone downscale. Example: Flux2 `1024x1024` uses a `128 × 64 × 64` latent.

## PiD Upscale

`PiD Upscale` accepts `IMAGE` and returns `IMAGE`. It is separate from latent decode: the node cuts the image into tiles, encodes each tile with the matching VAE, runs native 4-step PiD, blends tiles, then resizes to the selected final factor.

| Setting | Values / behavior |
| --- | --- |
| `pid_ckpt_type` | `2k` uses 512px tiles; `2kto4k` uses 1024px tiles. |
| `backbone` | `zimage`, `zimage-turbo`, `flux`, `flux2`, `flux2-klein-4b`, `flux2-klein-9b`, `sd3`. |
| `model_precision` | Same limits as PiD decode; use `bf16` for best quality. |
| `upscale_factor` | Final output size: `2x`, `4x`, `6x`, or `8x`. |
| `strength` | PiD detail regeneration sigma, `0.0` to `1.0`; default `0.4`. |
| `caption` | Optional string input; connect `PiD Caption Creator` or `PiD Text Prompt`. |

| Profile | Tile size | Overlap | Small-image prepass |
| --- | ---: | ---: | ---: |
| `2k` | 512 | 64 | Resize long edge to 512, PiD once, then tiled upscale. |
| `2kto4k` | 1024 | 128 | Resize long edge to 1024, PiD once, then tiled upscale. |

Upscale VAEs are required because image tiles must be encoded into each backbone latent format:

| Backbone family | Accepted VAE names |
| --- | --- |
| Flux1 / Z-Image | `ae.safetensors` |
| Flux2 / Flux2-Klein | `flux2_ae.safetensors`, `flux2-vae.safetensors` |
| SD3 | `sd3_vae.safetensors`, `diffusion_pytorch_model.safetensors` |

Final upscale size is always based on the original input image: `width × factor`, `height × factor`. SDXL and Qwen-Image are not available in `PiD Upscale` because this implementation only maps image VAEs for Flux1/Z-Image, Flux2/Flux2-Klein, and SD3.

## Recommended Capture Settings

| Backbone | LDM steps | Capture step | Sampler / scheduler |
| --- | ---: | ---: | --- |
| `flux`, `sd3` | 28 | 24 | `euler` / `flowmatch_euler_discrete` |
| `sdxl` | 30 | 26 | `euler` / `normal` |
| `flux2` | 50 | 46 | `euler` / `flowmatch_euler_discrete` |
| `flux2-klein-4b`, `flux2-klein-9b` | 4 | 4 | `euler` / `flowmatch_euler_discrete` |
| `qwenimage`, `qwenimage-2512` | 50 | 44 | `euler` / `flowmatch_euler_discrete` |
| `zimage` | 50 | 46 | `euler` / `flowmatch_euler_discrete`, `flowmatch_shift=3.0` |
| `zimage-turbo` | 9 | 9 | `euler` / `flowmatch_euler_discrete`, `flowmatch_shift=3.0` |

## Main Workflows

### Text-to-image / generation

```text
PiD Text Prompt -> normal text encode + PiD caption
PiD Empty Latent Image -> model sampler
PiD KSampler Capture pid_latent + pid_sigma -> PiD Prepare
PiD Prepare -> PiD Sample -> PiD Finalize -> Save Image
```

### Direct decode

```text
LATENT + caption + sigma -> PiD Decode -> Save Image
```

### Image-to-image clean decode

```text
Load Image -> Resize -> VAE Encode -> PiD Prepare -> PiD Sample -> PiD Finalize -> Save Image
```

### Tiled upscale

```text
Load Image -> PiD Caption Creator -> PiD Upscale -> Save Image
```

## Example Workflows

Included in `example_workflows/`:

```text
pid_flux_complete.json
pid_flux2_complete.json
pid_flux2_klein_4b_complete.json
pid_flux2_klein_9b_complete.json
pid_qwenimage_complete.json
pid_qwenimage_2512_complete.json
pid_sd3_complete.json
pid_sdxl_complete.json
pid_zimage_complete.json
pid_zimage_turbo_complete.json
pid_image_to_image_2k_complete.json
pid_image_to_image_2kto4k_complete.json
pid_upscale_complete.json
```

## License

MIT
