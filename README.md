# ComfyUI-PiD

Custom ComfyUI nodes for running NVIDIA PiD through ComfyUI's native PixelDiT/PiD model support.

<img width="1058" height="604" alt="ComfyUI-PiD workflow screenshot" src="https://github.com/user-attachments/assets/cc5a9da3-94c6-4546-9574-c8387d5dffdb" />

<img width="4096" height="2048" alt="ComfyUI-PiD example output" src="https://github.com/user-attachments/assets/7ccd55ee-e571-4996-9c9c-4b5cecbb4418" />

<img width="1361" height="562" alt="image" src="https://github.com/user-attachments/assets/ee881a77-3769-4ee1-bdbf-b65e97ba9773" />

This version uses the Comfy-Org repackaged PiD models from:

https://huggingface.co/Comfy-Org/PixelDiT

It no longer uses legacy NVIDIA checkpoint/source loading, Hydra configs, or the old custom model cache.

PiD is not a normal ComfyUI `VAE`. For the supported latent-conditioned PiD path, it needs a latent, a prompt/caption, and a sigma value:

```text
LATENT + caption + sigma -> native ComfyUI PiD -> IMAGE
```

Image-conditioning inputs are not part of the released latent-conditioned PiD path. Output size is inferred from the latent grid and selected PiD scale.

## Features

- Direct **PiD Decode** node that returns a ComfyUI `IMAGE`.
- Staged workflow: **PiD Prepare -> PiD Sample -> PiD Finalize**.
- **PiD Sample** runs in-process with ComfyUI-native PiD model loading and the NVIDIA-compatible distilled PiD student sampler.
- **PiD KSampler Capture** for grabbing an intermediate latent and matching sigma.
- Native ComfyUI model loading through `Comfy-Org/PixelDiT`.
- BF16/FP8 model precision selector with native Comfy-Org files.
- Auto-download into native ComfyUI model folders under `nvidia_pid` subfolders when `auto_download=true`.

## Install

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Merserk/ComfyUI-PiD.git
cd ComfyUI-PiD
python -m pip install -r requirements.txt
```

Restart ComfyUI.

Requirements:

- Recent ComfyUI with native PixelDiT/PiD support.
- Python `>=3.10`.
- NVIDIA CUDA GPU recommended.
- Enough VRAM for native PiD, especially for `2kto4k`.

`requirements.txt` does not install PyTorch because ComfyUI usually provides it.

## Required Models

The node downloads these from `Comfy-Org/PixelDiT` when needed, or you can place them manually. Use `model_precision=bf16` for the BF16 files or `model_precision=fp8` for the FP8 text encoder plus MXFP8 PiD diffusion files.

Text encoder:

```text
ComfyUI/models/text_encoders/nvidia_pid/gemma_2_2b_it_elm_bf16.safetensors
ComfyUI/models/text_encoders/nvidia_pid/gemma_2_2b_it_elm_fp8_scaled.safetensors
```

Diffusion models:

```text
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux1_512_to_2048_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux1_1024_to_4096_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux2_512_to_2048_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux2_1024_to_4096_4step_2606_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_sd3_512_to_2048_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_sd3_1024_to_4096_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_sdxl_1024_to_4096_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_qwenimage_1024_to_4096_4step_bf16.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux1_512_to_2048_4step_mxfp8.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux1_1024_to_4096_4step_mxfp8.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux2_512_to_2048_4step_mxfp8.safetensors
ComfyUI/models/diffusion_models/nvidia_pid/pid_flux2_1024_to_4096_4step_mxfp8.safetensors
```

Existing files in the root `models/diffusion_models` or `models/text_encoders` folders are still accepted. New downloads go into the `nvidia_pid` subfolders.

BF16 is the recommended quality default for PiD output. FP8/MXFP8 is available as a lower-VRAM option for Flux1 and Flux2, but it can show visible white speckle or high-frequency noise on some systems; switch `model_precision` back to `bf16` if that happens. SD3, SDXL, and QwenImage must use `model_precision=bf16`.

Native PiD uses the NVIDIA distilled 4-step student SDE schedule by default (`[0.999, 0.866, 0.634, 0.342, 0.0]`) while still loading the Comfy-Org `.safetensors` models through ComfyUI. This avoids the over-sharp grain that can appear when the distilled PiD checkpoint is driven through a generic Euler/simple sampler.

For compatibility with the original NVIDIA PiD nodes, captured low-quality latents are passed to PiD directly as `LQ_latent`. The normal PiD Decode and staged PiD Sample path use the original raw latent conditioning.

## Nodes

| Node | Purpose |
| --- | --- |
| **PiD Decode** | One-node native PiD decode from latent to image. |
| **PiD Text Prompt** | One prompt box with `text` for CLIP and `caption` for PiD. |
| **PiD KSampler Capture** | KSampler-compatible sampler that returns final latent, captured PiD latent, and sigma. |
| **PiD Prepare** | Prepares latent, caption, native model paths, and metadata on CPU. |
| **PiD Sample** | Runs native PiD sampling in-process from prepared CPU latent data. |
| **PiD Finalize** | Converts native PiD pixel output to ComfyUI `IMAGE`. |
| **PiD Empty Latent Image** | Creates PiD-friendly SD3-style empty latents from base-resolution presets. |
| **PiD Upscale** | Image-only tiled PiD upscaler with 2x/4x/6x/8x output factors. |

## Supported Backbones

| Value | Native PiD file family | Latent channels | Checkpoints |
| --- | --- | ---: | --- |
| `zimage` | Flux1 PiD | 16 | `2k`, `2kto4k` |
| `zimage-turbo` | Flux1 PiD | 16 | `2k`, `2kto4k` |
| `flux` | Flux1 PiD | 16 | `2k`, `2kto4k` |
| `flux2` | Flux2 PiD | 128 | `2k`, `2kto4k` |
| `flux2-klein-4b` | Flux2 PiD | 128 | `2k`, `2kto4k` |
| `flux2-klein-9b` | Flux2 PiD | 128 | `2k`, `2kto4k` |
| `sd3` | SD3 PiD | 16 | `2k`, `2kto4k` |
| `sdxl` | SDXL PiD | 4 | `2kto4k` |
| `qwenimage` | QwenImage PiD | 16 | `2kto4k` |
| `qwenimage-2512` | QwenImage PiD | 16 | `2kto4k` |

`dinov2` and `siglip` are no longer supported because Comfy-Org does not provide native PiD files for those backbones.

`scale=0` uses the native checkpoint scale, currently `4x` for all supported models. SDXL and QwenImage only use the `2kto4k` PiD checkpoint in the native model set.

BF16 Flux2 `2kto4k` uses the newer `pid_flux2_1024_to_4096_4step_2606_bf16.safetensors` file. FP8 Flux2 `2kto4k` uses `pid_flux2_1024_to_4096_4step_mxfp8.safetensors`.

All currently released native PiD checkpoints are 4-step distilled. The UI may still allow other `pid_steps` and manual `scale` values for testing or low-VRAM debugging, but the recommended default is `pid_steps=4` and the checkpoint's native scale.

## Staged Workflow

Use staged nodes when you want to separate latent preparation, native sampling, and final image conversion:

```text
PiD KSampler Capture pid_latent -> PiD Prepare latent
PiD Text Prompt caption         -> PiD Prepare caption
PiD Prepare                     -> PiD Sample
PiD Sample                      -> PiD Finalize
PiD Finalize image              -> Save Image
```

Recommended Z-Image capture settings:

```text
steps = 50
sampler_name = euler
scheduler = flowmatch_euler_discrete
flowmatch_shift = 3.0
capture_step = 46
```

`flowmatch_euler_discrete` is the PiD node's built-in Diffusers-style FlowMatch Euler Discrete sigma schedule. Z-Image and Z-Image-Turbo use a FlowMatch Euler Discrete schedule with `shift=3.0`. `euler + beta` remains available as a community/experimental ComfyUI setting, but it is not the exact Diffusers-style schedule.

`PiD KSampler Capture` counts `capture_step` like NVIDIA's `save_xt_steps`: the number of completed LDM denoising passes. For example, `capture_step=46` captures the latent at `sigmas[46]`. If `capture_step` is equal to or greater than the effective sampler step count, the node returns the final clean latent with `sigma=0`.

Suggested capture settings by backbone:

| Backbone | LDM steps | Suggested capture | Recommended latent | Suggested KSampler combo |
| --- | ---: | --- | --- | --- |
| `flux`, `sd3` | 28 | 24 | captured latent | `euler` + `flowmatch_euler_discrete` |
| `sdxl` | 30 | 26 | captured latent | `euler` + Comfy SDXL scheduler (`normal`/model default) |
| `flux2` | 50 | 46 | captured latent | `euler` + `flowmatch_euler_discrete` |
| `flux2-klein-4b`, `flux2-klein-9b` | 4 | 2 or 3 | final `x0` | `euler` + `flowmatch_euler_discrete` |
| `qwenimage`, `qwenimage-2512` | 50 | 44 | captured latent | `euler` + `flowmatch_euler_discrete` |
| `zimage` | 50 | 46 | captured latent | `euler` + `flowmatch_euler_discrete`, `flowmatch_shift=3.0` |
| `zimage-turbo` | 9 | 7 optional | final `x0` | `euler` + `flowmatch_euler_discrete`, `flowmatch_shift=3.0` |

Recommended capture settings are workflow-dependent. Existing Z-Image, Flux2, SD3, SDXL, and Qwen capture workflows should continue to use the same capture node and sigma path.

For Qwen-Image workflows, keep the diffusion model `UNETLoader` `weight_dtype` set to `default`. ComfyUI's `fp8_e4m3fn_fast` path can produce speckled Qwen latents before PiD runs; `PiD KSampler Capture` rejects that combination with a clear error.

`PiD Prepare` accepts only the PiD latent and caption as graph inputs; sigma comes from the captured latent when available or from the manual sigma widget.

## Output Size Guide

```text
512x512 base   + 2k     + scale 4 -> 2048x2048
1024x1024 base + 2kto4k + scale 4 -> 4096x4096
```

`PiD Upscale` always uses the native 2K/512px PiD checkpoint at its trained 4x scale. It tiles normal images into 512px tiles, stitches the 4x PiD tiles, then Lanczos-resizes to the selected output factor. Images below 512px on their longest edge are first resized near 512px, PiD-upscaled once, then tiled and PiD-upscaled again before the final Lanczos resize.

The upscaler exposes only one graph input (`IMAGE`) and one graph output (`IMAGE`). Backbone, auto-download, precision, output factor, and strength are widgets. Strength maps to PiD detail regeneration sigma: `off=0.0`, `low=0.2`, `medium=0.4`, `high=0.6`. SDXL and Qwen backbones are intentionally excluded because the native model set only provides them as `2kto4k` checkpoints.

Large outputs can require a lot of VRAM. If a run fails, try:

1. Use a smaller base latent.
2. Keep cleanup options enabled.
3. Use the staged workflow to free memory between prepare/sample/finalize.
4. Try FP8/MXFP8 where supported.
5. Restart ComfyUI after CUDA allocator crashes.

## Offline Setup

For offline use, download the needed files while online:

```bash
hf download Comfy-Org/PixelDiT --local-dir ComfyUI/models --include "diffusion_models/pid_*_bf16.safetensors" "diffusion_models/pid_*_mxfp8.safetensors" "text_encoders/gemma_2_2b_it_elm_bf16.safetensors" "text_encoders/gemma_2_2b_it_elm_fp8_scaled.safetensors"
```

Place the downloaded diffusion files under `models/diffusion_models/nvidia_pid` and the text encoder under `models/text_encoders/nvidia_pid`, then set `auto_download=false`.

## Notes

- This is a community wrapper around NVIDIA PiD support for ComfyUI, not an official NVIDIA or ComfyUI project.
- PiD outputs `IMAGE`, not a ComfyUI `VAE`; no separate image-conditioning input is required for the supported latent-conditioned checkpoints.
- NVIDIA's PiD weights may have separate license/usage terms. Check the model card before commercial use.
- Final latents with `sigma=0.0` can work. For Z-Image-Turbo and Flux2-Klein, final `x0` is the recommended latent; for normal Z-Image / Flux2 / Qwen-Image, captured intermediate latents usually better match the PiD recipe.
- SDXL captured latents are automatically converted from Comfy/k-diffusion's variance-exploding frame to the VP frame expected by PiD.
- Root-level `models/diffusion_models` and `models/text_encoders` files are accepted for compatibility, but new downloads are placed in the `nvidia_pid` subfolders.

## PiD Empty Latent Image

`PiD Empty Latent Image` is a lightweight preset wrapper for PiD-friendly base resolutions.
It intentionally mirrors `EmptySD3LatentImage` output and creates SD3-style empty latents with shape:

```text
[batch, 16, height / 8, width / 8]
```

It does **not** expose a backbone selector. The node only provides the `2k` / `2kto4k` switch, a dynamic resolution preset list, and `batch_size`. Backbone selection stays in the PiD processing nodes.

## Example Workflows

Complete backbone-specific workflows are included in `example_workflows/`. They include the generation side plus PiD decode side: model loader, text encoder loader, prompt encode, empty latent, `PiD KSampler Capture`, `PiD Prepare`, `PiD Sample`, `PiD Finalize`, and `Save Image`.

Included complete workflows:

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
```

These workflows output PiD images directly; no VAE baseline image is used.

Additional clean-image workflows are also included:

- `pid_image_to_image_2k_complete.json`
- `pid_image_to_image_2kto4k_complete.json`

These use `LoadImage -> ResizeImagesByLongerEdge -> VAEEncode -> PiDPrepare -> PiDSample -> PiDFinalize`.

## License

This project is released under the MIT License.
