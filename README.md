# ComfyUI-PiD

**ComfyUI custom nodes** for using NVIDIA **PiD** as a pixel diffusion decoder.

<img width="1058" height="604" alt="image" src="https://github.com/user-attachments/assets/cc5a9da3-94c6-4546-9574-c8387d5dffdb" />

<img width="4096" height="2048" alt="1111111111111111" src="https://github.com/user-attachments/assets/7ccd55ee-e571-4996-9c9c-4b5cecbb4418" />

PiD is not a normal ComfyUI `VAE`. It needs a latent, a prompt/caption, a sigma value, and optionally a native decoder baseline image:

```text
LATENT + caption + sigma + optional baseline IMAGE -> PiD -> IMAGE
```

For the official latent-conditioned PiD checkpoints, this node can infer the baseline size from the latent and skip the extra VAE/baseline image path to reduce VRAM use.

## Features

- Direct **PiD Decode** node that returns a ComfyUI `IMAGE`.
- Staged low-VRAM workflow: **PiD Prepare → PiD Sample → PiD Finalize**.
- **PiD Sample** runs in a subprocess so CUDA memory is released after sampling.
- **PiD KSampler Capture** for grabbing an intermediate latent and matching sigma.
- Lazy setup: PiD source, checkpoints, and required assets are prepared on first run when `auto_download=true`.
- Exact low-VRAM pixel chunking and sequential block offload for large outputs.

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
- Python `>=3.10`
- NVIDIA CUDA GPU
- Working ComfyUI install
- Enough VRAM for PiD, especially for `2kto4k` or large output scales

`requirements.txt` does not install PyTorch because ComfyUI usually provides it.

## Nodes

| Node | Purpose |
| --- | --- |
| **PiD Decode** | One-node PiD decode from latent to image. |
| **PiD Text Prompt** | One prompt box with `text` for CLIP and `caption` for PiD. |
| **PiD KSampler Capture** | KSampler-compatible sampler that returns final latent, captured PiD latent, and sigma. |
| **PiD Prepare** | Prepares latent, caption, checkpoint, assets, and metadata on CPU. |
| **PiD Sample** | Runs the heavy PiD sampling step in a subprocess. |
| **PiD Finalize** | Converts sampled PiD output back to ComfyUI `IMAGE`. |

## Supported backbones

| Value | Backbone | Latent channels | Checkpoints |
| --- | --- | ---: | --- |
| `zimage` | Z-Image / Flux-compatible | 16 | `2k`, `2kto4k` |
| `flux` | Flux | 16 | `2k`, `2kto4k` |
| `flux2` | Flux2 | 128 | `2k`, `2kto4k` |
| `sd3` | Stable Diffusion 3 | 16 | `2k`, `2kto4k` |
| `sdxl` | Stable Diffusion XL | 4 | `2kto4k` |
| `qwenimage` | Qwen-Image | 16 | `2kto4k` |
| `qwenimage-2512` | Qwen-Image-2512 | 16 | `2kto4k` |
| `dinov2` | DINOv2 RAE | 768 | `2k` |
| `siglip` | SigLIP Scale-RAE | 1152 | `2k` |

`scale=0` uses NVIDIA's default scale for the selected checkpoint: usually `4x`, or `8x` for SigLIP Scale-RAE.
SDXL and Qwen-Image only ship NVIDIA's `2kto4k` PiD checkpoint. Flux2 `2kto4k`
uses the newer `_2606` checkpoint, which replaces the older color-drifting
release.

## Basic workflow

For Z-Image / Flux-style workflows:

```text
PiD Text Prompt text    -> CLIP Text Encode
PiD Text Prompt caption -> PiD Decode caption
KSampler latent         -> PiD Decode latent
PiD Decode image        -> Save Image
```

Recommended first test settings:

```text
backbone = zimage
pid_ckpt_type = 2k
pid_steps = 4
scale = 1 or 2
cfg_scale = 1.0
sigma = 0.0
auto_download = true
unload_comfy_before_pid = true
aggressive_cleanup = true
sequential_offload = auto_low_vram
pid_weight_precision = fp32_compatible
pixel_chunk_patches = 0
```

For official latent-conditioned checkpoints, leave `vae` and `baseline_image` disconnected unless you specifically need an external baseline size.

## Lowest-VRAM staged workflow

Use the staged nodes when VRAM is tight:

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
scheduler = beta
capture_step = 46
```

`PiD Sample` runs in a separate Python process, so its CUDA context is destroyed after the sample is finished.

Use these PiD Sample settings for minimum VRAM:

```text
sequential_offload = auto_low_vram
pid_weight_precision = fp32_compatible
pixel_chunk_patches = 0
```

`auto_low_vram` chunks the large pixel-block AdaLN and MLP tensors while preserving
one global attention pass. It also keeps full-image positional data in system RAM.
At 4096x4096 output, the positional cache uses approximately 1 GiB of RAM.

Available offload policies:

| Value | Behavior |
| --- | --- |
| `auto_low_vram` | Default minimum-VRAM policy with automatic chunk sizing. |
| `disabled` | Legacy upstream behavior without PiD block offload or chunking. |
| `sequential_blocks` | Balanced exact-output block offload with chunked pixel work. |
| `sequential_blocks_aggressive` | Preserved for older workflows; now uses the improved low-VRAM policy. |

`pixel_chunk_patches=0` selects the chunk size automatically. A 16 GiB GPU uses
`4096` patches for a 4K output.

`pid_weight_precision=bf16_weights_experimental` casts PiD network weights after
load for additional savings. It is not the default because output changes. In the
4K synthetic comparison, mean absolute delta was `0.04556` and RMSE was `0.07943`
against `fp32_compatible`.

## Output size guide

```text
512x512 base  + 2k     + scale 4 -> 2048x2048
1024x1024 base + 2kto4k + scale 4 -> 4096x4096
```

Large outputs can require a lot of VRAM. If a run fails, try:
1. Lower `scale`.
2. Use a smaller base latent.
3. Keep cleanup options enabled.
4. Keep `sequential_offload=auto_low_vram` and `pixel_chunk_patches=0`.
5. Restart ComfyUI after CUDA allocator crashes.

## PiD source and weights

By default, the NVIDIA PiD source checkout lives under the custom node:

```text
ComfyUI/custom_nodes/ComfyUI-PiD/vendor/PiD
```

Downloaded weights and assets live in ComfyUI's shared models directory:

```text
ComfyUI/models/nvidia_pid/checkpoints
```

You can override the PiD source location with:
- the `pid_source_dir` node input
- `PID_REPO_DIR`
- `COMFYUI_PID_REPO_DIR`

These overrides affect the source checkout only. Weights and assets continue to
use `ComfyUI/models/nvidia_pid/checkpoints`.

When `auto_download=true`, the node downloads missing PiD source, checkpoints,
and assets as needed. Existing weights from older versions under
`vendor/PiD/checkpoints` are moved into the shared models directory on first use.

## Offline setup

PiD can run without internet after its source and Python dependencies are
installed and the required models are available locally. The common offline
layout is:

```text
ComfyUI/models/nvidia_pid/
  checkpoints/
    sdxl_vae.safetensors                         # sdxl backbone only
    QwenImage_VAE_2d.pth                         # qwenimage backbones only
  huggingface/
    Efficient-Large-Model/gemma-2-2b-it/
    facebook/dinov2-with-registers-base/        # dinov2 backbone only
    google/siglip2-so400m-patch14-224/          # siglip backbone only
```

The Gemma snapshot is required for every PiD decode. The SDXL/Qwen VAE files,
DINOv2 snapshot, and SigLIP snapshot are only required when their matching
backbones are selected.

To prepare an offline installation manually, clone NVIDIA's source while online
and download the required models:

```bash
git clone --depth 1 https://github.com/nv-tlabs/PiD.git ComfyUI/custom_nodes/ComfyUI-PiD/vendor/PiD
hf download nvidia/PiD --local-dir ComfyUI/models/nvidia_pid --include "checkpoints/*"
hf download Efficient-Large-Model/gemma-2-2b-it --local-dir ComfyUI/models/nvidia_pid/huggingface/Efficient-Large-Model/gemma-2-2b-it --exclude "gemma-2-2b-it.safetensors"
hf download facebook/dinov2-with-registers-base --local-dir ComfyUI/models/nvidia_pid/huggingface/facebook/dinov2-with-registers-base
hf download google/siglip2-so400m-patch14-224 --local-dir ComfyUI/models/nvidia_pid/huggingface/google/siglip2-so400m-patch14-224
```

The final two commands are optional unless you use their backbones.
`auto_download=false` enables strict local-only mode and reports any missing
files. With `auto_download=true`, existing complete local folders are used
without network calls; missing snapshots are downloaded lazily.

## Example workflow

A template workflow is included in:

```text
example_workflows/image_z_image_pid.json
```

After restart, open it from ComfyUI workflow templates or load the JSON manually.

## Notes

- This is a community wrapper around NVIDIA's public PiD code, not an official NVIDIA or ComfyUI project.
- PiD outputs `IMAGE`, not a ComfyUI `VAE`.
- NVIDIA's PiD weights may have separate license/usage terms. Check the model card before commercial use.
- Final latents with `sigma=0.0` can work, but captured intermediate latents usually better match the official PiD recipe.

## License

This project is released under the MIT License.
