# ComfyUI-PiD

Experimental ComfyUI custom node for using NVIDIA **PiD** as a pixel diffusion decoder.

PiD is not a normal `.safetensors` VAE. NVIDIA's official path gives PiD a latent, the native decoder/VAE baseline image, a sigma value, and the prompt:

```text
LATENT + baseline IMAGE + prompt + sigma -> PiD -> IMAGE
```

The node can create the baseline through a connected ComfyUI `VAE`, or you can connect a pre-decoded `baseline_image` from another workflow component. For the current official latent-conditioned PiD checkpoints, the node can also infer the baseline size directly from the latent and skip the baseline image to reduce VRAM pressure.

## Node

- **PiD Decode**: decodes a PiD-supported latent and outputs `IMAGE`.
- **PiD Text Prompt**: one prompt box with outputs for both CLIP text encoding and PiD caption conditioning.
- **PiD Prepare / PiD Sample / PiD Finalize**: staged low-VRAM flow. **PiD Sample** runs in a subprocess so CUDA memory is released when sampling finishes.

There are no separate setup/download/unload nodes. PiD source, checkpoints, and required asset files are prepared lazily when **PiD Decode** runs with `auto_download=true`.

## Supported Backbones

The backbone list follows NVIDIA's official PiD checkpoint registry:

| Node value | Official backbone | Latent channels | Checkpoints |
| --- | --- | ---: | --- |
| `zimage` | Z-Image, reuses Flux weights | 16 | `2k`, `2kto4k` |
| `flux` | Flux | 16 | `2k`, `2kto4k` |
| `flux2` | Flux2 | 128 | `2k`, `2kto4k` |
| `sd3` | Stable Diffusion 3 | 16 | `2k`, `2kto4k` |
| `dinov2` | DINOv2 RAE | 768 | `2k` |
| `siglip` | SigLIP Scale-RAE | 1152 | `2k` |

`scale=0` means "use NVIDIA's default scale for that checkpoint": 4x for Flux, Flux2, SD3, Z-Image, and DINOv2; 8x for SigLIP Scale-RAE.

## Install From GitHub

Clone this custom node into ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Merserk/ComfyUI-PiD.git
cd ComfyUI-PiD
python -m pip install -r requirements.txt
```

Restart ComfyUI.

Do not download PiD model weights during install. The node downloads only what it needs the first time you run it.

## Workflow Template

This repository includes a ComfyUI template workflow at:

```text
example_workflows/image_z_image_pid_baseline_offload.json
```

After installing or cloning the node, restart ComfyUI and open **Workflow → Browse Workflow Templates**. The workflow should appear under the custom-node template category for this node. The template uses the lower-VRAM path:

```text
KSampler LATENT -> PiD Decode latent
PiD Decode vae and baseline_image disconnected
```

## Registry Metadata

This repository is ready for Comfy Registry publishing. It includes:

- `pyproject.toml` with `PublisherId = "merserk"` and `DisplayName = "ComfyUI-PiD"`
- MIT `LICENSE`
- `.comfyignore`
- optional GitHub Actions workflow at `.github/workflows/publish_action.yml`

Manual publish:

```bash
python -m pip install comfy-cli
comfy node publish
```

GitHub Actions publish: add a repository secret named `REGISTRY_ACCESS_TOKEN` containing your Comfy Registry API key, then run the workflow manually or bump `version` in `pyproject.toml` and push to `main`.

## Automatic Downloads

With `auto_download=true`, **PiD Decode** will:

1. Clone NVIDIA's PiD source into `vendor/PiD` if it is missing.
2. Download the selected PiD checkpoint from `nvidia/PiD`.
3. Download the extra tokenizer/VAE assets needed by the selected backbone.

For Flux/Z-Image, the node first tries to copy your existing:

```text
ComfyUI/models/vae/ae.safetensors
```

to:

```text
ComfyUI/custom_nodes/ComfyUI-PiD/vendor/PiD/checkpoints/ae.safetensors
```

If no local file is found, it downloads `checkpoints/ae.safetensors` from `nvidia/PiD`.

If Hugging Face requires a token, set one before starting ComfyUI:

```bash
export HF_TOKEN=hf_your_token_here
```

Windows PowerShell:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

## Basic Workflow

For Z-Image/Flux-style workflows:

```text
PiD Text Prompt text -> CLIP Text Encode
PiD Text Prompt caption -> PiD Decode caption

KSampler LATENT -> PiD Decode latent
PiD Decode image -> Save Image
```

For the lowest VRAM peak on the official Z-Image/Flux/SD3-style checkpoints, leave both optional `vae` and `baseline_image` disconnected. The node infers the base image size from the latent grid and avoids the extra VAE decode/baseline tensor. If you need an exact externally decoded baseline size, connect a pre-decoded `baseline_image` and leave the optional `vae` input disconnected.

Recommended first test settings on **PiD Decode**:

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
sequential_offload = disabled
```

For nonstandard or future image-conditioned PiD checkpoints where the matching VAE is not available as a ComfyUI `VAE`, connect a pre-decoded `baseline_image` instead.

### VRAM offload behavior

This version is optimized so **PiD Decode** keeps only CPU copies of the input latent and optional baseline image before the PiD stage. It then asks ComfyUI to unload the previously loaded Z-Image/CLIP/VAE models and clears CUDA cache before loading PiD. During PiD sampling it offloads NVIDIA's Gemma text encoder and unused PiD VAE encoder after caption encoding, and it lazily computes PiD LQ conditioning features per block instead of storing every block's feature tensor at once. After the PiD image is converted back to a ComfyUI `IMAGE`, the PiD model is also removed from this node's private cache and moved off CUDA.

For the lowest VRAM peak, prefer this workflow shape:

```text
KSampler LATENT -> PiD Decode latent
PiD Text Prompt caption -> PiD Decode caption

Leave PiD Decode vae and baseline_image disconnected unless you explicitly need a pre-decoded baseline.
```

The old direct `VAE -> PiD Decode vae` path still works, but it is no longer the lowest-VRAM path for the official latent-conditioned checkpoints.

### Sequential block offload

`PiD Decode` includes an optional `sequential_offload` setting:

```text
disabled                       fastest; previous behavior
sequential_blocks              lower VRAM; slower
sequential_blocks_aggressive   lowest VRAM attempt; slowest
```

This is a best-effort memory mode for the PiD/DiT stage. It detects the largest transformer/DiT block stack, moves those blocks to CPU before sampling, then moves each block to CUDA only for its own forward pass. This can reduce peak VRAM, but it will be much slower.

Recommended order when testing a borderline 4K run:

```text
1. disabled
2. sequential_blocks
3. sequential_blocks_aggressive
```

If ComfyUI reports that no block stack could be detected, set `sequential_offload=disabled`.

## Notes

1. This is a practical ComfyUI wrapper around NVIDIA's public PiD code, not an official NVIDIA or ComfyUI node.
2. This node outputs `IMAGE`, not a ComfyUI `VAE`, because PiD is a conditional pixel diffusion decoder.
3. NVIDIA's best generated-image demos use captured intermediate latents, for example Z-Image around step 46 of 50. A final ComfyUI latent with `sigma=0.0` can work, but it is not identical to the official capture recipe.
4. PiD currently expects CUDA and significant VRAM, especially with `2kto4k` and high output scales.
5. NVIDIA's PiD weights have their own license/terms. Check the Hugging Face model card before using them.


## Staged PiD workflow (real stage-based nodes)

This build adds four staged nodes:

```text
PiD Prepare
PiD Sample
PiD Finalize
PiD Decode (Staged)
```

Recommended staged wiring for lower VRAM:

```text
KSampler LATENT -> PiD Prepare latent
PiD Text Prompt -> PiD Prepare caption

PiD Prepare -> PiD Sample
PiD Sample -> PiD Finalize
PiD Finalize -> Save Image
```

What each stage does:

- **PiD Prepare**: resolves/checks PiD assets, decodes the baseline image only if connected/needed, and stores the latent plus optional baseline on CPU. If no baseline is connected, it infers the base image size from the latent. It can also run cleanup immediately after preparation.
- **PiD Sample**: saves the prepared CPU tensors to a temporary file, launches a separate Python process, runs PiD there with text/VAE offload and lazy LQ features, writes the output tensor back to CPU, then exits. When the subprocess exits, its CUDA context is destroyed.
- **PiD Finalize**: converts the CPU PiD tensor into a normal ComfyUI IMAGE output.
- **PiD Decode (Staged)**: convenience wrapper that runs the three stages internally.

This keeps the main ComfyUI process out of the heavy PiD sampling stage, which is stronger than normal `empty_cache()` cleanup.

Recommended first test:

```text
PiD Sample:
sequential_offload = disabled
aggressive_cleanup = true
```

If it still exceeds dedicated VRAM, try:

```text
sequential_offload = sequential_blocks
```

This can be much slower. If 1024 -> 4096 still fails after using **PiD Sample** with no connected baseline/VAE, the PiD block activations themselves are too large for the available dedicated VRAM and you should use 960 -> 3840, 896 -> 3584, or 512 -> 2048.


## Troubleshooting

### Widget values shift after switching browser tabs

This build uses ComfyUI's canonical combo syntax for `backbone` and `pid_ckpt_type`. If a workflow was already saved while the node was broken, delete and re-add the **PiD Decode** node once, or manually set the values again after installing this build.



### Template does not appear in ComfyUI

Make sure the workflow file is in `example_workflows/`, restart ComfyUI, and hard-refresh the browser.

### Missing dependencies

Install the custom node requirements and restart ComfyUI:

```bash
python -m pip install -r requirements.txt
```

### Missing checkpoint or asset

Set `auto_download=true` on **PiD Decode**. The node downloads the selected checkpoint and the required VAE/tokenizer assets on first run.

### Output looks wrong

Check that:

- `backbone` matches the latent you are feeding into PiD.
- The latent channel count matches the selected backbone.
- If you connect a `VAE` or `baseline_image`, it matches that latent. For the official latent-conditioned checkpoints, both can stay disconnected.
- Start with `sigma=0.0`, `cfg_scale=1.0`, `pid_steps=4`, and `scale=1` or `2`.

### VRAM or cudaMallocAsync errors

This build unloads ComfyUI models before PiD and unloads PiD again after the decode. If VRAM is still high, check the workflow first: leave both optional `vae` and `baseline_image` disconnected, and use the staged **PiD Sample** node. If the 4K path is still just over the limit, try `sequential_offload=sequential_blocks`, then `sequential_blocks_aggressive`.

For 16GB GPUs, start with one of these:

- 1024x1024 base latent + `pid_ckpt_type=2k` + `scale=1` or `scale=2`
- 512x512 base latent + `pid_ckpt_type=2k` + `scale=4`

Avoid 1024x1024 base latent + `pid_ckpt_type=2kto4k` + `scale=4` on 16GB cards. That asks PiD to generate roughly 4096x4096 inside ComfyUI and commonly triggers CUDA allocator/VRAM failures.

If a CUDA allocator internal assert occurs, restart ComfyUI before trying again. The CUDA process can remain unstable after that error.
