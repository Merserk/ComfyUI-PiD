# ComfyUI-PiD

Experimental ComfyUI custom node for using NVIDIA **PiD** as a pixel diffusion decoder.

PiD is not a normal `.safetensors` VAE. NVIDIA's official path gives PiD a latent, the native decoder/VAE baseline image, a sigma value, and the prompt:

```text
LATENT + baseline IMAGE + prompt + sigma -> PiD -> IMAGE
```

The node can create the baseline through a connected ComfyUI `VAE`, or you can connect a pre-decoded `baseline_image` from another workflow component.

## Node

- **PiD Decode**: decodes a PiD-supported latent and outputs `IMAGE`.
- **PiD Auto Settings**: reads a latent shape and produces recommended decode settings for **PiD Decode**.
- **PiD Text Prompt**: one prompt box with outputs for both CLIP text encoding and PiD caption conditioning.

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

sampler LATENT
        -> PiD Auto Settings
        -> PiD Decode auto_settings

Optional visible wiring:
PiD Auto Settings backbone -> PiD Decode backbone_in
PiD Auto Settings pid_ckpt_type -> PiD Decode pid_ckpt_type_in
PiD Auto Settings pid_steps -> PiD Decode pid_steps_in
PiD Auto Settings scale -> PiD Decode scale_in
PiD Auto Settings cfg_scale -> PiD Decode cfg_scale_in
PiD Auto Settings sigma -> PiD Decode sigma_in

sampler LATENT
        + matching ComfyUI VAE
        + positive prompt text
        -> PiD Decode
        -> Save Image
```

For backbones where the matching VAE is not available as a ComfyUI `VAE`, connect a pre-decoded `baseline_image` instead.

Recommended first test settings:

```text
PiD Auto Settings:
backbone = auto
preset = balanced
base_width = 0
base_height = 0

PiD Decode:
connect auto_settings from PiD Auto Settings, or connect the individual settings outputs
connect caption from PiD Text Prompt
auto_download = true
```

`base_width=0` and `base_height=0` mean the settings node estimates the pixel size from the latent tensor. Set them manually only when your workflow uses unusual latent scaling.

Presets:

- `safe`: lower VRAM, smaller output.
- `balanced`: good first choice.
- `quality`: higher output when the latent size and checkpoint support it.
- `4k`: asks for the largest official scale; use on high-VRAM GPUs.

## Notes

1. This is a practical ComfyUI wrapper around NVIDIA's public PiD code, not an official NVIDIA or ComfyUI node.
2. This node outputs `IMAGE`, not a ComfyUI `VAE`, because PiD is a conditional pixel diffusion decoder.
3. NVIDIA's best generated-image demos use captured intermediate latents, for example Z-Image around step 46 of 50. A final ComfyUI latent with `sigma=0.0` can work, but it is not identical to the official capture recipe.
4. PiD currently expects CUDA and significant VRAM, especially with `2kto4k` and high output scales.
5. NVIDIA's PiD weights have their own license/terms. Check the Hugging Face model card before using them.

## Troubleshooting

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
- Your connected `VAE` can decode that latent, or `baseline_image` is the correct native baseline image.
- Start with `sigma=0.0`, `cfg_scale=1.0`, `pid_steps=4`, and `scale=1` or `2`.

### VRAM or cudaMallocAsync errors

For 16GB GPUs, start with one of these:

- 1024x1024 base latent + `pid_ckpt_type=2k` + `scale=1` or `scale=2`
- 512x512 base latent + `pid_ckpt_type=2k` + `scale=4`

Avoid 1024x1024 base latent + `pid_ckpt_type=2kto4k` + `scale=4` on 16GB cards. That asks PiD to generate roughly 4096x4096 inside ComfyUI and commonly triggers CUDA allocator/VRAM failures.

If a CUDA allocator internal assert occurs, restart ComfyUI before trying again. The CUDA process can remain unstable after that error.
