# Changelog

## Precision update

- Added a `precision` setting to **PiD Decode** with `bf16`, `fp16`, `fp8_e4m3fn`, and `fp8_e5m2`.
- `bf16` remains the default. `fp16` is intended as a lower-precision / lower-VRAM option.
- Added clear error handling for unsupported or failed FP8 casting.

## GitHub/Registry ready build

- Added MIT `LICENSE`.
- Added `pyproject.toml` for Comfy Registry publishing under publisher `merserk`.
- Added `.comfyignore`.
- Added optional GitHub Actions publishing workflow.
- Moved the baseline-offload workflow to `example_workflows/image_z_image_pid_baseline_offload.json` so ComfyUI can show it in Workflow Templates.

## VRAM offload lean build

- Removed the separate automatic settings module from the package.
- Node registration now exposes only **PiD Decode** and **PiD Text Prompt**.
- Updated README to document manual **PiD Decode** settings only.
- Kept the VRAM offload behavior from the previous build: CPU staging before PiD, Comfy model unload before PiD, and PiD unload after decode.

## VRAM offload build

- Before loading PiD, `PiD Decode` now detaches the input latent and baseline image to CPU.
- `PiD Decode` clears Python references to upstream latent, VAE, and baseline tensors before ComfyUI model unload.
- `_free_cuda_memory()` now also runs Python garbage collection before Comfy/CUDA cache cleanup.
- After a successful PiD decode, the PiD model is removed from this node's private cache and moved off CUDA so the next Z-Image generation does not keep PiD in VRAM.
- The same PiD unload cleanup is attempted after PiD inference errors.
- Added an example workflow that sends `VAEDecode IMAGE` into `PiD Decode baseline_image` and leaves `PiD Decode vae` disconnected.
