# Changelog

## 0.2.1

- Added **PiD Sample (Subprocess)**. This runs the heavy PiD sampling stage in a separate Python process so the main ComfyUI process does not need to hold the PiD CUDA context during sampling.
- The subprocess writes the PiD output tensor back to CPU, exits, and lets CUDA release the subprocess memory.
- Recommended for 4K experiments on 16GB GPUs when the in-process **PiD Sample** node spills into shared GPU memory or fails near the end.

## 0.2.0

- Added real staged PiD nodes: **PiD Prepare**, **PiD Sample**, **PiD Finalize**, and **PiD Decode (Staged)**.
- The staged flow keeps the baseline image and latent on CPU between stages, so ComfyUI can unload models and free VRAM between the preparation and PiD sampling phases.
- Added the `cleanup_after_prepare` option to aggressively free Comfy/Z-Image memory before the PiD sample stage.
- Added sequential offload support to the new PiD Sample stage.

## Sequential block offload build

- Based on the pre-FP8 GitHub-ready build. No FP8/float8 precision modes are included.
- Added optional `sequential_offload` setting to **PiD Decode**:
  - `disabled` keeps the previous behavior.
  - `sequential_blocks` CPU-offloads the detected largest PiD/DiT block stack and moves one block to CUDA only while it runs.
  - `sequential_blocks_aggressive` does the same and also clears CUDA cache after each block.
- The new setting is added after `pid_source_dir` to avoid shifting old required widget values as much as possible.

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
