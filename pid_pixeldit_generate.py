from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

try:
    import comfy.sample as comfy_sample
    import comfy.samplers as comfy_samplers
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils
except Exception:  # pragma: no cover - ComfyUI-only imports
    comfy_sample = None
    comfy_samplers = None
    comfy_sd = None
    comfy_utils = None

try:
    from .pid_decode import (
        PIXELDIT_TEXT_ENCODER_FILES,
        PiDNodeError,
        _encode_pixeldit_conditioning,
        _ensure_comfy_org_file,
        _free_cuda_memory,
        _load_native_pid_model,
        _load_pixeldit_clip,
        _log_cuda_peak_memory,
        _make_pid_progress_bar,
        _make_pixel_latent,
        _native_pixel_to_comfy_image,
        _reset_cuda_peak_memory_stats,
        _update_pid_progress_bar,
    )
except ImportError:  # pragma: no cover
    from pid_decode import (
        PIXELDIT_TEXT_ENCODER_FILES,
        PiDNodeError,
        _encode_pixeldit_conditioning,
        _ensure_comfy_org_file,
        _free_cuda_memory,
        _load_native_pid_model,
        _load_pixeldit_clip,
        _log_cuda_peak_memory,
        _make_pid_progress_bar,
        _make_pixel_latent,
        _native_pixel_to_comfy_image,
        _reset_cuda_peak_memory_stats,
        _update_pid_progress_bar,
    )


PIXELDIT_GENERATION_PRECISIONS = ["bf16", "fp8"]
PIXELDIT_GENERATION_FILES = {
    "bf16": "pixeldit_1300m_1024px_bf16.safetensors",
    "fp8": "pixeldit_1300m_1024px_mxfp8.safetensors",
}
PIXELDIT_NEGATIVE_PROMPT = "low quality, worst quality, over-saturated, blurry, deformed, watermark"

PIXELDIT_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "1024x1024 (1:1 Square)": (1024, 1024),
    "840x1256 (2:3 Portrait Photo)": (840, 1256),
    "1256x840 (3:2 Photo)": (1256, 840),
    "888x1184 (3:4 Portrait Standard)": (888, 1184),
    "1184x888 (4:3 Standard)": (1184, 888),
    "768x1368 (9:16 Portrait Widescreen)": (768, 1368),
    "1368x768 (16:9 Widescreen)": (1368, 768),
    "1568x672 (21:9 Ultrawide)": (1568, 672),
}

if comfy_samplers is not None:
    PIXELDIT_SAMPLERS = list(comfy_samplers.KSampler.SAMPLERS)
    PIXELDIT_SCHEDULERS = list(comfy_samplers.KSampler.SCHEDULERS)
else:  # Lets the module import for standalone unit tests.
    PIXELDIT_SAMPLERS = ["er_sde"]
    PIXELDIT_SCHEDULERS = ["simple"]


@dataclass(frozen=True)
class PixelDiTGenerationSpec:
    model_precision: str
    diffusion_filename: str
    text_encoder_filename: str


def _pixeldit_generation_spec(model_precision: str) -> PixelDiTGenerationSpec:
    precision = str(model_precision or "bf16").strip().lower()
    if precision not in PIXELDIT_GENERATION_PRECISIONS:
        raise PiDNodeError(
            f"Unknown PixelDiT model_precision={precision!r}; "
            f"expected one of {PIXELDIT_GENERATION_PRECISIONS}."
        )
    return PixelDiTGenerationSpec(
        model_precision=precision,
        diffusion_filename=PIXELDIT_GENERATION_FILES[precision],
        text_encoder_filename=PIXELDIT_TEXT_ENCODER_FILES[precision],
    )


def _pixeldit_resolution(resolution: str) -> Tuple[int, int]:
    try:
        width, height = PIXELDIT_RESOLUTIONS[str(resolution)]
    except KeyError as exc:
        raise PiDNodeError(
            f"Unknown PixelDiT resolution={resolution!r}; expected one of {list(PIXELDIT_RESOLUTIONS)}."
        ) from exc
    if width <= 0 or height <= 0:
        raise PiDNodeError(f"PixelDiT resolution must be positive, got {width}x{height}.")
    return int(width), int(height)


def _require_pixeldit_generation_support() -> None:
    if comfy_sample is None or comfy_samplers is None or comfy_sd is None or comfy_utils is None:
        raise PiDNodeError("PixelDiT Generate must run inside ComfyUI 0.28.0 or newer.")
    if not hasattr(comfy_sd.CLIPType, "PIXELDIT"):
        raise PiDNodeError("This ComfyUI build does not support PixelDiT. Update to ComfyUI 0.28.0 or newer.")
    try:
        from comfyui_version import __version__ as comfyui_version
    except Exception:
        return
    parts = tuple(int(value) for value in re.findall(r"\d+", str(comfyui_version))[:3])
    if parts and parts < (0, 28, 0):
        raise PiDNodeError(
            f"PixelDiT Generate requires ComfyUI 0.28.0 or newer; found {comfyui_version}."
        )


def _ensure_pixeldit_generation_assets(
    spec: PixelDiTGenerationSpec,
    allow_download: bool = True,
):
    diffusion_path = _ensure_comfy_org_file(
        "diffusion_models",
        "diffusion_models",
        spec.diffusion_filename,
        allow_download=allow_download,
    )
    text_encoder_path = _ensure_comfy_org_file(
        "text_encoders",
        "text_encoders",
        spec.text_encoder_filename,
        allow_download=allow_download,
    )
    return diffusion_path, text_encoder_path


def _format_pixeldit_generation_error(
    exc: BaseException,
    spec: PixelDiTGenerationSpec,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    scheduler: str,
) -> PiDNodeError:
    message = str(exc)
    context = (
        f"PixelDiT generation failed at {width}x{height} using {spec.diffusion_filename}, "
        f"steps={steps}, cfg={cfg_scale:g}, sampler={sampler_name}, scheduler={scheduler}. "
    )
    if "out of memory" in message.lower() or "cuda" in message.lower() or "cudamalloc" in message.lower():
        context += "This is usually VRAM pressure; try model_precision='fp8' and keep cleanup enabled. "
    return PiDNodeError(f"{context}Original error: {message}")


def _run_pixeldit_generation(
    spec: PixelDiTGenerationSpec,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    scheduler: str,
    seed: int,
    *,
    allow_download: bool = True,
    unload_comfy_before_generation: bool = True,
    aggressive_cleanup: bool = True,
) -> torch.Tensor:
    _require_pixeldit_generation_support()
    if sampler_name not in PIXELDIT_SAMPLERS:
        raise PiDNodeError(f"Unknown PixelDiT sampler={sampler_name!r}.")
    if scheduler not in PIXELDIT_SCHEDULERS:
        raise PiDNodeError(f"Unknown PixelDiT scheduler={scheduler!r}.")
    if int(steps) < 1:
        raise PiDNodeError("PixelDiT steps must be at least 1.")
    if float(cfg_scale) < 0.0:
        raise PiDNodeError("PixelDiT cfg_scale must be non-negative.")

    diffusion_path, text_encoder_path = _ensure_pixeldit_generation_assets(
        spec,
        allow_download=allow_download,
    )
    if unload_comfy_before_generation:
        _free_cuda_memory(aggressive=bool(aggressive_cleanup))

    clip = None
    model = None
    pixel_samples = None
    samples = None
    try:
        clip = _load_pixeldit_clip(text_encoder_path)
        positive = _encode_pixeldit_conditioning(clip, prompt or "")
        negative = _encode_pixeldit_conditioning(clip, negative_prompt or "")
        clip = None
        _free_cuda_memory(aggressive=True)

        model = _load_native_pid_model(diffusion_path)
        pixel_samples = _make_pixel_latent(1, (int(height), int(width)))["samples"]
        noise = comfy_sample.prepare_noise(pixel_samples, int(seed), None)
        pbar = _make_pid_progress_bar(int(steps))

        def sampler_callback(step, x0, x, total):
            del x0, x
            _update_pid_progress_bar(pbar, min(int(step) + 1, int(total)), int(total))

        disable_pbar = not bool(getattr(comfy_utils, "PROGRESS_BAR_ENABLED", True))
        print(
            "[ComfyUI-PiD] PixelDiT Generate: "
            f"model={spec.diffusion_filename}, precision={spec.model_precision}, "
            f"size={width}x{height}, steps={steps}, cfg={float(cfg_scale):g}, "
            f"sampler={sampler_name}, scheduler={scheduler}, seed={seed}",
            flush=True,
        )
        _reset_cuda_peak_memory_stats()
        with torch.inference_mode():
            samples = comfy_sample.sample(
                model,
                noise,
                int(steps),
                float(cfg_scale),
                str(sampler_name),
                str(scheduler),
                positive,
                negative,
                pixel_samples,
                denoise=1.0,
                force_full_denoise=True,
                callback=sampler_callback,
                disable_pbar=disable_pbar,
                seed=int(seed),
            )
        _update_pid_progress_bar(pbar, int(steps), int(steps))
        _log_cuda_peak_memory("PixelDiT Generate")
        return samples.detach()
    finally:
        clip = None
        model = None
        pixel_samples = None
        _free_cuda_memory(aggressive=bool(aggressive_cleanup))


class PixelDiTGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": (
                    "STRING",
                    {"default": PIXELDIT_NEGATIVE_PROMPT, "multiline": True, "dynamicPrompts": True},
                ),
                "model_precision": (PIXELDIT_GENERATION_PRECISIONS, {"default": "bf16"}),
                "resolution": (list(PIXELDIT_RESOLUTIONS), {"default": "1024x1024 (1:1 Square)"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (PIXELDIT_SAMPLERS, {"default": "er_sde"}),
                "scheduler": (PIXELDIT_SCHEDULERS, {"default": "simple"}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "auto_download": ("BOOLEAN", {"default": True}),
                "unload_comfy_before_generation": ("BOOLEAN", {"default": True}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "PiD/Generation"

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        model_precision: str,
        resolution: str,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        scheduler: str,
        seed: int,
        auto_download: bool,
        unload_comfy_before_generation: bool = True,
        aggressive_cleanup: bool = True,
    ):
        spec = _pixeldit_generation_spec(model_precision)
        width, height = _pixeldit_resolution(resolution)
        try:
            samples = _run_pixeldit_generation(
                spec,
                prompt,
                negative_prompt,
                width,
                height,
                int(steps),
                float(cfg_scale),
                sampler_name,
                scheduler,
                int(seed),
                allow_download=bool(auto_download),
                unload_comfy_before_generation=bool(unload_comfy_before_generation),
                aggressive_cleanup=bool(aggressive_cleanup),
            )
            image = _native_pixel_to_comfy_image(samples)
            del samples
            return (image,)
        except Exception as exc:
            _free_cuda_memory(aggressive=True)
            if isinstance(exc, PiDNodeError) and str(exc).startswith("PixelDiT generation failed"):
                raise
            raise _format_pixeldit_generation_error(
                exc,
                spec,
                width,
                height,
                int(steps),
                float(cfg_scale),
                sampler_name,
                scheduler,
            ) from exc


NODE_CLASS_MAPPINGS = {
    "PixelDiTGenerate": PixelDiTGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelDiTGenerate": "PixelDiT Generate",
}
