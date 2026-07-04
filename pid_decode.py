"""ComfyUI-PiD native Comfy-Org backend.

The public custom nodes are kept compatible with older workflows, but model
loading now uses ComfyUI's native PixelDiT/PiD implementation and Comfy-Org
`.safetensors` files instead of legacy checkpoint/source code loading.
"""

from __future__ import annotations

import gc
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

try:
    import comfy.model_management as model_management
    import comfy.sample as comfy_sample
    import comfy.samplers as comfy_samplers
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils
    import folder_paths
    import node_helpers
except Exception:  # pragma: no cover - ComfyUI-only imports
    model_management = None
    comfy_sample = None
    comfy_samplers = None
    comfy_sd = None
    comfy_utils = None
    folder_paths = None
    node_helpers = None


COMFY_ORG_REPO_ID = "Comfy-Org/PixelDiT"
PIXELDIT_TEXT_ENCODER_FILES = {
    "bf16": "gemma_2_2b_it_elm_bf16.safetensors",
    "fp8": "gemma_2_2b_it_elm_fp8_scaled.safetensors",
}
# Optional alternate: Unsloth 4-bit Gemma (lower VRAM)
UNSLOTH_GEMMA_REPO_ID = "unsloth/gemma-2-2b-it-bnb-4bit"
UNSLOTH_GEMMA_FILENAME = "model.safetensors"
TEXT_ENCODER_VARIANTS = ["default", "unsloth-4bit"]
NATIVE_PID_SUBFOLDER = "nvidia_pid"
PID_CKPT_TYPES = ["2k", "2kto4k"]
MODEL_PRECISION_CHOICES = ["bf16", "fp8"]
NATIVE_PID_STUDENT_T_LIST = (0.999, 0.866, 0.634, 0.342, 0.0)

TORCH_COMPILE_MODES = ["disabled", "default", "max-autotune"]

# These are the base/LDM image sizes that PiD will decode from.
# Final PiD output is normally 4x these dimensions for the released LDM checkpoints.
PID_BASE_RESOLUTIONS: Dict[str, Tuple[Tuple[str, int, int], ...]] = {
    "2k": (
        ("512x512 (1:1)", 512, 512),
        ("576x432 (4:3)", 576, 432),
        ("432x576 (3:4)", 432, 576),
        ("624x416 (3:2)", 624, 416),
        ("416x624 (2:3)", 416, 624),
        ("672x384 (16:9)", 672, 384),
        ("384x672 (9:16)", 384, 672),
        ("784x336 (21:9)", 784, 336),
        ("336x784 (9:21)", 336, 784),
    ),
    "2kto4k": (
        ("1024x1024 (1:1)", 1024, 1024),
        ("1024x768 (4:3)", 1024, 768),
        ("768x1024 (3:4)", 768, 1024),
        ("1008x672 (3:2)", 1008, 672),
        ("672x1008 (2:3)", 672, 1008),
        ("1024x576 (16:9)", 1024, 576),
        ("576x1024 (9:16)", 576, 1024),
        ("1008x432 (21:9)", 1008, 432),
        ("432x1008 (9:21)", 432, 1008),
    ),
}

class PiDNodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class PiDBackbone:
    label: str
    registry_key: str
    latent_format: str
    latent_channels: int
    default_scale: int
    latent_downscale: int
    ckpt_types: Tuple[str, ...]


@dataclass(frozen=True)
class NativePiDModelSpec:
    backbone: str
    label: str
    registry_key: str
    ckpt_type: str
    model_precision: str
    diffusion_filename: str
    text_encoder_filename: str
    latent_format: str
    latent_channels: int
    latent_downscale: int
    scale: int

    @property
    def experiment(self) -> str:
        return self.diffusion_filename

    @property
    def relpath(self) -> str:
        return f"diffusion_models/{self.diffusion_filename}"


PID_BACKBONES: Dict[str, PiDBackbone] = {
    "zimage": PiDBackbone("Z-Image", "flux1", "flux", 16, 4, 8, ("2k", "2kto4k")),
    "zimage-turbo": PiDBackbone("Z-Image-Turbo", "flux1", "flux", 16, 4, 8, ("2k", "2kto4k")),
    "flux": PiDBackbone("Flux", "flux1", "flux", 16, 4, 8, ("2k", "2kto4k")),
    "flux2": PiDBackbone("Flux2", "flux2", "flux", 128, 4, 16, ("2k", "2kto4k")),
    "flux2-klein-4b": PiDBackbone("Flux2-Klein-4B", "flux2", "flux", 128, 4, 16, ("2k", "2kto4k")),
    "flux2-klein-9b": PiDBackbone("Flux2-Klein-9B", "flux2", "flux", 128, 4, 16, ("2k", "2kto4k")),
    "sd3": PiDBackbone("SD3", "sd3", "sd3", 16, 4, 8, ("2k", "2kto4k")),
    "sdxl": PiDBackbone("SDXL", "sdxl", "sdxl", 4, 4, 8, ("2kto4k",)),
    "qwenimage": PiDBackbone("Qwen-Image", "qwenimage", "qwenimage", 16, 4, 8, ("2kto4k",)),
    "qwenimage-2512": PiDBackbone("Qwen-Image-2512", "qwenimage", "qwenimage", 16, 4, 8, ("2kto4k",)),
}

PID_NATIVE_FILES: Dict[Tuple[str, str, str], str] = {
    ("bf16", "flux1", "2k"): "pid_flux1_512_to_2048_4step_bf16.safetensors",
    ("bf16", "flux1", "2kto4k"): "pid_flux1_1024_to_4096_4step_bf16.safetensors",
    ("bf16", "flux2", "2k"): "pid_flux2_512_to_2048_4step_bf16.safetensors",
    ("bf16", "flux2", "2kto4k"): "pid_flux2_1024_to_4096_4step_2606_bf16.safetensors",
    ("bf16", "sd3", "2k"): "pid_sd3_512_to_2048_4step_bf16.safetensors",
    ("bf16", "sd3", "2kto4k"): "pid_sd3_1024_to_4096_4step_bf16.safetensors",
    ("bf16", "sdxl", "2kto4k"): "pid_sdxl_1024_to_4096_4step_bf16.safetensors",
    ("bf16", "qwenimage", "2kto4k"): "pid_qwenimage_1024_to_4096_4step_bf16.safetensors",
    ("fp8", "flux1", "2k"): "pid_flux1_512_to_2048_4step_mxfp8.safetensors",
    ("fp8", "flux1", "2kto4k"): "pid_flux1_1024_to_4096_4step_mxfp8.safetensors",
    ("fp8", "flux2", "2k"): "pid_flux2_512_to_2048_4step_mxfp8.safetensors",
    ("fp8", "flux2", "2kto4k"): "pid_flux2_1024_to_4096_4step_mxfp8.safetensors",
}

BACKBONE_CHOICES = list(PID_BACKBONES.keys())


def _pid_base_resolution_labels(ckpt_type: str) -> List[str]:
    return [label for label, _width, _height in PID_BASE_RESOLUTIONS[str(ckpt_type)]]


def _pid_base_resolution_size_map(ckpt_type: str) -> Dict[str, Tuple[int, int]]:
    return {
        label: (int(width), int(height))
        for label, width, height in PID_BASE_RESOLUTIONS[str(ckpt_type)]
    }


def _pid_valid_base_sizes(ckpt_type: str) -> Tuple[Tuple[int, int], ...]:
    return tuple((int(width), int(height)) for _label, width, height in PID_BASE_RESOLUTIONS[str(ckpt_type)])


def _format_pid_valid_base_sizes(ckpt_type: str) -> str:
    return ", ".join(f"{width}x{height}" for width, height in _pid_valid_base_sizes(ckpt_type))


def _pid_size_class_hint(ckpt_type: str) -> str:
    if str(ckpt_type) == "2k":
        return "512-class base latents such as 512x512"
    if str(ckpt_type) == "2kto4k":
        return "1024-class base latents such as 1024x1024"
    return "a matching trained base latent size"


def _require_comfy() -> None:
    if comfy_sd is None or comfy_sample is None or folder_paths is None or node_helpers is None:
        raise PiDNodeError(
            "Native PiD must run inside ComfyUI, or with the ComfyUI root added to PYTHONPATH."
        )


def _free_cuda_memory(aggressive: bool = False) -> None:
    try:
        gc.collect()
    except Exception:
        pass
    if model_management is not None:
        if aggressive:
            for name in ("unload_all_models", "unload_model_clones"):
                fn = getattr(model_management, name, None)
                if callable(fn):
                    try:
                        fn()
                    except TypeError:
                        try:
                            fn(None)
                        except Exception:
                            pass
                    except Exception:
                        pass
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if aggressive:
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _vram_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return float(props.total_memory) / (1024 ** 3)
    except Exception:
        return 0.0


def _reset_cuda_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _log_cuda_peak_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        allocated_mib = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
        reserved_mib = float(torch.cuda.max_memory_reserved()) / (1024 ** 2)
        print(
            f"[ComfyUI-PiD] {label}: peak allocated={allocated_mib:.1f} MiB, "
            f"peak reserved={reserved_mib:.1f} MiB",
            flush=True,
        )
    except Exception:
        pass


def _log_native_decode_plan(infer_image_size: Tuple[int, int]) -> None:
    h, w = int(infer_image_size[0]), int(infer_image_size[1])
    print(
        "[ComfyUI-PiD] native ComfyUI backend: "
        f"image={w}x{h}, gpu_capacity={_vram_total_gb():.1f} GiB",
        flush=True,
    )


def _log_pid_decode_plan(
    spec: NativePiDModelSpec,
    latent_shape: Tuple[int, ...],
    base_size: Tuple[int, int],
    infer_image_size: Tuple[int, int],
    sigma: float,
) -> None:
    base_h, base_w = int(base_size[0]), int(base_size[1])
    out_h, out_w = int(infer_image_size[0]), int(infer_image_size[1])
    print(
        "[ComfyUI-PiD] native decode plan: "
        f"backbone={spec.backbone}, ckpt={spec.ckpt_type}, precision={spec.model_precision}, "
        f"model={spec.diffusion_filename}, latent_shape={list(latent_shape)}, "
        f"base={base_w}x{base_h}, output={out_w}x{out_h}, sigma={float(sigma):.6f}, "
        f"gpu_capacity={_vram_total_gb():.1f} GiB",
        flush=True,
    )


def _normalize_model_precision(model_precision: str) -> str:
    if isinstance(model_precision, bool):
        return "bf16"
    precision = str(model_precision or "bf16").strip().lower()
    if precision not in MODEL_PRECISION_CHOICES:
        raise PiDNodeError(f"Unknown model_precision={precision!r}; expected one of {MODEL_PRECISION_CHOICES}")
    return precision


def _checkpoint_for(backbone: str, ckpt_type: str, model_precision: str = "bf16") -> NativePiDModelSpec:
    backbone = str(backbone).strip()
    ckpt_type = str(ckpt_type).strip()
    model_precision = _normalize_model_precision(model_precision)
    if backbone not in PID_BACKBONES:
        raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
    info = PID_BACKBONES[backbone]
    if ckpt_type not in info.ckpt_types:
        supported = ", ".join(info.ckpt_types)
        raise PiDNodeError(
            f"{info.label} does not have a native Comfy-Org {ckpt_type!r} PiD model. "
            f"Supported pid_ckpt_type values: {supported}."
        )
    if model_precision == "fp8" and info.registry_key == "flux2" and ckpt_type == "2kto4k":
        raise PiDNodeError(
            "Flux2 / Flux2-Klein PiD 2kto4k must use model_precision='bf16'. "
            "The FP8/MXFP8 Comfy-Org Flux2 4K file maps to the older pre-_2606 checkpoint "
            "family that NVIDIA replaced because it can produce color drift/artifacts."
        )
    try:
        filename = PID_NATIVE_FILES[(model_precision, info.registry_key, ckpt_type)]
    except KeyError as exc:
        if model_precision == "fp8":
            raise PiDNodeError(
                f"Comfy-Org does not provide an FP8/mxfp8 PiD diffusion model for "
                f"backbone={backbone!r}, pid_ckpt_type={ckpt_type!r}. "
                "Use model_precision='bf16' for this combination."
            ) from exc
        raise PiDNodeError(
            f"No native Comfy-Org PiD model registered for backbone={backbone!r}, "
            f"pid_ckpt_type={ckpt_type!r}, model_precision={model_precision!r}."
        ) from exc
    if model_precision == "fp8" and info.registry_key == "flux1":
        warnings.warn(
            "PiD model_precision='fp8' uses MXFP8 Flux1/Z-Image PiD files and may produce "
            "visible white speckle on some systems. Use model_precision='bf16' for best quality.",
            RuntimeWarning,
            stacklevel=2,
        )
    return NativePiDModelSpec(
        backbone=backbone,
        label=info.label,
        registry_key=info.registry_key,
        ckpt_type=ckpt_type,
        model_precision=model_precision,
        diffusion_filename=filename,
        text_encoder_filename=PIXELDIT_TEXT_ENCODER_FILES[model_precision],
        latent_format=info.latent_format,
        latent_channels=info.latent_channels,
        latent_downscale=info.latent_downscale,
        scale=info.default_scale,
    )


def _normalize_scale_for_checkpoint(backbone: str, ckpt: NativePiDModelSpec, scale: int) -> int:
    if int(scale) <= 0:
        return int(ckpt.scale)
    scale = int(scale)
    if scale != int(ckpt.scale):
        print(
            f"[ComfyUI-PiD] warning: {ckpt.label} {ckpt.diffusion_filename} was trained for "
            f"scale={ckpt.scale}; using manual scale={scale}.",
            flush=True,
        )
    return scale


def _warn_if_non_distilled_step_count(pid_steps: int) -> None:
    try:
        steps = int(pid_steps)
    except Exception:
        return
    if steps != 4:
        print(
            "[ComfyUI-PiD] warning: Comfy-Org PiD models are 4-step distilled; "
            f"pid_steps={steps} is experimental/out-of-distribution.",
            flush=True,
        )


def _latent_pid_source_backbone(latent: dict) -> Optional[str]:
    if not isinstance(latent, dict):
        return None
    source = latent.get("pid_source_backbone")
    if source is None:
        return None
    source = str(source).strip().lower()
    return source or None


def _compatible_source_targets(source: str) -> Tuple[str, ...]:
    if source in ("zimage", "zimage-turbo", "flux"):
        return ("zimage", "zimage-turbo", "flux")
    if source in ("flux2", "flux2-klein-4b", "flux2-klein-9b"):
        return ("flux2", "flux2-klein-4b", "flux2-klein-9b")
    if source in ("qwenimage", "qwenimage-2512"):
        return ("qwenimage", "qwenimage-2512")
    return (source,)


def _validate_latent_source_backbone(latent: dict, backbone: str) -> None:
    source = _latent_pid_source_backbone(latent)
    if source is None:
        return
    target = str(backbone).strip()
    if target not in _compatible_source_targets(source):
        raise PiDNodeError(
            f"This latent was captured from {source!r}, but PiD is set to backbone={target!r}. "
            "Select a matching PiD backbone for this latent."
        )


def _sdxl_vp_from_ve_latent(samples: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, float]:
    sigma_f = float(sigma)
    if abs(sigma_f) < 1e-12:
        return samples, sigma_f
    denom = float((sigma_f * sigma_f + 1.0) ** 0.5)
    return samples / denom, sigma_f / denom


def _prepare_latent_for_pid_backbone(samples: torch.Tensor, sigma: float, backbone: str) -> Tuple[torch.Tensor, float]:
    if str(backbone).strip() == "sdxl":
        return _sdxl_vp_from_ve_latent(samples, sigma)
    return samples, float(sigma)


def _latent_samples(latent: dict) -> torch.Tensor:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise PiDNodeError("Expected a ComfyUI LATENT dict containing key 'samples'.")
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]
    if samples.ndim == 5:
        if samples.shape[2] == 1:
            samples = samples[:, :, 0, :, :]
        elif samples.shape[1] == 1:
            samples = samples[:, 0, :, :, :]
        else:
            raise PiDNodeError(
                "Expected latent samples as [B,C,H,W], [B,C,1,H,W], or [B,1,C,H,W], "
                f"got shape {list(samples.shape)}"
            )
    if samples.ndim != 4:
        raise PiDNodeError(f"Expected latent samples as [B,C,H,W], got shape {list(samples.shape)}")
    return samples


def _latent_pid_sigma(latent: dict, fallback: float) -> float:
    if isinstance(latent, dict) and "pid_sigma" in latent and abs(float(fallback)) < 1e-12:
        try:
            return float(latent["pid_sigma"])
        except Exception:
            pass
    return float(fallback)


def _infer_lq_size_from_latent(samples: torch.Tensor, backbone: str) -> Tuple[int, int]:
    info = PID_BACKBONES.get(str(backbone).strip(), PID_BACKBONES["zimage"])
    return int(samples.shape[-2]) * int(info.latent_downscale), int(samples.shape[-1]) * int(info.latent_downscale)


def _validate_pid_base_resolution(
    spec: NativePiDModelSpec,
    base_size: Tuple[int, int],
) -> None:
    base_h, base_w = int(base_size[0]), int(base_size[1])
    valid_sizes = set(_pid_valid_base_sizes(spec.ckpt_type))
    if (base_w, base_h) in valid_sizes:
        return
    raise PiDNodeError(
        f"{spec.label} supports pid_ckpt_type={spec.ckpt_type!r}, but that checkpoint must be paired with "
        f"{_pid_size_class_hint(spec.ckpt_type)}. Got a {base_w}x{base_h} base/LDM latent. "
        f"Valid {spec.ckpt_type!r} base sizes are: {_format_pid_valid_base_sizes(spec.ckpt_type)}. "
        "For Flux2-Klein 2K output, set PiD Empty Latent to '2k' / '512x512 (1:1)' "
        "and set ModelSamplingFlux width/height to 512. For 4K output from a 1024 latent, "
        "use pid_ckpt_type='2kto4k'."
    )


def _preferred_model_folder(folder_name: str, preferred_leaf: str) -> Path:
    _require_comfy()
    paths = [Path(p) for p in folder_paths.get_folder_paths(folder_name)]
    for path in paths:
        if path.name.lower() == preferred_leaf.lower():
            return path
    return paths[0]


def _existing_model_file(folder_name: str, filename: str) -> Optional[Path]:
    if folder_paths is not None:
        for candidate in (f"{NATIVE_PID_SUBFOLDER}/{filename}", filename):
            try:
                found = folder_paths.get_full_path(folder_name, candidate)
                if found:
                    return Path(found)
            except Exception:
                pass
    return None


def _ensure_comfy_org_file(
    folder_name: str,
    repo_subdir: str,
    filename: str,
    allow_download: bool = True,
) -> Path:
    existing = _existing_model_file(folder_name, filename)
    if existing is not None and existing.is_file():
        return existing

    target_dir = _preferred_model_folder(folder_name, repo_subdir) / NATIVE_PID_SUBFOLDER
    target = target_dir / filename
    if target.is_file():
        return target
    if not allow_download:
        raise PiDNodeError(
            f"Missing Comfy-Org PixelDiT file: {target}\n"
            f"Download {repo_subdir}/{filename} from {COMFY_ORG_REPO_ID}, or enable auto_download."
        )

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise PiDNodeError(
            "auto_download requires huggingface-hub. Install this node's requirements.txt and try again."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ComfyUI-PiD] downloading {COMFY_ORG_REPO_ID}/{repo_subdir}/{filename}", flush=True)
    cached_path = Path(hf_hub_download(
        repo_id=COMFY_ORG_REPO_ID,
        filename=f"{repo_subdir}/{filename}",
    ))
    shutil.copy2(str(cached_path), str(target))
    if not target.is_file():
        raise PiDNodeError(f"Hugging Face download finished but the model file is missing: {target}")
    return target

def _ensure_text_encoder(
    model_precision: str,
    text_encoder_variant: str = "default",
    allow_download: bool = True,
) -> Path:
    """Gemma text encoder loader with optional Unsloth 4-bit variant."""
    if text_encoder_variant == "unsloth-4bit":
        existing = _existing_model_file("text_encoders", UNSLOTH_GEMMA_FILENAME)
        if existing and existing.is_file():
            return existing

        target_dir = _preferred_model_folder("text_encoders", NATIVE_PID_SUBFOLDER)
        target = target_dir / UNSLOTH_GEMMA_FILENAME

        if not target.is_file() and allow_download:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise PiDNodeError("auto_download requires huggingface-hub") from None

            print(f"[ComfyUI-PiD] Downloading Unsloth 4-bit Gemma: {UNSLOTH_GEMMA_REPO_ID}")
            target_dir.mkdir(parents=True, exist_ok=True)
            cached = Path(hf_hub_download(repo_id=UNSLOTH_GEMMA_REPO_ID, filename=UNSLOTH_GEMMA_FILENAME))
            shutil.copy2(cached, target)

        if target.is_file():
            try:
                import bitsandbytes
            except ImportError:
                warnings.warn("bitsandbytes not installed — 4-bit loading may fail.", RuntimeWarning)
            return target
        raise PiDNodeError(f"Unsloth 4-bit Gemma not found at {target}")

    # Default Comfy-Org
    return _ensure_comfy_org_file(
        "text_encoders",
        "text_encoders",
        PIXELDIT_TEXT_ENCODER_FILES[model_precision],
        allow_download=allow_download,
    )

def _ensure_native_pid_assets(spec: NativePiDModelSpec, text_encoder_variant: str = "default", allow_download: bool = True) -> Tuple[Path, Path]:
    diffusion_path = _ensure_comfy_org_file(
        "diffusion_models",
        "diffusion_models",
        spec.diffusion_filename,
        allow_download=allow_download,
    )
    text_encoder_path = _ensure_text_encoder(spec.model_precision, text_encoder_variant, allow_download)
    return diffusion_path, text_encoder_path

def _load_native_pid_model(diffusion_model_path: Path):
    _require_comfy()
    return comfy_sd.load_diffusion_model(str(diffusion_model_path), model_options={})


def _load_pixeldit_clip(text_encoder_path: Path):
    _require_comfy()
    clip_type = getattr(comfy_sd.CLIPType, "PIXELDIT")
    return comfy_sd.load_clip(
        ckpt_paths=[str(text_encoder_path)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options={},
    )


def _encode_pixeldit_conditioning(clip, text: str):
    tokens = clip.tokenize(text or "")
    try:
        return clip.encode_from_tokens_scheduled(tokens, show_pbar=False)
    except TypeError:
        return clip.encode_from_tokens_scheduled(tokens)


def _apply_pid_conditioning(
    conditioning,
    latent_cpu: torch.Tensor,
    sigma: float,
):
    _require_comfy()
    sigma = float(sigma)
    if sigma < 0.0 or sigma > 1.0:
        raise PiDNodeError(
            f"Native PiD degrade_sigma must be between 0.0 and 1.0; got {sigma}. "
            "Use the PiD KSampler Capture pid_sigma output or set sigma to a normalized native value."
        )
    # Preserve the original NVIDIA PiD node behavior: captured Comfy latents
    # are passed as LQ_latent without Flux/SD3/SDXL process_in scaling.
    lq_latent = latent_cpu
    if lq_latent.ndim == 5:
        lq_latent = lq_latent[:, :, 0]
    sigma_t = torch.tensor([sigma], dtype=torch.float32)
    return node_helpers.conditioning_set_values(
        conditioning,
        {"lq_latent": lq_latent.contiguous(), "degrade_sigma": sigma_t},
    )


def _native_pid_student_sigmas(pid_steps: int, device=None) -> torch.Tensor:
    steps = int(pid_steps)
    base = torch.tensor(NATIVE_PID_STUDENT_T_LIST, dtype=torch.float32, device=device)
    if steps == 4:
        return base
    if 1 <= steps < 4:
        indices = torch.linspace(0, len(base) - 1, steps + 1, device=device).round().long()
        return base[indices]
    raise PiDNodeError(
        "Native PiD's NVIDIA student SDE sampler supports pid_steps 1-4. "
        "Use pid_steps=4 for released distilled PiD checkpoints."
    )


class _PiDStudentSDESampler:
    """NVIDIA distilled PiD update loop using ComfyUI's native model wrapper."""

    def __init__(self, pid_steps: int):
        self.pid_steps = int(pid_steps)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        del latent_image, denoise_mask, disable_pbar
        device = noise.device
        dtype = noise.dtype
        sigmas = sigmas.to(device=device, dtype=torch.float32)
        total_steps = max(1, int(sigmas.numel()) - 1)
        seed = int(extra_args.get("seed", 0) or 0)

        # NVIDIA's distilled PiD sampler draws both the initial image noise and
        # intermediate SDE noise from one CUDA generator. ComfyUI normally passes
        # CPU-generated noise into samplers, so recreate the reference stream here.
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        x = torch.randn(tuple(noise.shape), dtype=torch.float32, generator=generator, device=generator_device)
        x = x.to(device=device, dtype=dtype)

        model_options = extra_args.get("model_options", {})
        batch = int(noise.shape[0])
        view_shape = [batch] + [1] * (noise.ndim - 1)
        for index, (t_cur, t_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            t_cur_batch = t_cur.expand(batch).to(device=device)
            x0_pred = model_wrap(x, t_cur_batch, model_options=model_options, seed=seed)
            if callback is not None:
                callback(index, x0_pred, x, total_steps)
            if float(t_next.item()) > 0.0:
                eps = torch.randn(
                    tuple(x0_pred.shape),
                    dtype=torch.float32,
                    generator=generator,
                    device=generator_device,
                )
                eps = eps.to(device=device, dtype=dtype)
                t_next_b = t_next.reshape(1).expand(view_shape).to(device=device, dtype=x0_pred.dtype)
                x = (1.0 - t_next_b) * x0_pred + t_next_b * eps
            else:
                x = x0_pred
        return x.clamp(-1.0, 1.0)


def _sample_native_pid_student_sde(
    model,
    noise: torch.Tensor,
    pid_steps: int,
    cfg_scale: float,
    positive,
    negative,
    pixel_samples: torch.Tensor,
    callback,
    disable_pbar: bool,
    seed: int,
) -> torch.Tensor:
    if comfy_samplers is None:
        raise PiDNodeError("ComfyUI samplers are unavailable; native PiD sampling cannot run.")
    guider = comfy_samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(float(cfg_scale))
    sigmas = _native_pid_student_sigmas(pid_steps, device=noise.device)
    sampler = _PiDStudentSDESampler(pid_steps)
    return guider.sample(
        noise,
        pixel_samples,
        sampler,
        sigmas,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=int(seed),
    )


def _intermediate_device():
    if model_management is not None:
        try:
            return model_management.intermediate_device()
        except Exception:
            pass
    return torch.device("cpu")


def _make_pixel_latent(batch_size: int, infer_image_size: Tuple[int, int]) -> dict:
    height, width = int(infer_image_size[0]), int(infer_image_size[1])
    if height <= 0 or width <= 0:
        raise PiDNodeError(f"infer_image_size must be positive, got {(height, width)!r}")
    samples = torch.zeros([int(batch_size), 3, height, width], device=_intermediate_device())
    return {"samples": samples}


def _make_pid_progress_bar(total: int):
    if comfy_utils is None:
        return None
    try:
        if not getattr(comfy_utils, "PROGRESS_BAR_ENABLED", True):
            return None
        return comfy_utils.ProgressBar(max(1, int(total)))
    except Exception:
        return None


def _update_pid_progress_bar(pbar, current: int, total: int) -> None:
    if pbar is None:
        return
    try:
        pbar.update_absolute(int(current), int(total))
    except Exception:
        pass


def _format_pid_runtime_error(
    exc: BaseException,
    infer_image_size: Tuple[int, int],
    ckpt_type: str,
    scale: int,
) -> PiDNodeError:
    h, w = int(infer_image_size[0]), int(infer_image_size[1])
    msg = str(exc)
    lower = msg.lower()
    advice = (
        f"Native PiD inference failed while generating {w}x{h} with {ckpt_type!r}, scale={scale}.\n\n"
        "Most common causes are missing Comfy-Org PixelDiT files, latent/backbone mismatch, "
        "or VRAM pressure while ComfyUI loads the native PiD model.\n\n"
        "Try enabling auto_download, keeping cleanup options enabled, or using a smaller base latent.\n\n"
        f"Original error: {msg}"
    )
    if "out of memory" in lower or "cuda" in lower or "cudamalloc" in lower:
        return PiDNodeError(advice)
    return PiDNodeError(f"Native PiD inference failed. Original error: {msg}")


def _resolve_native_pid_paths(
    spec: NativePiDModelSpec,
    allow_download: bool = True,
    diffusion_model_path: Optional[Path] = None,
    text_encoder_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    _require_comfy()
    if diffusion_model_path and text_encoder_path:
        diffusion_path = Path(diffusion_model_path)
        text_path = Path(text_encoder_path)
        if not diffusion_path.is_file():
            raise PiDNodeError(f"Missing prepared native PiD diffusion model: {diffusion_path}")
        if not text_path.is_file():
            raise PiDNodeError(f"Missing prepared PixelDiT text encoder: {text_path}")
    else:
        diffusion_path, text_path = _ensure_native_pid_assets(spec, allow_download=allow_download)
        if diffusion_model_path:
            diffusion_path = Path(diffusion_model_path)
        if text_encoder_path:
            text_path = Path(text_encoder_path)
    return diffusion_path, text_path


class _NativePiDSession:
    """Loaded native PiD model pair that can sample multiple tiles."""
    def __init__(self, spec: NativePiDModelSpec, diffusion_path: Path, text_encoder_path: Path, enable_torch_compile: str = "disabled"):
        self.spec = spec
        self.diffusion_path = Path(diffusion_path)
        self.text_encoder_path = Path(text_encoder_path)
        self.model = _load_native_pid_model(self.diffusion_path)
        self.clip = _load_pixeldit_clip(self.text_encoder_path)

        # Apply torch.compile if requested
        if enable_torch_compile != "disabled":
            mode = None if enable_torch_compile == "default" else enable_torch_compile
            print(f"[ComfyUI-PiD] Applying torch.compile (mode={mode}) to PiD model...")
            try:
                # Try to compile the inner model (common in ComfyUI ModelPatcher)
                if hasattr(self.model, "model") and hasattr(self.model.model, "forward"):
                    self.model.model = torch.compile(self.model.model, mode=mode, fullgraph=False)
                elif hasattr(self.model, "diffusion_model"):
                    self.model.diffusion_model = torch.compile(self.model.diffusion_model, mode=mode, fullgraph=False)
                else:
                    # Fallback: try the top level
                    self.model = torch.compile(self.model, mode=mode, fullgraph=False)
                print(f"[ComfyUI-PiD] torch.compile successful!")
            except Exception as e:
                print(f"[ComfyUI-PiD] torch.compile failed: {e}. Falling back to eager mode.")

    @classmethod
    def create(
        cls,
        spec: NativePiDModelSpec,
        *,
        allow_download: bool = True,
        diffusion_model_path: Optional[Path] = None,
        text_encoder_path: Optional[Path] = None,
        text_encoder_variant: str = "default",   # ← MUST HAVE THIS
        enable_torch_compile: str = "disabled",
    ) -> "_NativePiDSession":
        diffusion_path, text_path = _resolve_native_pid_paths(
            spec,
            allow_download=allow_download,
            diffusion_model_path=diffusion_model_path,
            text_encoder_path=text_encoder_path,
        )
        return cls(spec, diffusion_path, text_path, enable_torch_compile=enable_torch_compile)

    def close(self) -> None:
        self.clip = None
        self.model = None
        _free_cuda_memory(aggressive=True)

    def __enter__(self) -> "_NativePiDSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()

    def sample(
        self,
        caption: str,
        latent_cpu: torch.Tensor,
        sigma: float,
        infer_image_size: Tuple[int, int],
        pid_steps: int,
        cfg_scale: float,
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        positive = _encode_pixeldit_conditioning(self.clip, caption or "")
        negative = _encode_pixeldit_conditioning(self.clip, "")
        positive = _apply_pid_conditioning(positive, latent_cpu, sigma)
        negative = _apply_pid_conditioning(negative, latent_cpu, sigma)

        pixel_latent = _make_pixel_latent(int(latent_cpu.shape[0]), infer_image_size)
        pixel_samples = pixel_latent["samples"]
        noise = comfy_sample.prepare_noise(pixel_samples, int(seed), None)

        total_steps = max(1, int(pid_steps))

        def sampler_callback(step, x0, x, total):
            if progress_callback is not None:
                progress_callback(min(int(step) + 1, int(total)), int(total))

        disable_pbar = False
        if comfy_utils is not None:
            disable_pbar = not bool(getattr(comfy_utils, "PROGRESS_BAR_ENABLED", True))

        with torch.inference_mode():
            samples = _sample_native_pid_student_sde(
                self.model,
                noise,
                total_steps,
                float(cfg_scale),
                positive,
                negative,
                pixel_samples,
                sampler_callback,
                disable_pbar,
                int(seed),
            )
        if progress_callback is not None:
            progress_callback(total_steps, total_steps)
        return samples.detach()


def _run_native_pid_decode(
    spec: NativePiDModelSpec,
    caption: str,
    latent_cpu: torch.Tensor,
    sigma: float,
    infer_image_size: Tuple[int, int],
    pid_steps: int,
    cfg_scale: float,
    seed: int,
    *,
    allow_download: bool = True,
    diffusion_model_path: Optional[Path] = None,
    text_encoder_path: Optional[Path] = None,
    enable_torch_compile: str = "disabled",   # ← ADD THIS
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    with _NativePiDSession.create(
        spec,
        allow_download=allow_download,
        diffusion_model_path=diffusion_model_path,
        text_encoder_path=text_encoder_path,
        enable_torch_compile=enable_torch_compile,   # ← ADD THIS
    ) as session:
        return session.sample(
            caption,
            latent_cpu,
            sigma,
            infer_image_size,
            pid_steps,
            cfg_scale,
            seed,
            progress_callback=progress_callback,
        )


def _native_pixel_to_comfy_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 5:
        if image.shape[2] == 1 and image.shape[1] in (1, 3, 4):
            image = image[:, :, 0, :, :]
        elif image.shape[1] == 1 and image.shape[2] in (1, 3, 4):
            image = image[:, 0, :, :, :]
        else:
            raise PiDNodeError(f"Expected native PiD output as [B,C,H,W], got shape {list(image.shape)}")
    if image.ndim != 4:
        raise PiDNodeError(f"Expected native PiD output as [B,C,H,W], got shape {list(image.shape)}")
    if image.shape[1] not in (1, 3, 4):
        raise PiDNodeError(f"Expected native PiD output with 1, 3, or 4 channels; got shape {list(image.shape)}")
    image = image[:, :3].detach().float().cpu()
    # Native PixelDiT/PiD samples are pixel-space latents in [-1, 1].
    image = image.add(1.0).div(2.0).clamp(0.0, 1.0)
    return image.movedim(1, -1).contiguous()


def _normalize_pid_samples(samples):
    return samples


class PiDDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "backbone": (BACKBONE_CHOICES, {"default": "zimage"}),
                "pid_ckpt_type": (PID_CKPT_TYPES, {"default": "2k"}),
                "pid_steps": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "scale": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "auto_download": ("BOOLEAN", {"default": True}),
                "model_precision": (MODEL_PRECISION_CHOICES, {"default": "bf16"}),
                "enable_torch_compile": (TORCH_COMPILE_MODES, {"default": "disabled"}),
                "text_encoder_variant": (TEXT_ENCODER_VARIANTS, {"default": "default"}),
                "unload_comfy_before_pid": ("BOOLEAN", {"default": True}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
                "caption": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "PiD"

    def decode(
        self,
        latent,
        caption: str,
        backbone: str,
        pid_ckpt_type: str,
        pid_steps: int,
        scale: int,
        cfg_scale: float,
        sigma: float,
        seed: int,
        auto_download: bool,
        model_precision: str = "bf16",
        unload_comfy_before_pid: bool = True,
        aggressive_cleanup: bool = True,
        pid_source_dir: str = "",
        text_encoder_variant: str = "default",   # ← ADD THIS LINE
        enable_torch_compile: str = "disabled",
    ):
        del pid_source_dir

        backbone = str(backbone).strip()
        pid_ckpt_type = str(pid_ckpt_type).strip()
        _validate_latent_source_backbone(latent, backbone)
        spec = _checkpoint_for(backbone, pid_ckpt_type, model_precision)
        scale = _normalize_scale_for_checkpoint(backbone, spec, int(scale))

        samples = _latent_samples(latent)
        sigma = _latent_pid_sigma(latent, sigma)
        if samples.shape[1] != spec.latent_channels:
            raise PiDNodeError(
                f"{spec.label} PiD expects {spec.latent_channels}-channel latents. "
                f"Got {samples.shape[1]} channels."
            )
        samples_cpu = samples.detach().to("cpu").contiguous()
        samples_cpu, sigma = _prepare_latent_for_pid_backbone(samples_cpu, sigma, backbone)
        h, w = _infer_lq_size_from_latent(samples, backbone)
        _validate_pid_base_resolution(spec, (h, w))
        infer_image_size = (int(h) * int(scale), int(w) * int(scale))
        _log_pid_decode_plan(spec, tuple(samples.shape), (h, w), infer_image_size, sigma)
        del samples
        latent = None

        if unload_comfy_before_pid:
            _free_cuda_memory(aggressive=bool(aggressive_cleanup))

        _warn_if_non_distilled_step_count(pid_steps)
        pbar = _make_pid_progress_bar(pid_steps)

        def update_progress(current: int, total: int) -> None:
            _update_pid_progress_bar(pbar, current, total)

        _reset_cuda_peak_memory_stats()
        try:
            out = _run_native_pid_decode(
                spec,
                caption or "",
                samples_cpu,
                float(sigma),
                infer_image_size,
                int(pid_steps),
                float(cfg_scale),
                int(seed),
                allow_download=bool(auto_download),
                progress_callback=update_progress if pbar is not None else None,
                enable_torch_compile=enable_torch_compile,   # ← ADD THIS
            )
            _log_cuda_peak_memory("direct native decode")
        except Exception as exc:
            _free_cuda_memory(aggressive=True)
            raise _format_pid_runtime_error(
                exc,
                infer_image_size,
                f"{backbone}/{pid_ckpt_type}/{spec.diffusion_filename}",
                int(scale),
            ) from exc

        image = _native_pixel_to_comfy_image(out)
        del out
        del samples_cpu
        _free_cuda_memory(aggressive=bool(aggressive_cleanup))
        return (image,)


NODE_CLASS_MAPPINGS = {
    "PiDDecode": PiDDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDDecode": "PiD Decode",
}
