from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils
    import folder_paths
except Exception:  # pragma: no cover - ComfyUI-only imports
    comfy_sd = None
    comfy_utils = None
    folder_paths = None

try:
    from .pid_decode import (
        MODEL_PRECISION_CHOICES,
        NATIVE_PID_SUBFOLDER,
        PID_BACKBONES,
        PiDNodeError,
        _NativePiDSession,
        _checkpoint_for,
        _format_pid_runtime_error,
        _free_cuda_memory,
        _log_cuda_peak_memory,
        _make_pid_progress_bar,
        _native_pixel_to_comfy_image,
        _preferred_model_folder,
        _require_comfy,
        _reset_cuda_peak_memory_stats,
        _update_pid_progress_bar,
    )
except ImportError:  # pragma: no cover
    from pid_decode import (
        MODEL_PRECISION_CHOICES,
        NATIVE_PID_SUBFOLDER,
        PID_BACKBONES,
        PiDNodeError,
        _NativePiDSession,
        _checkpoint_for,
        _format_pid_runtime_error,
        _free_cuda_memory,
        _log_cuda_peak_memory,
        _make_pid_progress_bar,
        _native_pixel_to_comfy_image,
        _preferred_model_folder,
        _require_comfy,
        _reset_cuda_peak_memory_stats,
        _update_pid_progress_bar,
    )

# Optional alternate: Unsloth 4-bit Gemma (lower VRAM)
UNSLOTH_GEMMA_REPO_ID = "unsloth/gemma-2-2b-it-bnb-4bit"
UNSLOTH_GEMMA_FILENAME = "model.safetensors"
TEXT_ENCODER_VARIANTS = ["default", "unsloth-4bit"]

TORCH_COMPILE_MODES = ["disabled", "default", "max-autotune"]

PID_UPSCALE_BACKBONES = [
    name for name, info in PID_BACKBONES.items()
    if "2k" in info.ckpt_types
]
UPSCALE_FACTOR_CHOICES = ["2x", "4x", "6x", "8x"]
PID_UPSCALE_CKPT_TYPES = ["2k", "2kto4k"]
PID_UPSCALE_NATIVE_SCALE = 4
PID_UPSCALE_STEPS = 4
PID_UPSCALE_CFG = 1.0

_VAE_DOWNLOADS = {
    "flux1": {
        "local_names": ("ae.safetensors",),
        "target_name": "ae.safetensors",
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "split_files/vae/ae.safetensors",
    },
    "flux2": {
        "local_names": ("flux2-vae.safetensors", "flux2_ae.safetensors"),
        "target_name": "flux2_ae.safetensors",
        "repo_id": "nvidia/PiD",
        "filename": "checkpoints/flux2_ae.safetensors",
    },
    "sd3": {
        "local_names": ("sd3_vae.safetensors", "diffusion_pytorch_model.safetensors"),
        "target_name": "sd3_vae.safetensors",
        "repo_id": "nvidia/PiD",
        "filename": "checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors",
    },
}


@dataclass(frozen=True)
class SpatialTile:
    index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class UpscaleProfile:
    tile_size: int
    tile_overlap: int
    small_edge: int


PID_UPSCALE_PROFILES: Dict[str, UpscaleProfile] = {
    "2k": UpscaleProfile(tile_size=512, tile_overlap=64, small_edge=512),
    "2kto4k": UpscaleProfile(tile_size=1024, tile_overlap=128, small_edge=1024),
}


def _upscale_profile_for(pid_ckpt_type: str) -> UpscaleProfile:
    ckpt_type = str(pid_ckpt_type).strip()
    try:
        profile = PID_UPSCALE_PROFILES[ckpt_type]
    except KeyError as exc:
        raise PiDNodeError(
            f"Unknown PiD Upscale pid_ckpt_type={pid_ckpt_type!r}; expected one of {PID_UPSCALE_CKPT_TYPES}"
        ) from exc
    validate_tile_settings(profile.tile_size, profile.tile_overlap)
    return profile


def validate_tile_settings(tile_size: int, overlap: int) -> None:
    if int(tile_size) <= 0:
        raise PiDNodeError("PiD Upscale tile size must be greater than 0.")
    if int(overlap) < 0:
        raise PiDNodeError("PiD Upscale tile overlap must be greater than or equal to 0.")
    if int(overlap) >= int(tile_size):
        raise PiDNodeError("PiD Upscale tile overlap must be smaller than tile size.")


def tile_origins(length: int, tile_size: int, overlap: int) -> List[int]:
    validate_tile_settings(tile_size, overlap)
    length = int(length)
    tile_size = int(tile_size)
    overlap = int(overlap)
    if length <= 0:
        raise PiDNodeError("PiD Upscale image dimensions must be greater than 0.")
    if length <= tile_size:
        return [0]

    step = tile_size - overlap
    origins = list(range(0, max(length - tile_size, 0) + 1, step))
    final_origin = length - tile_size
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def generate_tiles(width: int, height: int, tile_size: int, overlap: int) -> List[SpatialTile]:
    xs = tile_origins(width, tile_size, overlap)
    ys = tile_origins(height, tile_size, overlap)
    tiles: List[SpatialTile] = []
    index = 0
    for y in ys:
        for x in xs:
            tiles.append(
                SpatialTile(
                    index=index,
                    x=x,
                    y=y,
                    width=min(tile_size, width - x),
                    height=min(tile_size, height - y),
                )
            )
            index += 1
    return tiles


def scaled_tile_bounds(
    tile: SpatialTile,
    scale: float,
    output_width: int,
    output_height: int,
) -> Tuple[int, int, int, int]:
    x0 = int(round(tile.x * scale))
    y0 = int(round(tile.y * scale))
    x1 = min(int(output_width), int(round((tile.x + tile.width) * scale)))
    y1 = min(int(output_height), int(round((tile.y + tile.height) * scale)))
    return x0, y0, x1, y1


def extract_reflect_tile(image: torch.Tensor, tile: SpatialTile, tile_size: int) -> torch.Tensor:
    if image.ndim != 3:
        raise PiDNodeError(f"Expected image tile source as [H,W,C], got shape {list(image.shape)}")

    cropped = image[tile.y: tile.y + tile.height, tile.x: tile.x + tile.width, :]
    pad_h = int(tile_size) - int(cropped.shape[0])
    pad_w = int(tile_size) - int(cropped.shape[1])
    if pad_h == 0 and pad_w == 0:
        return cropped.clone()

    mode = "reflect"
    if cropped.shape[0] <= 1 or cropped.shape[1] <= 1 or pad_h >= cropped.shape[0] or pad_w >= cropped.shape[1]:
        mode = "replicate"

    tile_chw = cropped.movedim(-1, 0).unsqueeze(0)
    padded = F.pad(tile_chw, (0, pad_w, 0, pad_h), mode=mode)
    return padded.squeeze(0).movedim(0, -1).contiguous()


def raised_cosine_mask(
    tile: SpatialTile,
    width: int,
    height: int,
    overlap: int,
    full_width: int,
    full_height: int,
) -> torch.Tensor:
    if int(width) <= 0 or int(height) <= 0:
        raise PiDNodeError("PiD Upscale mask dimensions must be greater than 0.")

    mask_x = torch.ones(int(width), dtype=torch.float32)
    mask_y = torch.ones(int(height), dtype=torch.float32)
    scaled_overlap = min(int(overlap), max(int(width) - 1, 0), max(int(height) - 1, 0))

    if scaled_overlap > 0:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, scaled_overlap + 2, dtype=torch.float32)[1:-1])
        if tile.x > 0:
            mask_x[:scaled_overlap] = ramp
        if tile.x + tile.width < int(full_width):
            mask_x[-scaled_overlap:] = ramp.flip(0)
        if tile.y > 0:
            mask_y[:scaled_overlap] = ramp
        if tile.y + tile.height < int(full_height):
            mask_y[-scaled_overlap:] = ramp.flip(0)

    return mask_y[:, None] * mask_x[None, :]


def stitch_tiles(
    tiles: Iterable[Tuple[SpatialTile, torch.Tensor]],
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
    scale: float,
    overlap: int,
) -> torch.Tensor:
    canvas: Optional[torch.Tensor] = None
    weights = torch.zeros((int(output_height), int(output_width), 1), dtype=torch.float32)

    for tile, tile_output in tiles:
        if tile_output.ndim != 3:
            raise PiDNodeError(f"Expected tile output as [H,W,C], got shape {list(tile_output.shape)}")

        x0, y0, x1, y1 = scaled_tile_bounds(tile, scale, output_width, output_height)
        target_w = x1 - x0
        target_h = y1 - y0
        if target_w <= 0 or target_h <= 0:
            continue

        cropped = tile_output[:target_h, :target_w, :].detach().float().cpu()
        if canvas is None:
            canvas = torch.zeros((int(output_height), int(output_width), cropped.shape[2]), dtype=torch.float32)

        scaled_overlap = int(round(float(overlap) * float(scale)))
        mask = raised_cosine_mask(tile, target_w, target_h, scaled_overlap, input_width, input_height)[:, :, None]
        canvas[y0:y1, x0:x1, :] += cropped * mask
        weights[y0:y1, x0:x1, :] += mask

    if canvas is None:
        raise PiDNodeError("PiD Upscale did not produce any stitched tiles.")
    if torch.any(weights <= 0.0):
        raise PiDNodeError("PiD Upscale tiling produced uncovered output pixels.")

    return (canvas / weights).clamp(0.0, 1.0).contiguous()


def _parse_upscale_factor(upscale_factor: str) -> int:
    text = str(upscale_factor).strip().lower()
    if text.endswith("x"):
        text = text[:-1]
    try:
        factor = int(text)
    except ValueError as exc:
        raise PiDNodeError(f"Unknown upscale_factor={upscale_factor!r}; expected one of {UPSCALE_FACTOR_CHOICES}") from exc
    if f"{factor}x" not in UPSCALE_FACTOR_CHOICES:
        raise PiDNodeError(f"Unknown upscale_factor={upscale_factor!r}; expected one of {UPSCALE_FACTOR_CHOICES}")
    return factor


def _parse_strength_sigma(strength) -> float:
    try:
        sigma = float(strength)
    except (TypeError, ValueError) as exc:
        raise PiDNodeError(
            f"Unknown PiD Upscale strength={strength!r}; expected a numeric sigma from 0.0 to 1.0."
        ) from exc
    if sigma < 0.0 or sigma > 1.0:
        raise PiDNodeError(f"PiD Upscale strength sigma must be between 0.0 and 1.0; got {sigma}.")
    return sigma


def _add_latent_noise(clean_latent: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    sigma = float(sigma)
    if sigma < 0.0 or sigma > 1.0:
        raise PiDNodeError(f"PiD Upscale strength sigma must be between 0.0 and 1.0; got {sigma}.")
    if sigma <= 0.0:
        return clean_latent

    latent = clean_latent.float()
    generator = torch.Generator(device=latent.device).manual_seed(int(seed))
    noise = torch.randn(
        latent.shape,
        generator=generator,
        device=latent.device,
        dtype=latent.dtype,
    )
    return ((1.0 - sigma) * latent + sigma * noise).to(dtype=clean_latent.dtype).contiguous()


def _round_to_multiple(value: float, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _resize_image(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if comfy_utils is None:
        raise PiDNodeError("PiD Upscale resizing requires ComfyUI's comfy.utils module.")
    if image.ndim != 3:
        raise PiDNodeError(f"Expected image as [H,W,C], got shape {list(image.shape)}")
    samples = image.unsqueeze(0).movedim(-1, 1).detach().float().cpu()
    resized = comfy_utils.common_upscale(samples, int(width), int(height), "lanczos", "disabled")
    return resized.movedim(1, -1).squeeze(0).clamp(0.0, 1.0).contiguous()


def _resize_to_long_edge(image: torch.Tensor, long_edge: int, multiple: int) -> torch.Tensor:
    height, width = int(image.shape[0]), int(image.shape[1])
    if max(width, height) <= 0:
        raise PiDNodeError("PiD Upscale cannot resize an empty image.")
    scale = float(long_edge) / float(max(width, height))
    new_w = _round_to_multiple(width * scale, multiple)
    new_h = _round_to_multiple(height * scale, multiple)
    return _resize_image(image, new_w, new_h)


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise PiDNodeError("PiD Upscale expected a ComfyUI IMAGE tensor.")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise PiDNodeError(f"PiD Upscale expected IMAGE as [B,H,W,C], got shape {list(image.shape)}")
    if image.shape[-1] < 3:
        raise PiDNodeError(f"PiD Upscale expected at least 3 image channels, got shape {list(image.shape)}")
    return image[..., :3].detach().float().cpu().clamp(0.0, 1.0).contiguous()


def _vae_existing_path(names: Tuple[str, ...]) -> Optional[Path]:
    if folder_paths is None:
        return None

    candidates: List[Path] = []
    for name in names:
        for rel_name in (f"{NATIVE_PID_SUBFOLDER}/{name}", name):
            try:
                found = folder_paths.get_full_path("vae", rel_name)
                if found:
                    candidates.append(Path(found))
            except Exception:
                pass

    try:
        for listed in folder_paths.get_filename_list("vae"):
            listed_path = Path(listed)
            for name in names:
                if listed_path.name.lower() == name.lower():
                    try:
                        found = folder_paths.get_full_path("vae", listed)
                        if found:
                            candidates.append(Path(found))
                    except Exception:
                        pass
    except Exception:
        pass

    seen = set()
    for path in candidates:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _download_vae_file(config: dict) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise PiDNodeError(
            "PiD Upscale auto_download requires huggingface-hub. Install this node's requirements.txt and try again."
        ) from exc

    target_dir = _preferred_model_folder("vae", "vae") / NATIVE_PID_SUBFOLDER
    target = target_dir / str(config["target_name"])
    if target.is_file():
        return target

    target_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[ComfyUI-PiD] downloading {config['repo_id']}/{config['filename']} for PiD Upscale VAE",
        flush=True,
    )
    cached = Path(hf_hub_download(repo_id=str(config["repo_id"]), filename=str(config["filename"])))
    shutil.copy2(str(cached), str(target))
    if not target.is_file():
        raise PiDNodeError(f"PiD Upscale VAE download finished but the file is missing: {target}")
    return target


def _ensure_vae_path(backbone: str, allow_download: bool) -> Path:
    info = PID_BACKBONES.get(str(backbone).strip())
    if info is None:
        raise PiDNodeError(f"Unknown PiD Upscale backbone={backbone!r}; expected one of {PID_UPSCALE_BACKBONES}")

    config = _VAE_DOWNLOADS.get(info.registry_key)
    if config is None:
        raise PiDNodeError(f"PiD Upscale does not have an image VAE mapping for backbone={backbone!r}.")

    existing = _vae_existing_path(tuple(config["local_names"]))
    if existing is not None:
        return existing

    if not allow_download:
        names = ", ".join(config["local_names"])
        raise PiDNodeError(
            f"PiD Upscale could not find a local VAE for backbone={backbone!r}. "
            f"Expected one of: {names}. Enable auto_download or place the file under ComfyUI/models/vae."
        )

    return _download_vae_file(config)


def _load_image_vae(backbone: str, allow_download: bool):
    _require_comfy()
    if comfy_sd is None or comfy_utils is None or folder_paths is None:
        raise PiDNodeError("PiD Upscale VAE loading must run inside ComfyUI.")

    vae_path = _ensure_vae_path(backbone, allow_download)
    sd, metadata = comfy_utils.load_torch_file(str(vae_path), return_metadata=True)
    vae = comfy_sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    print(f"[ComfyUI-PiD] PiD Upscale using VAE: {vae_path}", flush=True)
    return vae


def _encode_tile_latent(vae, image: torch.Tensor, expected_channels: int) -> torch.Tensor:
    latent = vae.encode(image.unsqueeze(0))
    if getattr(latent, "is_nested", False):
        latent = latent.unbind()[0]
    if latent.ndim == 5:
        if latent.shape[2] == 1:
            latent = latent[:, :, 0, :, :]
        elif latent.shape[1] == 1:
            latent = latent[:, 0, :, :, :]
    if latent.ndim != 4:
        raise PiDNodeError(f"PiD Upscale VAE returned unsupported latent shape {list(latent.shape)}")
    if int(latent.shape[1]) != int(expected_channels):
        raise PiDNodeError(
            f"PiD Upscale backbone expects {expected_channels}-channel latents, "
            f"but the selected VAE returned {latent.shape[1]} channels."
        )
    return latent.detach().to("cpu").contiguous()


def _pid_upscale_once(
    session: _NativePiDSession,
    vae,
    spec,
    image: torch.Tensor,
    caption: str,
    seed: int,
    sigma: float,
    progress_callback: Optional[Callable[[int, int], None]],
) -> torch.Tensor:
    latent_cpu = _encode_tile_latent(vae, image, spec.latent_channels)
    latent_cpu = _add_latent_noise(latent_cpu, sigma, int(seed))
    base_h = int(latent_cpu.shape[-2]) * int(spec.latent_downscale)
    base_w = int(latent_cpu.shape[-1]) * int(spec.latent_downscale)
    infer_image_size = (base_h * PID_UPSCALE_NATIVE_SCALE, base_w * PID_UPSCALE_NATIVE_SCALE)
    out = session.sample(
        caption or "",
        latent_cpu,
        float(sigma),
        infer_image_size,
        PID_UPSCALE_STEPS,
        PID_UPSCALE_CFG,
        int(seed),
        progress_callback=progress_callback,
    )
    image_out = _native_pixel_to_comfy_image(out)[0]
    del out
    del latent_cpu
    return image_out


def _planned_pid_calls(width: int, height: int, latent_downscale: int, profile: UpscaleProfile) -> int:
    if max(int(width), int(height)) < profile.small_edge:
        scale = float(profile.small_edge) / float(max(int(width), int(height)))
        pre_w = _round_to_multiple(int(width) * scale, latent_downscale)
        pre_h = _round_to_multiple(int(height) * scale, latent_downscale)
        work_w = pre_w * PID_UPSCALE_NATIVE_SCALE
        work_h = pre_h * PID_UPSCALE_NATIVE_SCALE
        return 1 + len(generate_tiles(work_w, work_h, profile.tile_size, profile.tile_overlap))
    return len(generate_tiles(int(width), int(height), profile.tile_size, profile.tile_overlap))


def _run_tiled_upscale(
    image: torch.Tensor,
    pid_once: Callable[[torch.Tensor, int], torch.Tensor],
    seed_base: int,
    latent_downscale: int,
    profile: UpscaleProfile,
) -> torch.Tensor:
    original_h, original_w = int(image.shape[0]), int(image.shape[1])

    working = image
    next_seed = int(seed_base)
    if max(original_w, original_h) < profile.small_edge:
        pre_input = _resize_to_long_edge(working, profile.small_edge, latent_downscale)
        working = pid_once(pre_input, next_seed)
        next_seed += 1

    work_h, work_w = int(working.shape[0]), int(working.shape[1])
    tiles = generate_tiles(work_w, work_h, profile.tile_size, profile.tile_overlap)
    tile_outputs = []
    for tile in tiles:
        tile_input = extract_reflect_tile(working, tile, profile.tile_size)
        tile_outputs.append((tile, pid_once(tile_input, next_seed + tile.index)))

    output_w = work_w * PID_UPSCALE_NATIVE_SCALE
    output_h = work_h * PID_UPSCALE_NATIVE_SCALE
    return stitch_tiles(
        tile_outputs,
        input_width=work_w,
        input_height=work_h,
        output_width=output_w,
        output_height=output_h,
        scale=PID_UPSCALE_NATIVE_SCALE,
        overlap=profile.tile_overlap,
    )


class PiDUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pid_ckpt_type": (PID_UPSCALE_CKPT_TYPES, {"default": "2k"}),
                "backbone": (PID_UPSCALE_BACKBONES, {"default": "flux"}),
                "auto_download": ("BOOLEAN", {"default": True}),
                "model_precision": (MODEL_PRECISION_CHOICES, {"default": "bf16"}),
                "enable_torch_compile": (TORCH_COMPILE_MODES, {"default": "disabled"}),
                "upscale_factor": (UPSCALE_FACTOR_CHOICES, {"default": "4x"}),
                "strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "text_encoder_variant": (TEXT_ENCODER_VARIANTS, {"default": "default"}),
            },
            "optional": {
                "caption": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "PiD"

    def upscale(
        self,
        image,
        pid_ckpt_type: str,
        backbone: str,
        auto_download: bool,
        model_precision: str = "bf16",
        enable_torch_compile: str = "disabled",
        text_encoder_variant: str = "default",
        upscale_factor: str = "4x",
        strength=0.4,
        caption: str = "",
    ):
        _require_comfy()
        backbone = str(backbone).strip()
        if backbone not in PID_UPSCALE_BACKBONES:
            raise PiDNodeError(
                f"PiD Upscale backbone={backbone!r} does not have a supported image VAE mapping. "
                f"Expected one of {PID_UPSCALE_BACKBONES}."
            )

        pid_ckpt_type = str(pid_ckpt_type).strip()
        caption = str(caption or "").strip()
        profile = _upscale_profile_for(pid_ckpt_type)
        factor = _parse_upscale_factor(upscale_factor)
        sigma = _parse_strength_sigma(strength)
        images = _ensure_image_batch(image)
        _, original_h, original_w, _ = images.shape

        spec = _checkpoint_for(backbone, pid_ckpt_type, model_precision)
        if int(spec.scale) != PID_UPSCALE_NATIVE_SCALE:
            raise PiDNodeError(f"PiD Upscale expected a native 4x PiD checkpoint, got scale={spec.scale}.")

        total_pid_calls = sum(
            _planned_pid_calls(original_w, original_h, spec.latent_downscale, profile)
            for _ in range(int(images.shape[0]))
        )
        total_steps = max(1, total_pid_calls * PID_UPSCALE_STEPS)
        pbar = _make_pid_progress_bar(total_steps)
        completed_steps = 0

        def aggregate_progress(current: int, total: int) -> None:
            if pbar is not None:
                _update_pid_progress_bar(pbar, completed_steps + int(current), total_steps)

        vae = _load_image_vae(backbone, bool(auto_download))
        outputs = []
        _free_cuda_memory(aggressive=True)
        _reset_cuda_peak_memory_stats()

        try:
            with _NativePiDSession.create(spec, text_encoder_variant=text_encoder_variant, allow_download=bool(auto_download), enable_torch_compile=enable_torch_compile,) as session:

                def run_pid_once(tile_image: torch.Tensor, seed: int) -> torch.Tensor:
                    nonlocal completed_steps
                    result = _pid_upscale_once(
                        session,
                        vae,
                        spec,
                        tile_image,
                        caption,
                        int(seed),
                        sigma,
                        aggregate_progress if pbar is not None else None,
                    )
                    completed_steps += PID_UPSCALE_STEPS
                    if pbar is not None:
                        _update_pid_progress_bar(pbar, min(completed_steps, total_steps), total_steps)
                    return result

                for batch_index, single_image in enumerate(images):
                    print(
                        f"[ComfyUI-PiD] PiD Upscale image {batch_index + 1}/{images.shape[0]}: "
                        f"{original_w}x{original_h}, backbone={backbone}, pid_ckpt_type={pid_ckpt_type}, "
                        f"checkpoint={spec.diffusion_filename}, tile={profile.tile_size}, factor={factor}x, "
                        f"output={original_w * factor}x{original_h * factor}, caption_chars={len(caption)}, "
                        f"strength={strength} (sigma={sigma:g})",
                        flush=True,
                    )
                    stitched = _run_tiled_upscale(
                        single_image,
                        run_pid_once,
                        seed_base=batch_index * 100000,
                        latent_downscale=spec.latent_downscale,
                        profile=profile,
                    )
                    final = _resize_image(stitched, original_w * factor, original_h * factor)
                    outputs.append(final)
            _log_cuda_peak_memory("PiD Upscale")
        except Exception as exc:
            _free_cuda_memory(aggressive=True)
            infer_size = (original_h * factor, original_w * factor)
            context = (
                f"{exc}\n"
                f"PiD Upscale context: input={original_w}x{original_h}, output={original_w * factor}x{original_h * factor}, "
                f"backbone={backbone}, pid_ckpt_type={pid_ckpt_type}, checkpoint={spec.diffusion_filename}, "
                f"tile={profile.tile_size}, overlap={profile.tile_overlap}, factor={factor}x, "
                f"caption_chars={len(caption)}, strength={strength}, sigma={sigma:g}."
            )
            raise _format_pid_runtime_error(
                RuntimeError(context),
                infer_size,
                f"{backbone}/{pid_ckpt_type}/{spec.diffusion_filename}",
                PID_UPSCALE_NATIVE_SCALE,
            ) from exc
        finally:
            del vae
            _free_cuda_memory(aggressive=True)

        return (torch.stack(outputs, dim=0).contiguous(),)


NODE_CLASS_MAPPINGS = {
    "PiDUpscale": PiDUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDUpscale": "PiD Upscale",
}
