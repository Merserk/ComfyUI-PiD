"""
ComfyUI-PiD

Experimental ComfyUI nodes for NVIDIA PiD (Pixel Diffusion Decoder).

PiD is not a native ComfyUI VAE object. NVIDIA's inference path needs:
    latent + native decoder/baseline image + sigma + caption -> PiD -> final image
So the decode node accepts a LATENT plus either a matching ComfyUI VAE or a
pre-decoded baseline IMAGE, then returns IMAGE.
"""

from __future__ import annotations

import contextlib
import gc
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Optional, Tuple

import torch

try:
    import comfy.model_management as model_management
except Exception:  # pragma: no cover - only available inside ComfyUI
    model_management = None

try:
    import folder_paths  # ComfyUI model path helper
except Exception:  # pragma: no cover - only available inside ComfyUI
    folder_paths = None

NODE_DIR = Path(__file__).resolve().parent
# Expected layout: ComfyUI/custom_nodes/ComfyUI-PiD-ZImage/pid_decode.py
COMFYUI_DIR = NODE_DIR.parent.parent
VENDOR_DIR = NODE_DIR / "vendor"
DEFAULT_PID_DIR = VENDOR_DIR / "PiD"
DEFAULT_PID_MODEL_DIR = COMFYUI_DIR / "models" / "nvidia_pid"
PID_REPO_URL = "https://github.com/nv-tlabs/PiD.git"
PID_ZIP_URL = "https://github.com/nv-tlabs/PiD/archive/refs/heads/main.zip"
HF_REPO_ID = "nvidia/PiD"
AE_REL_PATH = "checkpoints/ae.safetensors"
AE_FILENAME = "ae.safetensors"


@dataclass(frozen=True)
class PiDCheckpoint:
    experiment: str
    relpath: str
    scale: int


@dataclass(frozen=True)
class PiDBackbone:
    label: str
    registry_key: str
    latent_channels: int
    default_scale: int
    latent_downscale: int
    ckpt_types: Tuple[str, ...]
    aux_files: Tuple[str, ...] = ()
    needs_flux_ae: bool = False


@dataclass(frozen=True)
class PiDHFSnapshot:
    repo_id: str
    required_files: Tuple[str, ...]


# Mirrors NVIDIA's official pid/_src/inference/checkpoint_registry.py.
PID_CKPTS: Dict[Tuple[str, str], PiDCheckpoint] = {
    ("flux", "2k"): PiDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        relpath="checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
    ("flux2", "2k"): PiDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux2_distill_4step",
        relpath="checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
    ("sd3", "2k"): PiDCheckpoint(
        experiment="PiD_res2k_sr4x_official_sd3_distill_4step",
        relpath="checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
    ("zimage", "2k"): PiDCheckpoint(
        experiment="PiD_res2k_sr4x_official_flux_distill_4step",
        relpath="checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
    ("rae", "2k"): PiDCheckpoint(
        experiment="PiD_res2k_sr4x_official_dinov2_distill_4step",
        relpath="checkpoints/PiD_res2k_sr4x_official_dinov2_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
    ("scale_rae", "2k"): PiDCheckpoint(
        experiment="PiD_res2k_sr8x_official_siglip_distill_4step",
        relpath="checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step/model_ema_bf16.pth",
        scale=8,
    ),
    ("flux", "2kto4k"): PiDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_flux_distill_4step",
        relpath="checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
    ("flux2", "2kto4k"): PiDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_flux2_distill_4step",
        relpath="checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
    ("sd3", "2kto4k"): PiDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_sd3_distill_4step",
        relpath="checkpoints/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
    ("zimage", "2kto4k"): PiDCheckpoint(
        experiment="PiD_res2kto4k_sr4x_official_flux_distill_4step",
        relpath="checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
        scale=4,
    ),
}


PID_BACKBONES: Dict[str, PiDBackbone] = {
    "zimage": PiDBackbone("Z-Image", "zimage", 16, 4, 8, ("2k", "2kto4k"), needs_flux_ae=True),
    "flux": PiDBackbone("Flux", "flux", 16, 4, 8, ("2k", "2kto4k"), needs_flux_ae=True),
    "flux2": PiDBackbone(
        "Flux2",
        "flux2",
        128,
        4,
        16,
        ("2k", "2kto4k"),
        aux_files=("checkpoints/flux2_ae.safetensors",),
    ),
    "sd3": PiDBackbone(
        "SD3",
        "sd3",
        16,
        4,
        8,
        ("2k", "2kto4k"),
        aux_files=("checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors",),
    ),
    "dinov2": PiDBackbone(
        "DINOv2 RAE",
        "rae",
        768,
        4,
        16,
        ("2k",),
        aux_files=(
            "checkpoints/rae/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt",
            "checkpoints/rae/stats/dinov2/wReg_base/imagenet1k_512/stat.pt",
        ),
    ),
    "siglip": PiDBackbone(
        "SigLIP Scale-RAE",
        "scale_rae",
        1152,
        8,
        16,
        ("2k",),
        aux_files=(
            "checkpoints/scale_rae/decoder/XL_decoder_config.json",
            "checkpoints/scale_rae/decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt",
        ),
    ),
}

BACKBONE_CHOICES = list(PID_BACKBONES.keys())
SEQUENTIAL_OFFLOAD_CHOICES = [
    "auto_low_vram",
    "disabled",
    "sequential_blocks",
    "sequential_blocks_aggressive",
]
PID_WEIGHT_PRECISION_CHOICES = ["fp32_compatible", "bf16_weights_experimental"]
_PIXEL_PATCH_SIZE = 16

GEMMA_SNAPSHOT = PiDHFSnapshot(
    repo_id="Efficient-Large-Model/gemma-2-2b-it",
    required_files=(
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ),
)
DINOv2_SNAPSHOT = PiDHFSnapshot(
    repo_id="facebook/dinov2-with-registers-base",
    required_files=("config.json", "model.safetensors", "preprocessor_config.json"),
)
SIGLIP_SNAPSHOT = PiDHFSnapshot(
    repo_id="google/siglip2-so400m-patch14-224",
    required_files=(
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ),
)

_MODEL_CACHE: Dict[Tuple[object, ...], object] = {}


@dataclass(frozen=True)
class PiDMemoryPlan:
    requested_mode: str
    pid_weight_precision: str
    infer_image_size: Tuple[int, int]
    total_pixel_patches: int
    pixel_chunk_patches: int
    chunk_pixel_blocks: bool
    cpu_cache_pixel_positions: bool
    skip_vae_encoder: bool
    offload_patch_blocks: bool
    offload_pixel_blocks: bool


class PiDNodeError(RuntimeError):
    pass


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def _python_executable() -> str:
    return sys.executable or "python"


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise PiDNodeError("Command failed:\n" + " ".join(cmd) + "\n\n" + proc.stdout)
    return proc.stdout.strip()


def _free_cuda_memory(aggressive: bool = False) -> None:
    """Best-effort cleanup for running PiD after a large ComfyUI model.

    ComfyUI often keeps the Z-Image UNet/TE/VAE staged for dynamic VRAM. PiD is
    another 1.36B parameter model, so on 16GB cards it helps to decode the native
    baseline image first, then release Comfy's loaded models before PiD inference.
    """
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


def _trim_cuda_cache() -> None:
    """Release unused CUDA allocator cache without synchronizing live work."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
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


def resolve_pid_memory_plan(
    sequential_offload: str,
    pid_weight_precision: str,
    pixel_chunk_patches: int,
    infer_image_size: Tuple[int, int],
    total_vram_gb: Optional[float] = None,
    supplies_lq_latent: bool = True,
) -> PiDMemoryPlan:
    mode = str(sequential_offload or "auto_low_vram").strip().lower()
    mode = {"none": "disabled", "off": "disabled"}.get(mode, mode)
    if mode not in SEQUENTIAL_OFFLOAD_CHOICES:
        raise PiDNodeError(
            f"Unknown sequential_offload={mode!r}; expected one of {SEQUENTIAL_OFFLOAD_CHOICES}"
        )

    precision = str(pid_weight_precision or "fp32_compatible").strip().lower()
    if precision not in PID_WEIGHT_PRECISION_CHOICES:
        raise PiDNodeError(
            f"Unknown pid_weight_precision={precision!r}; expected one of {PID_WEIGHT_PRECISION_CHOICES}"
        )

    try:
        requested_chunk_patches = int(pixel_chunk_patches)
    except Exception as exc:
        raise PiDNodeError(f"pixel_chunk_patches must be an integer, got {pixel_chunk_patches!r}") from exc
    if requested_chunk_patches < 0:
        raise PiDNodeError("pixel_chunk_patches must be zero (automatic) or a positive integer.")

    h, w = int(infer_image_size[0]), int(infer_image_size[1])
    if h <= 0 or w <= 0:
        raise PiDNodeError(f"infer_image_size must be positive, got {(h, w)!r}")
    total_pixel_patches = max(1, (h // _PIXEL_PATCH_SIZE) * (w // _PIXEL_PATCH_SIZE))

    low_vram = mode in ("auto_low_vram", "sequential_blocks_aggressive")
    balanced = mode == "sequential_blocks"
    capacity_gb = _vram_total_gb() if total_vram_gb is None else float(total_vram_gb)
    if requested_chunk_patches > 0:
        chunk_patches = min(total_pixel_patches, requested_chunk_patches)
    elif low_vram:
        automatic_chunk = 4096 if capacity_gb <= 16.5 else 8192
        chunk_patches = min(total_pixel_patches, automatic_chunk)
    elif balanced:
        chunk_patches = min(total_pixel_patches, 8192)
    else:
        chunk_patches = 0

    chunk_pixel_blocks = 0 < chunk_patches < total_pixel_patches
    return PiDMemoryPlan(
        requested_mode=mode,
        pid_weight_precision=precision,
        infer_image_size=(h, w),
        total_pixel_patches=total_pixel_patches,
        pixel_chunk_patches=chunk_patches,
        chunk_pixel_blocks=chunk_pixel_blocks,
        cpu_cache_pixel_positions=low_vram and chunk_pixel_blocks,
        skip_vae_encoder=bool(supplies_lq_latent) and mode != "disabled",
        offload_patch_blocks=mode != "disabled",
        offload_pixel_blocks=low_vram,
    )


def _log_pid_memory_plan(plan: PiDMemoryPlan) -> None:
    capacity_gb = _vram_total_gb()
    h, w = plan.infer_image_size
    print(
        "[ComfyUI-PiD] memory plan: "
        f"mode={plan.requested_mode}, image={w}x{h}, gpu_capacity={capacity_gb:.1f} GiB, "
        f"pixel_patches={plan.total_pixel_patches}, chunk_patches={plan.pixel_chunk_patches or 'disabled'}, "
        f"precision={plan.pid_weight_precision}, cpu_pos_cache={plan.cpu_cache_pixel_positions}, "
        f"skip_vae={plan.skip_vae_encoder}, patch_offload={plan.offload_patch_blocks}, "
        f"pixel_offload={plan.offload_pixel_blocks}",
        flush=True,
    )


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


def _format_pid_runtime_error(exc: BaseException, infer_image_size: Tuple[int, int], ckpt_type: str, scale: int) -> PiDNodeError:
    msg = str(exc)
    lower = msg.lower()
    h, w = int(infer_image_size[0]), int(infer_image_size[1])
    advice = (
        f"PiD inference failed while generating {w}x{h} with pid_ckpt_type={ckpt_type!r}, scale={scale}.\n\n"
        "Most common cause: VRAM/allocator pressure. Large outputs should normally use "
        "auto_low_vram so live pixel-block tensors are chunked before they can spill into shared memory.\n\n"
        "Try these first:\n"
        "  1. Set sequential_offload=auto_low_vram and pixel_chunk_patches=0.\n"
        "  2. If memory is still tight, set pixel_chunk_patches=2048.\n"
        "  3. Keep unload_comfy_before_pid=true and aggressive_cleanup=true.\n"
        "  4. Lower scale or use a smaller base latent.\n"
        "  5. Restart ComfyUI after a CUDA allocator crash.\n\n"
        f"Original error: {msg}"
    )
    if "out of memory" in lower or "cudamallocasync" in lower or "internal assert" in lower or "cuda" in lower:
        return PiDNodeError(advice)
    return PiDNodeError(f"PiD inference failed. Original error: {msg}")


def _resolve_pid_dir(pid_source_dir: str = "") -> Path:
    # Priority: explicit UI path -> environment variable -> bundled vendor path.
    if pid_source_dir and pid_source_dir.strip():
        return Path(pid_source_dir.strip()).expanduser().resolve()
    env_path = os.environ.get("PID_REPO_DIR") or os.environ.get("COMFYUI_PID_REPO_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_PID_DIR.resolve()


def _resolve_pid_model_dir() -> Path:
    """Return the ComfyUI-managed root for NVIDIA PiD weights and assets."""
    if folder_paths is not None:
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            return (Path(models_dir) / "nvidia_pid").expanduser().resolve()
    return DEFAULT_PID_MODEL_DIR.resolve()


def _migrate_legacy_checkpoints(model_dir: Path) -> List[str]:
    """Move bundled legacy checkpoints into ComfyUI's shared model directory."""
    legacy_dir = DEFAULT_PID_DIR / "checkpoints"
    target_dir = model_dir / "checkpoints"
    messages: List[str] = []
    if not legacy_dir.is_dir():
        return messages

    for src in sorted(path for path in legacy_dir.rglob("*") if path.is_file()):
        relpath = src.relative_to(legacy_dir)
        target = target_dir / relpath
        if target.exists():
            messages.append(f"Legacy PiD asset kept because destination exists: {src}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
        messages.append(f"Moved legacy PiD asset: {src} -> {target}")

    for directory in sorted((path for path in legacy_dir.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        legacy_dir.rmdir()
    except OSError:
        pass
    return messages


def _pid_is_present(pid_dir: Path) -> bool:
    return (pid_dir / "pid" / "_src" / "utils" / "model_loader.py").is_file()


def _ensure_pid_source(pid_dir: Path, allow_download: bool = True) -> List[str]:
    messages: List[str] = []
    if _pid_is_present(pid_dir):
        if str(pid_dir) not in sys.path:
            sys.path.insert(0, str(pid_dir))
        messages.append(f"PiD source found: {pid_dir}")
        return messages

    if not allow_download:
        raise PiDNodeError(
            f"PiD source was not found at {pid_dir}.\n"
            "Set auto_download=true, or set PID_REPO_DIR / COMFYUI_PID_REPO_DIR "
            "to your local PiD checkout."
        )

    pid_dir.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git:
        messages.append(f"Cloning PiD source into {pid_dir} ...")
        _run([git, "clone", "--depth", "1", PID_REPO_URL, str(pid_dir)])
    else:
        messages.append("git not found; downloading PiD main.zip instead ...")
        zip_path = pid_dir.parent / "PiD-main.zip"
        urllib.request.urlretrieve(PID_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(pid_dir.parent)
        extracted = pid_dir.parent / "PiD-main"
        if pid_dir.exists():
            shutil.rmtree(pid_dir)
        extracted.rename(pid_dir)
        zip_path.unlink(missing_ok=True)

    if not _pid_is_present(pid_dir):
        raise PiDNodeError(f"PiD source download finished but expected files were not found under {pid_dir}")

    if str(pid_dir) not in sys.path:
        sys.path.insert(0, str(pid_dir))
    messages.append("PiD source is ready.")
    return messages


def _ensure_hf_download_available() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception as exc:
        raise PiDNodeError(
            "Missing dependency: huggingface_hub.\n"
            "Install this custom node's requirements, or manually run:\n"
            f"{_python_executable()} -m pip install \"huggingface-hub>=1.0\""
        ) from exc


def _hf_snapshot_dir(model_dir: Path, snapshot: PiDHFSnapshot) -> Path:
    return model_dir / "huggingface" / Path(snapshot.repo_id)


def _missing_hf_snapshot_files(model_dir: Path, snapshot: PiDHFSnapshot) -> List[str]:
    snapshot_dir = _hf_snapshot_dir(model_dir, snapshot)
    return [relpath for relpath in snapshot.required_files if not (snapshot_dir / relpath).is_file()]


def _required_hf_snapshots(backbone: str) -> Tuple[PiDHFSnapshot, ...]:
    if backbone not in PID_BACKBONES:
        raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
    snapshots = [GEMMA_SNAPSHOT]
    if backbone == "dinov2":
        snapshots.append(DINOv2_SNAPSHOT)
    elif backbone == "siglip":
        snapshots.append(SIGLIP_SNAPSHOT)
    return tuple(snapshots)


def _download_hf_snapshot(model_dir: Path, snapshot: PiDHFSnapshot) -> str:
    snapshot_dir = _hf_snapshot_dir(model_dir, snapshot)
    missing = _missing_hf_snapshot_files(model_dir, snapshot)
    if not missing:
        return f"Hugging Face snapshot already exists: {snapshot_dir}"

    _ensure_hf_download_available()
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=snapshot.repo_id,
        local_dir=str(snapshot_dir),
        allow_patterns=list(snapshot.required_files),
        token=token,
    )
    missing = _missing_hf_snapshot_files(model_dir, snapshot)
    if missing:
        missing_text = "\n".join(f"  - {snapshot_dir / relpath}" for relpath in missing)
        raise PiDNodeError(
            f"Hugging Face snapshot download finished but required files are missing for {snapshot.repo_id}:\n"
            f"{missing_text}"
        )
    return f"Downloaded Hugging Face snapshot: {snapshot.repo_id} -> {snapshot_dir}"


def _ensure_hf_snapshot(model_dir: Path, snapshot: PiDHFSnapshot, allow_download: bool = True) -> Path:
    snapshot_dir = _hf_snapshot_dir(model_dir, snapshot)
    missing = _missing_hf_snapshot_files(model_dir, snapshot)
    if not missing:
        return snapshot_dir
    if allow_download:
        try:
            _download_hf_snapshot(model_dir, snapshot)
            return snapshot_dir
        except Exception as exc:
            if isinstance(exc, PiDNodeError):
                raise
            raise PiDNodeError(
                f"Could not download required Hugging Face snapshot {snapshot.repo_id} to:\n"
                f"  {snapshot_dir}\n\n"
                f"Original download error: {exc}"
            ) from exc

    missing_text = "\n".join(f"  - {snapshot_dir / relpath}" for relpath in missing)
    raise PiDNodeError(
        f"PiD requires the local Hugging Face snapshot {snapshot.repo_id} at:\n"
        f"  {snapshot_dir}\n\n"
        "auto_download=false enables strict local-only mode. Download or copy these missing files:\n"
        f"{missing_text}"
    )


def _ensure_hf_snapshots(model_dir: Path, backbone: str, allow_download: bool = True) -> None:
    for snapshot in _required_hf_snapshots(backbone):
        _ensure_hf_snapshot(model_dir, snapshot, allow_download=allow_download)



def _candidate_comfy_ae_paths() -> List[Path]:
    """Return likely local Flux/Z-Image AE VAE paths from ComfyUI.

    NVIDIA PiD's own config instantiates a Flux VAE encoder/tokenizer. Most
    ComfyUI Z-Image workflows already have the exact same file under
    ComfyUI/models/vae/.
    """
    candidates: List[Path] = []

    # Ask ComfyUI's model registry first. This respects extra_model_paths.yaml.
    if folder_paths is not None:
        try:
            get_full_path = getattr(folder_paths, "get_full_path", None)
            if get_full_path is not None:
                p = get_full_path("vae", AE_FILENAME)
                if p:
                    candidates.append(Path(p))
        except Exception:
            pass
        try:
            get_full_path_or_raise = getattr(folder_paths, "get_full_path_or_raise", None)
            if get_full_path_or_raise is not None:
                p = get_full_path_or_raise("vae", AE_FILENAME)
                if p:
                    candidates.append(Path(p))
        except Exception:
            pass
        try:
            for name in folder_paths.get_filename_list("vae"):
                if Path(name).name.lower() == AE_FILENAME:
                    try:
                        p = folder_paths.get_full_path("vae", name)
                        if p:
                            candidates.append(Path(p))
                    except Exception:
                        pass
        except Exception:
            pass

    # Fallback to standard ComfyUI folder locations.
    vae_dir = COMFYUI_DIR / "models" / "vae"
    candidates.append(vae_dir / AE_FILENAME)
    try:
        candidates.extend(vae_dir.glob(f"**/{AE_FILENAME}"))
    except Exception:
        pass

    # Deduplicate while preserving order.
    seen = set()
    out: List[Path] = []
    for c in candidates:
        try:
            cr = c.expanduser().resolve()
        except Exception:
            continue
        if cr in seen:
            continue
        seen.add(cr)
        out.append(cr)
    return out


def _copy_local_ae_to_pid(model_dir: Path) -> Optional[str]:
    target = model_dir / AE_REL_PATH
    if target.is_file():
        return f"Flux AE VAE already exists: {target}"

    for src in _candidate_comfy_ae_paths():
        if src.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            return f"Copied Flux AE VAE from {src} to {target}"
    return None


def _download_ae_safetensors(model_dir: Path) -> str:
    target = model_dir / AE_REL_PATH
    if target.is_file():
        return f"Flux AE VAE already exists: {target}"

    _ensure_hf_download_available()
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=AE_REL_PATH,
        local_dir=str(model_dir),
        token=token,
    )
    if not target.is_file():
        raise PiDNodeError(f"Hugging Face download finished but Flux AE VAE is missing: {target}")
    return f"Downloaded Flux AE VAE: {target}"


def _ensure_ae_safetensors(model_dir: Path, allow_download: bool = True) -> Path:
    """Ensure PiD's required Flux AE exists in the shared model directory."""
    target = model_dir / AE_REL_PATH
    if target.is_file():
        return target

    copied = _copy_local_ae_to_pid(model_dir)
    if copied and target.is_file():
        return target

    if allow_download:
        try:
            _download_ae_safetensors(model_dir)
            return target
        except Exception as exc:
            local_candidates = "\n".join(f"  - {p}" for p in _candidate_comfy_ae_paths()) or "  - none found"
            raise PiDNodeError(
                "PiD requires Flux's AE VAE at:\n"
                f"  {target}\n\n"
                "I could not find/copy it from ComfyUI and could not download it automatically.\n"
                "Manual fix:\n"
                f"  1. Make sure {AE_FILENAME} exists in ComfyUI/models/vae/\n"
                f"  2. Copy it to {target}\n\n"
                "Local paths checked:\n"
                f"{local_candidates}\n\n"
                f"Original download/copy error: {exc}"
            ) from exc

    local_candidates = "\n".join(f"  - {p}" for p in _candidate_comfy_ae_paths()) or "  - none found"
    raise PiDNodeError(
        "PiD requires Flux's AE VAE at:\n"
        f"  {target}\n\n"
                "Set auto_download=true, or manually copy:\n"
                f"  ComfyUI/models/vae/{AE_FILENAME}\n"
        "to:\n"
        f"  {target}\n\n"
        "Local paths checked:\n"
        f"{local_candidates}"
    )


def _download_hf_file(model_dir: Path, relpath: str) -> str:
    target = model_dir / relpath
    if target.is_file():
        return f"Asset already exists: {target}"

    _ensure_hf_download_available()
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=relpath,
        local_dir=str(model_dir),
        token=token,
    )
    if not target.is_file():
        raise PiDNodeError(f"Hugging Face download finished but asset is missing: {target}")
    return f"Downloaded: {target}"


def _checkpoint_for(backbone: str, ckpt_type: str) -> PiDCheckpoint:
    if backbone not in PID_BACKBONES:
        raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
    backbone_info = PID_BACKBONES[backbone]
    if ckpt_type not in backbone_info.ckpt_types:
        supported = ", ".join(backbone_info.ckpt_types)
        raise PiDNodeError(
            f"{backbone_info.label} does not ship a {ckpt_type!r} checkpoint in NVIDIA's registry. "
            f"Supported pid_ckpt_type values: {supported}."
        )
    try:
        return PID_CKPTS[(backbone_info.registry_key, ckpt_type)]
    except KeyError as exc:
        raise PiDNodeError(f"No PiD checkpoint registered for backbone={backbone!r}, pid_ckpt_type={ckpt_type!r}") from exc


def _ensure_checkpoint(model_dir: Path, backbone: str, ckpt_type: str, allow_download: bool = True) -> Path:
    ckpt = _checkpoint_for(backbone, ckpt_type)
    ckpt_path = model_dir / ckpt.relpath
    if ckpt_path.is_file():
        return ckpt_path
    if not allow_download:
        raise PiDNodeError(
            f"PiD checkpoint is missing: {ckpt_path}\n"
            "Set auto_download=true or download the matching file from nvidia/PiD."
        )
    _download_hf_file(model_dir, ckpt.relpath)
    return ckpt_path


def _ensure_backbone_assets(model_dir: Path, backbone: str, allow_download: bool = True) -> None:
    info = PID_BACKBONES[backbone]
    if info.needs_flux_ae:
        _ensure_ae_safetensors(model_dir, allow_download=allow_download)

    missing = [relpath for relpath in info.aux_files if not (model_dir / relpath).is_file()]
    if missing and not allow_download:
        missing_text = "\n".join(f"  - {model_dir / relpath}" for relpath in missing)
        raise PiDNodeError(
            f"{info.label} PiD support needs these extra asset files:\n"
            f"{missing_text}\n\n"
            "Set auto_download=true or download them from nvidia/PiD."
        )
    for relpath in missing:
        _download_hf_file(model_dir, relpath)
    _ensure_hf_snapshots(model_dir, backbone, allow_download=allow_download)


def _redirect_pid_text_encoder(model_dir: Path) -> None:
    """Point NVIDIA PiD's built-in Gemma mapping at the local snapshot."""
    try:
        from pid._src.models import pixeldit_model
    except Exception as exc:
        raise PiDNodeError(f"Could not configure NVIDIA PiD's local text encoder path: {exc}") from exc
    pixeldit_model._TEXT_ENCODER_DICT["gemma-2-2b-it"] = _hydra_path(_hf_snapshot_dir(model_dir, GEMMA_SNAPSHOT))


def _hydra_path(path: Path) -> str:
    """Format an absolute local path for NVIDIA PiD's Hydra overrides."""
    return path.expanduser().resolve().as_posix()


def _pid_asset_experiment_opts(model_dir: Path, backbone: str) -> List[str]:
    """Redirect NVIDIA PiD's config-relative tokenizer assets to ComfyUI models."""
    def override(field: str, relpath: str) -> str:
        return f"++model.config.tokenizer.{field}={_hydra_path(model_dir / relpath)}"

    if backbone in ("zimage", "flux"):
        return [override("vae_pth", AE_REL_PATH)]
    if backbone == "flux2":
        return [override("vae_pth", "checkpoints/flux2_ae.safetensors")]
    if backbone == "sd3":
        return [override("vae_pth", "checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors")]
    if backbone == "dinov2":
        return [
            override("pretrained_path", f"huggingface/{DINOv2_SNAPSHOT.repo_id}"),
            override("pretrained_decoder_path", "checkpoints/rae/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt"),
            override("normalization_stat_path", "checkpoints/rae/stats/dinov2/wReg_base/imagenet1k_512/stat.pt"),
        ]
    if backbone == "siglip":
        return [
            override("pretrained_path", f"huggingface/{SIGLIP_SNAPSHOT.repo_id}"),
            override("pretrained_decoder_path", "checkpoints/scale_rae/decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt"),
            override("decoder_config_path", "checkpoints/scale_rae/decoder/XL_decoder_config.json"),
        ]
    raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")


def _import_pid_loader(pid_dir: Path):
    if str(pid_dir) not in sys.path:
        sys.path.insert(0, str(pid_dir))
    try:
        from pid._src.utils.model_loader import load_model_from_checkpoint
    except Exception as exc:
        raise PiDNodeError(
            "Could not import NVIDIA PiD. Usually this means PiD dependencies are not installed.\n\n"
            "Install the packages in requirements.txt, then restart ComfyUI.\n\n"
            f"Original import error: {exc}"
        ) from exc
    return load_model_from_checkpoint


def _load_pid_model(
    pid_dir: Path,
    model_dir: Path,
    backbone: str,
    ckpt_type: str,
    checkpoint_path: Path,
    dtype_choice: str,
    load_ema_to_reg: bool = False,
    skip_vae_encoder: bool = False,
    pid_weight_precision: str = "fp32_compatible",
):
    precision = str(pid_weight_precision or "fp32_compatible").strip().lower()
    if precision not in PID_WEIGHT_PRECISION_CHOICES:
        raise PiDNodeError(
            f"Unknown pid_weight_precision={precision!r}; expected one of {PID_WEIGHT_PRECISION_CHOICES}"
        )
    experiment = _checkpoint_for(backbone, ckpt_type).experiment
    config_file = "pid/_src/configs/pid/config.py"
    cache_key = (
        str(pid_dir),
        str(model_dir),
        backbone,
        ckpt_type,
        str(checkpoint_path),
        experiment,
        bool(load_ema_to_reg),
        dtype_choice,
        bool(skip_vae_encoder),
        precision,
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not torch.cuda.is_available():
        raise PiDNodeError("PiD's official model loader instantiates the decoder on CUDA. CUDA GPU is required.")

    load_model_from_checkpoint = _import_pid_loader(pid_dir)
    _redirect_pid_text_encoder(model_dir)
    experiment_opts = _pid_asset_experiment_opts(model_dir, backbone)
    if skip_vae_encoder:
        experiment_opts = [*experiment_opts, "model.config.tokenizer=null"]

    # PiD's config helper expects config_file to be relative to the PiD repo root.
    with _pushd(pid_dir):
        model, _config = load_model_from_checkpoint(
            experiment_name=experiment,
            checkpoint_path=str(checkpoint_path),
            config_file=config_file,
            enable_fsdp=False,
            experiment_opts=experiment_opts,
            strict=False,
            load_ema_to_reg=load_ema_to_reg,
        )
    model.eval()
    if precision == "bf16_weights_experimental":
        net = getattr(model, "net", None)
        if net is None:
            raise PiDNodeError("Loaded PiD model has no network to cast to BF16 weights.")
        net.to(dtype=torch.bfloat16)
        _trim_cuda_cache()
    _MODEL_CACHE[cache_key] = model
    return model


def _remove_model_from_cache(model: object) -> None:
    """Drop a specific PiD model from the private cache by object identity."""
    for key, cached_model in list(_MODEL_CACHE.items()):
        if cached_model is model:
            _MODEL_CACHE.pop(key, None)


def _unload_pid_model(model: object, aggressive: bool = True) -> None:
    """Move PiD off CUDA and clear the private cache after a decode run."""
    _remove_model_from_cache(model)
    try:
        to = getattr(model, "to", None)
        if callable(to):
            to("cpu")
    except Exception:
        pass
    _free_cuda_memory(aggressive=aggressive)


def _pid_lq_condition_type(model: object) -> str:
    config = getattr(model, "config", None)
    return str(getattr(config, "lq_condition_type", "latent") or "latent").strip().lower()


def _pid_uses_lq_image(model: object) -> bool:
    return _pid_lq_condition_type(model) in ("image", "image_latent")


def _pid_uses_lq_latent(model: object) -> bool:
    return _pid_lq_condition_type(model) in ("latent", "image_latent")


def _move_module_like_to_cpu(obj: object) -> None:
    if obj is None:
        return
    nested_model = getattr(obj, "model", None)
    nested_nested_model = getattr(nested_model, "model", None)
    for target in (obj, nested_model, nested_nested_model):
        if target is None:
            continue
        try:
            to = getattr(target, "to", None)
            if callable(to):
                to("cpu")
        except Exception:
            pass


def _offload_unused_pid_conditioners(model: object, include_vae: bool = True) -> None:
    _move_module_like_to_cpu(getattr(model, "text_encoder", None))
    null_caption = getattr(model, "_null_caption_embs", None)
    if isinstance(null_caption, torch.Tensor):
        try:
            setattr(model, "_null_caption_embs", null_caption.cpu())
        except Exception:
            pass
    if include_vae:
        _move_module_like_to_cpu(getattr(model, "vae_encoder", None))
    _trim_cuda_cache()


class _LazyLQFeatureList:
    """Compute PiD LQ projection heads just-in-time instead of keeping all heads."""

    def __init__(self, tokens: torch.Tensor, output_heads, pit_head):
        self.tokens = tokens
        self.output_heads = output_heads
        self.pit_head = pit_head

    def __len__(self) -> int:
        return len(self.output_heads) + (1 if self.pit_head is not None else 0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(idx)
        if idx < len(self.output_heads):
            return self.output_heads[idx](self.tokens)
        if self.pit_head is None:
            raise IndexError(idx)
        return self.pit_head(self.tokens)


def _enable_lazy_lq_projection(model: object) -> None:
    """Patch PiD's LQ projector so 4K runs do not cache every block feature."""
    net = getattr(model, "net", None)
    lq_proj = getattr(net, "lq_proj", None)
    if lq_proj is None or getattr(lq_proj, "_comfy_pid_lazy_lq", False):
        return
    if getattr(net, "_cp_group", None) is not None:
        return
    needed = (
        "image_conv",
        "latent_proj",
        "merge",
        "output_heads",
        "pit_head",
        "_align_image_to_patch_grid",
        "_align_latent_to_patch_grid",
    )
    if any(not hasattr(lq_proj, name) for name in needed):
        return

    def lazy_forward(
        lq_video_or_image=None,
        lq_latent=None,
        target_pH: int = 0,
        target_pW: int = 0,
    ):
        if target_pH <= 0 or target_pW <= 0:
            raise AssertionError("Must provide target_pH and target_pW")

        features = []
        if lq_proj.image_conv is not None and lq_video_or_image is not None:
            features.append(lq_proj._align_image_to_patch_grid(lq_video_or_image, target_pH, target_pW))
        if lq_proj.latent_proj is not None and lq_latent is not None:
            features.append(lq_proj._align_latent_to_patch_grid(lq_latent, target_pH, target_pW))

        if len(features) == 2 and lq_proj.merge is not None:
            merged = lq_proj.merge(torch.cat(features, dim=1))
        elif len(features) == 1:
            merged = features[0]
        else:
            ref = lq_video_or_image if lq_video_or_image is not None else lq_latent
            if ref is None:
                raise PiDNodeError("PiD low-VRAM LQ projection received neither LQ image nor latent.")
            b, device, dtype = ref.shape[0], ref.device, ref.dtype
            hidden_dim = int(getattr(lq_proj, "hidden_dim", 0) or getattr(lq_proj, "out_dim", 0))
            tokens = torch.zeros(
                b,
                int(target_pH) * int(target_pW),
                hidden_dim,
                device=device,
                dtype=dtype,
            )
            return _LazyLQFeatureList(tokens, lq_proj.output_heads, lq_proj.pit_head)

        tokens = merged.flatten(2).transpose(1, 2)
        return _LazyLQFeatureList(tokens, lq_proj.output_heads, lq_proj.pit_head)

    lq_proj.forward = lazy_forward
    lq_proj._comfy_pid_lazy_lq = True


def _generate_samples_low_vram(model: object, data_batch: dict, **kwargs):
    """Run PiD with text/VAE offload and lazy LQ conditioning."""
    _enable_lazy_lq_projection(model)

    batch = dict(data_batch)
    condition_type = _pid_lq_condition_type(model)
    if condition_type == "latent":
        batch.pop("LQ_video_or_image", None)
    elif condition_type == "image":
        batch.pop("LQ_latent", None)

    original_encode = getattr(model, "_encode_text_raw", None)
    if not callable(original_encode):
        return model.generate_samples_from_batch(batch, **kwargs)

    def encode_then_offload(captions):
        text_encoder = getattr(model, "text_encoder", None)
        try:
            to = getattr(text_encoder, "to", None)
            if callable(to):
                to("cuda")
        except Exception:
            pass
        result = original_encode(captions)
        _offload_unused_pid_conditioners(model, include_vae="LQ_latent" in batch)
        return result

    try:
        setattr(model, "_encode_text_raw", encode_then_offload)
        return model.generate_samples_from_batch(batch, **kwargs)
    finally:
        try:
            setattr(model, "_encode_text_raw", original_encode)
        except Exception:
            pass


def _make_pid_data_batch(
    model: object,
    caption: str,
    sigma: float,
    latent_cpu: torch.Tensor,
    baseline_cpu: Optional[torch.Tensor],
    device: str,
) -> dict:
    # Free heavyweight PiD conditioners before moving LQ inputs onto CUDA. The
    # text encoder is moved back only for caption encoding inside the sampler.
    _offload_unused_pid_conditioners(model, include_vae=_pid_uses_lq_latent(model))
    batch = int(latent_cpu.shape[0])
    data_batch = {
        getattr(model.config, "input_caption_key", "caption"): [caption or ""] * batch,
        "degrade_sigma": torch.full((batch,), float(sigma), device=device, dtype=torch.float32),
    }
    if _pid_uses_lq_latent(model):
        data_batch["LQ_latent"] = latent_cpu.to(device=device, dtype=torch.bfloat16)
    if _pid_uses_lq_image(model):
        if baseline_cpu is None:
            raise PiDNodeError(
                "This PiD checkpoint requires LQ image conditioning. Connect baseline_image or a matching VAE."
            )
        data_batch["LQ_video_or_image"] = (baseline_cpu.to(device=device, dtype=torch.bfloat16) * 2.0) - 1.0
    return data_batch


def _infer_baseline_size_from_latent(samples: torch.Tensor, backbone: str) -> Tuple[int, int]:
    info = PID_BACKBONES.get(backbone, PID_BACKBONES["zimage"])
    return int(samples.shape[-2]) * int(info.latent_downscale), int(samples.shape[-1]) * int(info.latent_downscale)


def _baseline_cpu_and_size(
    samples: torch.Tensor,
    backbone: str,
    vae=None,
    baseline_image=None,
) -> Tuple[Optional[torch.Tensor], Tuple[int, int]]:
    if baseline_image is None and vae is None:
        return None, _infer_baseline_size_from_latent(samples, backbone)
    if baseline_image is None:
        baseline_01 = _decode_baseline_with_comfy_vae(vae, samples, backbone)
    else:
        baseline_01 = _comfy_image_to_bchw_01(baseline_image)
    if baseline_01.shape[0] != samples.shape[0]:
        raise PiDNodeError(
            f"Batch mismatch: latent batch={samples.shape[0]}, baseline batch={baseline_01.shape[0]}"
        )
    baseline_cpu = baseline_01.detach().to("cpu").contiguous()
    return baseline_cpu, (int(baseline_cpu.shape[-2]), int(baseline_cpu.shape[-1]))


def _apply_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def _enable_cpu_cached_pixel_positions(model: object, chunk_patches: int) -> None:
    """Patch the pixel embedder to keep full-image position data in system RAM."""
    net = getattr(model, "net", None)
    embedder = getattr(net, "pixel_embedder", None)
    if embedder is None:
        raise PiDNodeError("Loaded PiD model has no pixel embedder for low-VRAM position caching.")
    if getattr(embedder, "_comfy_pid_cpu_position_chunks", 0) == int(chunk_patches):
        return
    if not hasattr(embedder, "_fetch_pixel_pos_image") or not hasattr(embedder, "proj"):
        raise PiDNodeError("Loaded PiD pixel embedder is incompatible with low-VRAM position caching.")

    original_forward = getattr(embedder, "_comfy_pid_original_forward", None) or embedder.forward
    setattr(embedder, "_comfy_pid_original_forward", original_forward)
    chunk_patches = max(1, int(chunk_patches))

    def chunked_forward(self, inputs, img_height=None, img_width=None, patch_size=None):
        if inputs.dim() != 4:
            return original_forward(inputs, img_height=img_height, img_width=img_width, patch_size=patch_size)
        if img_height is None or img_width is None or patch_size is None:
            raise AssertionError("Need H, W, patch_size for image mode")

        b, _, h, w = inputs.shape
        if h != img_height or w != img_width:
            raise AssertionError("Input spatial size mismatch")
        if (h % patch_size) != 0 or (w % patch_size) != 0:
            raise AssertionError("H and W must be divisible by patch_size")

        hs, ws = h // patch_size, w // patch_size
        patch_count = hs * ws
        p2 = patch_size * patch_size
        pos_key = ("image_patches_cpu", h, w, patch_size, str(inputs.dtype))
        pos_patches = self._pos_cache.get(pos_key)
        if pos_patches is None:
            pos_full = self._fetch_pixel_pos_image(h, w, "cpu", inputs.dtype)
            pos_patches = pos_full.view(hs, patch_size, ws, patch_size, self.hidden_size_output)
            pos_patches = pos_patches.permute(0, 2, 1, 3, 4).contiguous()
            pos_patches = pos_patches.view(patch_count, p2, self.hidden_size_output)
            self._pos_cache[pos_key] = pos_patches
            self._pos_cache.pop(("image", h, w), None)

        projected = inputs.permute(0, 2, 3, 1).contiguous()
        projected = self.proj(projected)
        projected = projected.view(b, hs, patch_size, ws, patch_size, self.hidden_size_output)
        projected = projected.permute(0, 1, 3, 2, 4, 5).contiguous()
        projected = projected.view(b * patch_count, p2, self.hidden_size_output)
        output = None
        for batch_index in range(b):
            batch_offset = batch_index * patch_count
            for patch_start in range(0, patch_count, chunk_patches):
                patch_end = min(patch_count, patch_start + chunk_patches)
                flat_start = batch_offset + patch_start
                flat_end = batch_offset + patch_end
                pos = pos_patches[patch_start:patch_end].to(device=projected.device, dtype=inputs.dtype)
                chunk = projected[flat_start:flat_end] + pos
                if output is None:
                    output = chunk.new_empty(projected.shape)
                output[flat_start:flat_end] = chunk
        _trim_cuda_cache()
        return output

    embedder.forward = MethodType(chunked_forward, embedder)
    embedder._comfy_pid_cpu_position_chunks = chunk_patches


def _enable_chunked_pixel_blocks(model: object, chunk_patches: int) -> None:
    """Chunk pixel-block AdaLN and MLP while retaining one global attention pass."""
    net = getattr(model, "net", None)
    blocks = getattr(net, "pixel_blocks", None)
    if blocks is None:
        raise PiDNodeError("Loaded PiD model has no pixel blocks for low-VRAM chunking.")
    chunk_patches = max(1, int(chunk_patches))

    for block in blocks:
        if getattr(block, "_comfy_pid_chunk_patches", 0) == chunk_patches:
            continue
        needed = (
            "pixel_dim",
            "patch_size",
            "attn_dim",
            "adaLN_modulation",
            "norm1",
            "compress_to_attn",
            "attn",
            "expand_from_attn",
            "norm2",
            "mlp",
            "_fetch_pos",
        )
        if any(not hasattr(block, name) for name in needed):
            raise PiDNodeError("Loaded PiD pixel block is incompatible with exact low-VRAM chunking.")

        original_forward = getattr(block, "_comfy_pid_original_forward", None) or block.forward
        setattr(block, "_comfy_pid_original_forward", original_forward)

        def chunked_forward(
            self,
            x,
            s_cond,
            image_height,
            image_width,
            patch_size,
            mask=None,
            _chunk_patches=chunk_patches,
            _original_forward=original_forward,
        ):
            if getattr(self, "_cp_group", None) is not None:
                return _original_forward(x, s_cond, image_height, image_width, patch_size, mask)

            bl, p2, channels = x.shape
            if channels != self.pixel_dim:
                raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {channels}")
            if patch_size != self.patch_size:
                raise AssertionError("PiTBlock expects fixed patch_size")
            if p2 != patch_size * patch_size:
                raise AssertionError("Token count per patch must equal patch_size^2")
            if (image_height % patch_size) != 0 or (image_width % patch_size) != 0:
                raise AssertionError("H and W must be divisible by patch_size")

            hs, ws = image_height // patch_size, image_width // patch_size
            local_patch_count = hs * ws
            if bl % local_patch_count != 0:
                raise AssertionError("Total sequences must be a multiple of local patch count")
            batch = bl // local_patch_count

            normalized = torch.empty_like(x)
            for start in range(0, bl, _chunk_patches):
                end = min(bl, start + _chunk_patches)
                params = self.adaLN_modulation(s_cond[start:end]).view(end - start, p2, 6 * self.pixel_dim)
                shift_msa, scale_msa, _, _, _, _ = torch.chunk(params, 6, dim=-1)
                normalized[start:end] = _apply_adaln(self.norm1(x[start:end]), shift_msa, scale_msa)

            compressed = self.compress_to_attn(normalized.view(bl, p2 * self.pixel_dim))
            del normalized
            pos = self._fetch_pos(hs, ws, x.device)
            attention = self.attn(compressed.view(batch, local_patch_count, self.attn_dim), pos, mask)
            del compressed
            expanded = self.expand_from_attn(attention.reshape(bl, self.attn_dim)).view(bl, p2, self.pixel_dim)
            del attention
            output = torch.empty_like(x)
            for start in range(0, bl, _chunk_patches):
                end = min(bl, start + _chunk_patches)
                params = self.adaLN_modulation(s_cond[start:end]).view(end - start, p2, 6 * self.pixel_dim)
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(params, 6, dim=-1)
                residual = x[start:end] + gate_msa * expanded[start:end]
                mlp = self.mlp(_apply_adaln(self.norm2(residual), shift_mlp, scale_mlp))
                output[start:end] = residual + gate_mlp * mlp
            return output

        block.forward = MethodType(chunked_forward, block)
        block._comfy_pid_chunk_patches = chunk_patches



class _SequentialBlockOffloader:
    """Move selected PiD block stacks to CUDA only for their forward calls."""

    def __init__(
        self,
        model,
        mode: str,
        device: str = "cuda",
        offload_patch_blocks: bool = True,
        offload_pixel_blocks: bool = False,
    ):
        self.model = model
        self.mode = str(mode or "disabled").strip().lower()
        self.device = device
        self.handles = []
        self.blocks = []
        self.container_names = []
        self.pixel_block_ids = set()
        if self.mode == "disabled":
            return
        if self.mode not in SEQUENTIAL_OFFLOAD_CHOICES:
            raise PiDNodeError(
                f"Unknown sequential_offload={self.mode!r}; expected one of {SEQUENTIAL_OFFLOAD_CHOICES}"
            )
        net = getattr(model, "net", None)
        if offload_patch_blocks:
            self._append_stack(getattr(net, "patch_blocks", None), "net.patch_blocks")
        if offload_pixel_blocks:
            pixel_blocks = self._append_stack(getattr(net, "pixel_blocks", None), "net.pixel_blocks")
            self.pixel_block_ids.update(id(block) for block in pixel_blocks)
        if not self.blocks:
            fallback, container_name = self._find_largest_block_stack(model)
            self._append_stack(fallback, container_name)
        if not self.blocks:
            raise PiDNodeError(
                "Sequential block offload could not find a transformer/DiT block stack in the loaded PiD model. "
                "Set sequential_offload=disabled for this model."
            )
        self._install()

    def _append_stack(self, stack, name: str):
        if stack is None:
            return []
        try:
            blocks = list(stack)
        except Exception:
            return []
        existing = {id(block) for block in self.blocks}
        appended = [block for block in blocks if id(block) not in existing]
        if appended:
            self.blocks.extend(appended)
            self.container_names.append(name)
        return appended

    @staticmethod
    def _module_param_count(module) -> int:
        try:
            return sum(int(p.numel()) for p in module.parameters(recurse=True))
        except Exception:
            return 0

    @staticmethod
    def _children(module):
        try:
            return list(module.children())
        except Exception:
            return []

    def _find_largest_block_stack(self, model):
        candidates = []
        for name, module in model.named_modules():
            children = self._children(module)
            if len(children) < 4:
                continue
            cls_name = module.__class__.__name__.lower()
            name_l = str(name).lower()
            child_classes = " ".join(c.__class__.__name__.lower() for c in children[:8])
            looks_like_blocks = (
                "blocks" in name_l
                or "block" in cls_name
                or "block" in child_classes
                or "transformer" in name_l
                or "dit" in name_l
            )
            if not looks_like_blocks:
                continue
            child_param_counts = [self._module_param_count(c) for c in children]
            total_params = sum(child_param_counts)
            # Avoid tiny helper containers; a real PiD block stack is large.
            if total_params < 10_000_000:
                continue
            candidates.append((total_params, name or module.__class__.__name__, children))

        if not candidates:
            return [], ""
        candidates.sort(key=lambda item: item[0], reverse=True)
        _total, name, blocks = candidates[0]
        return blocks, name

    def _install(self) -> None:
        # Move selected blocks to CPU before the sample loop starts.  The hooks
        # bring each block back only for its own forward pass.
        for block in self.blocks:
            try:
                block.to("cpu")
            except Exception:
                pass
        _trim_cuda_cache()

        for block in self.blocks:
            self.handles.append(block.register_forward_pre_hook(self._pre_forward))
            self.handles.append(block.register_forward_hook(self._post_forward))

    def _pre_forward(self, module, inputs):
        try:
            module.to(self.device)
        except Exception as exc:
            raise PiDNodeError(f"Sequential block offload failed moving a PiD block to CUDA: {exc}") from exc
        return None

    def _post_forward(self, module, inputs, output):
        try:
            module.to("cpu")
        except Exception:
            pass
        if id(module) in self.pixel_block_ids:
            _trim_cuda_cache()
        return output

    def cleanup(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()
        _trim_cuda_cache()


def configure_pid_runtime(model: object, plan: PiDMemoryPlan, device: str = "cuda"):
    if plan.cpu_cache_pixel_positions:
        _enable_cpu_cached_pixel_positions(model, plan.pixel_chunk_patches)
    if plan.chunk_pixel_blocks:
        _enable_chunked_pixel_blocks(model, plan.pixel_chunk_patches)
    if not plan.offload_patch_blocks and not plan.offload_pixel_blocks:
        return None
    return _SequentialBlockOffloader(
        model,
        plan.requested_mode,
        device=device,
        offload_patch_blocks=plan.offload_patch_blocks,
        offload_pixel_blocks=plan.offload_pixel_blocks,
    )


def _latent_samples(latent: dict) -> torch.Tensor:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise PiDNodeError("Expected a ComfyUI LATENT dict containing key 'samples'.")
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]
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


def _comfy_image_to_bchw_01(image: torch.Tensor) -> torch.Tensor:
    # Comfy IMAGE is [B,H,W,C] float 0..1.
    if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
        raise PiDNodeError(f"Expected ComfyUI IMAGE [B,H,W,C], got shape {list(image.shape)}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 1, 3)
    return image.permute(0, 3, 1, 2).contiguous().clamp(0, 1)


def _bchw_neg1_to_comfy_image(image: torch.Tensor) -> torch.Tensor:
    # PiD output is normally [B,C,H,W] in [-1,1]. Some PiD builds return a
    # single-frame video tensor [B,C,1,H,W] or [B,1,C,H,W]. Convert all of
    # those to Comfy IMAGE [B,H,W,C] in [0,1].
    if image.ndim == 5:
        # Observed from PiD/Z-Image: [B, C, T, H, W] with T=1.
        if image.shape[2] == 1 and image.shape[1] in (1, 3, 4):
            image = image[:, :, 0, :, :]
        # Defensive support for [B, T, C, H, W] with T=1.
        elif image.shape[1] == 1 and image.shape[2] in (1, 3, 4):
            image = image[:, 0, :, :, :]
        else:
            raise PiDNodeError(
                "Expected PiD output [B,C,H,W], [B,C,1,H,W], or [B,1,C,H,W], "
                f"got shape {list(image.shape)}"
            )
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise PiDNodeError(f"Expected PiD output [B,C,H,W], got shape {list(image.shape)}")
    if image.shape[1] == 4:
        image = image[:, :3, :, :]
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    if image.shape[1] != 3:
        raise PiDNodeError(f"Expected 1/3/4 output channels, got shape {list(image.shape)}")
    image = image.float().clamp(-1, 1)
    image = (image + 1.0) * 0.5
    return image.permute(0, 2, 3, 1).contiguous().cpu().clamp(0, 1)


def _normalize_pid_samples(samples):
    # Official code uses samples[0]; support tensor, list[tensor], or tuple returns.
    if isinstance(samples, torch.Tensor):
        return samples
    if isinstance(samples, (list, tuple)):
        if len(samples) == 1 and isinstance(samples[0], torch.Tensor) and samples[0].ndim == 4:
            return samples[0]
        tensors = [s for s in samples if isinstance(s, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        if tensors and all(t.ndim == 3 for t in tensors):
            return torch.stack(tensors, dim=0)
    raise PiDNodeError(f"Unsupported PiD output type: {type(samples)}")


def _decode_baseline_with_comfy_vae(vae, samples: torch.Tensor, backbone: str) -> torch.Tensor:
    try:
        image = vae.decode(samples)
    except Exception as exc:
        label = PID_BACKBONES.get(backbone, PID_BACKBONES["zimage"]).label
        raise PiDNodeError(
            f"ComfyUI VAE decode failed. Make sure the connected VAE can decode {label} "
            "latents, or connect a pre-decoded baseline_image instead."
        ) from exc
    return _comfy_image_to_bchw_01(image)


class PiDDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                # A normal (connectable) widget rather than forceInput: the
                # frontend excludes forceInput STRING inputs from
                # widgets_values (Comfy-Org/ComfyUI_frontend#3616), which
                # shifts the index mapping of the widgets that follow so
                # backbone/pid_ckpt_type reset to defaults on every reload.
                # Convert-widget-to-input still lets an upstream prompt feed it.
                "caption": ("STRING", {"multiline": True, "default": ""}),
                # Use ComfyUI's canonical combo syntax.  The older
                # ("COMBO", {"options": ...}) form can cause the frontend to
                # rebuild widget_values with the first combo dropped after tab
                # switches / node refreshes.
                "backbone": (BACKBONE_CHOICES, {"default": "zimage"}),
                "pid_ckpt_type": (["2k", "2kto4k"], {"default": "2k"}),
                "pid_steps": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "scale": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "auto_download": ("BOOLEAN", {"default": True}),
                "unload_comfy_before_pid": ("BOOLEAN", {"default": True}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "pid_source_dir": ("STRING", {"default": "", "multiline": False}),
                "sequential_offload": (SEQUENTIAL_OFFLOAD_CHOICES, {"default": "auto_low_vram"}),
                "pid_weight_precision": (PID_WEIGHT_PRECISION_CHOICES, {"default": "fp32_compatible"}),
                "pixel_chunk_patches": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 1024}),
                "baseline_image": ("IMAGE",),
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
        unload_comfy_before_pid: bool = True,
        aggressive_cleanup: bool = True,
        vae=None,
        pid_source_dir: str = "",
        sequential_offload: str = "auto_low_vram",
        pid_weight_precision: str = "fp32_compatible",
        pixel_chunk_patches: int = 0,
        baseline_image=None,
    ):
        backbone = str(backbone).strip()
        pid_ckpt_type = str(pid_ckpt_type).strip()
        sequential_offload = str(sequential_offload or "auto_low_vram").strip().lower()

        if sequential_offload not in SEQUENTIAL_OFFLOAD_CHOICES:
            raise PiDNodeError(
                f"Unknown sequential_offload={sequential_offload!r}; expected one of {SEQUENTIAL_OFFLOAD_CHOICES}"
            )

        if backbone not in PID_BACKBONES:
            raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
        backbone_info = PID_BACKBONES[backbone]
        ckpt = _checkpoint_for(backbone, pid_ckpt_type)
        if int(scale) <= 0:
            scale = int(ckpt.scale or backbone_info.default_scale)

        pid_dir = _resolve_pid_dir(pid_source_dir)
        model_dir = _resolve_pid_model_dir()
        _ensure_pid_source(pid_dir, allow_download=bool(auto_download))
        _migrate_legacy_checkpoints(model_dir)
        checkpoint_path = _ensure_checkpoint(model_dir, backbone, pid_ckpt_type, allow_download=bool(auto_download))
        _ensure_backbone_assets(model_dir, backbone, allow_download=bool(auto_download))

        samples = _latent_samples(latent)
        sigma = _latent_pid_sigma(latent, sigma)
        if samples.shape[1] != backbone_info.latent_channels:
            raise PiDNodeError(
                f"{backbone_info.label} PiD expects {backbone_info.latent_channels}-channel latents. "
                f"Got {samples.shape[1]} channels."
            )

        # Keep only CPU copies before unloading Z-Image / CLIP / VAE.
        # This prevents upstream CUDA tensors or a connected ComfyUI VAE object from
        # keeping VRAM alive during the PiD-only stage.
        samples_cpu = samples.detach().to("cpu").contiguous()
        baseline_cpu, baseline_size = _baseline_cpu_and_size(
            samples,
            backbone,
            vae=vae,
            baseline_image=baseline_image,
        )
        del samples
        vae = None
        baseline_image = None
        latent = None
        h, w = baseline_size
        infer_image_size = (int(h) * int(scale), int(w) * int(scale))
        plan = resolve_pid_memory_plan(
            sequential_offload=sequential_offload,
            pid_weight_precision=pid_weight_precision,
            pixel_chunk_patches=pixel_chunk_patches,
            infer_image_size=infer_image_size,
        )
        _log_pid_memory_plan(plan)

        # Decode or receive the low-res baseline first, then release Comfy's
        # Z-Image/TE/VAE models before loading/running PiD.
        if unload_comfy_before_pid:
            _free_cuda_memory(aggressive=bool(aggressive_cleanup))

        if not torch.cuda.is_available():
            raise PiDNodeError("CUDA GPU is required for PiD.")

        model = _load_pid_model(
            pid_dir=pid_dir,
            model_dir=model_dir,
            backbone=backbone,
            ckpt_type=pid_ckpt_type,
            checkpoint_path=checkpoint_path,
            dtype_choice="bf16",
            load_ema_to_reg=False,
            skip_vae_encoder=plan.skip_vae_encoder,
            pid_weight_precision=plan.pid_weight_precision,
        )

        device = "cuda"

        caption = caption or ""
        data_batch = _make_pid_data_batch(model, caption, float(sigma), samples_cpu, baseline_cpu, device)
        del samples_cpu
        del baseline_cpu

        if model_management is not None:
            try:
                model_management.throw_exception_if_processing_interrupted()
            except Exception:
                pass

        _free_cuda_memory(aggressive=bool(aggressive_cleanup))
        offloader = configure_pid_runtime(model, plan, device=device)
        _reset_cuda_peak_memory_stats()
        try:
            with torch.inference_mode():
                out = _generate_samples_low_vram(
                    model,
                    data_batch,
                    cfg_scale=float(cfg_scale),
                    num_steps=int(pid_steps),
                    seed=int(seed),
                    shift=None,
                    image_size=infer_image_size,
                )
            _log_cuda_peak_memory("direct decode")
        except Exception as exc:
            # After a CUDA allocator/internal-assert failure, the process may be in a bad state.
            # Still try to free memory and return a useful error to the ComfyUI UI.
            if offloader is not None:
                offloader.cleanup()
                offloader = None
            del data_batch
            _unload_pid_model(model, aggressive=True)
            del model
            raise _format_pid_runtime_error(exc, infer_image_size, f"{backbone}/{pid_ckpt_type}", int(scale)) from exc

        out = _normalize_pid_samples(out)
        image = _bchw_neg1_to_comfy_image(out)

        del out
        del data_batch

        if offloader is not None:
            offloader.cleanup()
            offloader = None

        _unload_pid_model(model, aggressive=bool(aggressive_cleanup))
        del model

        return (image,)


NODE_CLASS_MAPPINGS = {
    "PiDDecode": PiDDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDDecode": "PiD Decode",
}
