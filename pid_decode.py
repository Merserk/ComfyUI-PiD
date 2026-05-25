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
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from dataclasses import dataclass
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
    ckpt_types: Tuple[str, ...]
    aux_files: Tuple[str, ...] = ()
    needs_flux_ae: bool = False


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
    "zimage": PiDBackbone("Z-Image", "zimage", 16, 4, ("2k", "2kto4k"), needs_flux_ae=True),
    "flux": PiDBackbone("Flux", "flux", 16, 4, ("2k", "2kto4k"), needs_flux_ae=True),
    "flux2": PiDBackbone(
        "Flux2",
        "flux2",
        128,
        4,
        ("2k", "2kto4k"),
        aux_files=("checkpoints/flux2_ae.safetensors",),
    ),
    "sd3": PiDBackbone(
        "SD3",
        "sd3",
        16,
        4,
        ("2k", "2kto4k"),
        aux_files=("checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors",),
    ),
    "dinov2": PiDBackbone(
        "DINOv2 RAE",
        "rae",
        768,
        4,
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
        ("2k",),
        aux_files=(
            "checkpoints/scale_rae/decoder/XL_decoder_config.json",
            "checkpoints/scale_rae/decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt",
        ),
    ),
}

BACKBONE_CHOICES = list(PID_BACKBONES.keys())

_MODEL_CACHE: Dict[Tuple[str, str, str, str, str, bool, str], object] = {}


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


def _format_pid_runtime_error(exc: BaseException, infer_image_size: Tuple[int, int], ckpt_type: str, scale: int) -> PiDNodeError:
    msg = str(exc)
    lower = msg.lower()
    h, w = int(infer_image_size[0]), int(infer_image_size[1])
    advice = (
        f"PiD inference failed while generating {w}x{h} with pid_ckpt_type={ckpt_type!r}, scale={scale}.\n\n"
        "Most common cause: VRAM/allocator pressure. For a 16GB GPU, 1024 base + scale=4 "
        "means roughly 4096x4096 PiD output, which is usually too heavy inside ComfyUI while "
        "large generation models were just used.\n\n"
        "Try these first:\n"
        "  1. Set PiD scale=1 or scale=2 for 1024 base images.\n"
        "  2. For the 2k PiD checkpoint, use a 512x512 base latent with scale=4.\n"
        "  3. Keep unload_comfy_before_pid=true and aggressive_cleanup=true.\n"
        "  4. Restart ComfyUI after a CUDA allocator crash.\n"
        "  5. If your launcher supports it, try disabling cudaMallocAsync / CUDA malloc async.\n\n"
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



def _candidate_comfy_ae_paths() -> List[Path]:
    """Return likely local Flux/Z-Image AE VAE paths from ComfyUI.

    NVIDIA PiD's own config instantiates a Flux VAE encoder/tokenizer and expects
    ./checkpoints/ae.safetensors relative to the PiD repository root. Most ComfyUI
    Z-Image workflows already have the exact same file under ComfyUI/models/vae/.
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


def _copy_local_ae_to_pid(pid_dir: Path) -> Optional[str]:
    target = pid_dir / AE_REL_PATH
    if target.is_file():
        return f"Flux AE VAE already exists: {target}"

    for src in _candidate_comfy_ae_paths():
        if src.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            return f"Copied Flux AE VAE from {src} to {target}"
    return None


def _download_ae_safetensors(pid_dir: Path) -> str:
    _ensure_hf_download_available()
    from huggingface_hub import hf_hub_download

    target = pid_dir / AE_REL_PATH
    if target.is_file():
        return f"Flux AE VAE already exists: {target}"

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pid_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=AE_REL_PATH,
        local_dir=str(pid_dir),
        token=token,
    )
    if not target.is_file():
        raise PiDNodeError(f"Hugging Face download finished but Flux AE VAE is missing: {target}")
    return f"Downloaded Flux AE VAE: {target}"


def _ensure_ae_safetensors(pid_dir: Path, allow_download: bool = True) -> Path:
    """Ensure PiD's required ./checkpoints/ae.safetensors exists."""
    target = pid_dir / AE_REL_PATH
    if target.is_file():
        return target

    copied = _copy_local_ae_to_pid(pid_dir)
    if copied and target.is_file():
        return target

    if allow_download:
        try:
            _download_ae_safetensors(pid_dir)
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


def _download_hf_file(pid_dir: Path, relpath: str) -> str:
    _ensure_hf_download_available()
    from huggingface_hub import hf_hub_download

    target = pid_dir / relpath
    if target.is_file():
        return f"Asset already exists: {target}"

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pid_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=relpath,
        local_dir=str(pid_dir),
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


def _ensure_checkpoint(pid_dir: Path, backbone: str, ckpt_type: str, allow_download: bool = True) -> Path:
    ckpt = _checkpoint_for(backbone, ckpt_type)
    ckpt_path = pid_dir / ckpt.relpath
    if ckpt_path.is_file():
        return ckpt_path
    if not allow_download:
        raise PiDNodeError(
            f"PiD checkpoint is missing: {ckpt_path}\n"
            "Set auto_download=true or download the matching file from nvidia/PiD."
        )
    _download_hf_file(pid_dir, ckpt.relpath)
    return ckpt_path


def _ensure_backbone_assets(pid_dir: Path, backbone: str, allow_download: bool = True) -> None:
    info = PID_BACKBONES[backbone]
    if info.needs_flux_ae:
        _ensure_ae_safetensors(pid_dir, allow_download=allow_download)

    missing = [relpath for relpath in info.aux_files if not (pid_dir / relpath).is_file()]
    if not missing:
        return
    if not allow_download:
        missing_text = "\n".join(f"  - {pid_dir / relpath}" for relpath in missing)
        raise PiDNodeError(
            f"{info.label} PiD support needs these extra asset files:\n"
            f"{missing_text}\n\n"
            "Set auto_download=true or download them from nvidia/PiD."
        )
    for relpath in missing:
        _download_hf_file(pid_dir, relpath)


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
    backbone: str,
    ckpt_type: str,
    checkpoint_path: Path,
    dtype_choice: str,
    load_ema_to_reg: bool = False,
):
    experiment = _checkpoint_for(backbone, ckpt_type).experiment
    config_file = "pid/_src/configs/pid/config.py"
    cache_key = (
        str(pid_dir),
        backbone,
        ckpt_type,
        str(checkpoint_path),
        experiment,
        bool(load_ema_to_reg),
        dtype_choice,
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not torch.cuda.is_available():
        raise PiDNodeError("PiD's official model loader instantiates the decoder on CUDA. CUDA GPU is required.")

    load_model_from_checkpoint = _import_pid_loader(pid_dir)

    # PiD's config helper expects config_file to be relative to the PiD repo root.
    with _pushd(pid_dir):
        model, _config = load_model_from_checkpoint(
            experiment_name=experiment,
            checkpoint_path=str(checkpoint_path),
            config_file=config_file,
            enable_fsdp=False,
            experiment_opts=[],
            strict=False,
            load_ema_to_reg=load_ema_to_reg,
        )
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _latent_samples(latent: dict) -> torch.Tensor:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise PiDNodeError("Expected a ComfyUI LATENT dict containing key 'samples'.")
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[0]
    if samples.ndim != 4:
        raise PiDNodeError(f"Expected latent samples as [B,C,H,W], got shape {list(samples.shape)}")
    return samples


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
                "prompt": ("STRING", {"multiline": True, "default": ""}),
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
                "auto_settings": ("PID_SETTINGS",),
                "backbone_in": ("STRING", {"forceInput": True}),
                "pid_ckpt_type_in": ("STRING", {"forceInput": True}),
                "pid_steps_in": ("INT", {"forceInput": True}),
                "scale_in": ("INT", {"forceInput": True}),
                "cfg_scale_in": ("FLOAT", {"forceInput": True}),
                "sigma_in": ("FLOAT", {"forceInput": True}),
                "caption": ("STRING", {"forceInput": True}),
                "vae": ("VAE",),
                "pid_source_dir": ("STRING", {"default": "", "multiline": False}),
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
        prompt: str,
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
        auto_settings=None,
        backbone_in: str = "",
        pid_ckpt_type_in: str = "",
        pid_steps_in=None,
        scale_in=None,
        cfg_scale_in=None,
        sigma_in=None,
        caption: str = "",
        vae=None,
        pid_source_dir: str = "",
        baseline_image=None,
    ):
        if isinstance(auto_settings, dict):
            backbone = auto_settings.get("backbone", backbone)
            pid_ckpt_type = auto_settings.get("pid_ckpt_type", pid_ckpt_type)
            pid_steps = int(auto_settings.get("pid_steps", pid_steps))
            scale = int(auto_settings.get("scale", scale))
            cfg_scale = float(auto_settings.get("cfg_scale", cfg_scale))
            sigma = float(auto_settings.get("sigma", sigma))
            unload_comfy_before_pid = bool(auto_settings.get("unload_comfy_before_pid", unload_comfy_before_pid))
            aggressive_cleanup = bool(auto_settings.get("aggressive_cleanup", aggressive_cleanup))

        if backbone_in:
            backbone = str(backbone_in)
        if pid_ckpt_type_in:
            pid_ckpt_type = str(pid_ckpt_type_in)
        if pid_steps_in is not None:
            pid_steps = int(pid_steps_in)
        if scale_in is not None:
            scale = int(scale_in)
        if cfg_scale_in is not None:
            cfg_scale = float(cfg_scale_in)
        if sigma_in is not None:
            sigma = float(sigma_in)

        if backbone not in PID_BACKBONES:
            raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
        backbone_info = PID_BACKBONES[backbone]
        ckpt = _checkpoint_for(backbone, pid_ckpt_type)
        if int(scale) <= 0:
            scale = int(ckpt.scale or backbone_info.default_scale)

        pid_dir = _resolve_pid_dir(pid_source_dir)
        _ensure_pid_source(pid_dir, allow_download=bool(auto_download))
        checkpoint_path = _ensure_checkpoint(pid_dir, backbone, pid_ckpt_type, allow_download=bool(auto_download))
        _ensure_backbone_assets(pid_dir, backbone, allow_download=bool(auto_download))

        samples = _latent_samples(latent)
        if samples.shape[1] != backbone_info.latent_channels:
            raise PiDNodeError(
                f"{backbone_info.label} PiD expects {backbone_info.latent_channels}-channel latents. "
                f"Got {samples.shape[1]} channels."
            )

        if baseline_image is None:
            if vae is None:
                raise PiDNodeError(
                    "PiD needs a baseline image. Connect either a matching ComfyUI VAE "
                    "or a pre-decoded baseline_image."
                )
            baseline_01 = _decode_baseline_with_comfy_vae(vae, samples, backbone)
        else:
            baseline_01 = _comfy_image_to_bchw_01(baseline_image)

        if baseline_01.shape[0] != samples.shape[0]:
            raise PiDNodeError(
                f"Batch mismatch: latent batch={samples.shape[0]}, baseline batch={baseline_01.shape[0]}"
            )

        # Decode the low-res baseline first, then release Comfy's Z-Image/TE/VAE models
        # before loading/running PiD. This is important on 16GB cards.
        if unload_comfy_before_pid:
            _free_cuda_memory(aggressive=bool(aggressive_cleanup))

        if not torch.cuda.is_available():
            raise PiDNodeError("CUDA GPU is required for PiD.")

        model = _load_pid_model(
            pid_dir=pid_dir,
            backbone=backbone,
            ckpt_type=pid_ckpt_type,
            checkpoint_path=checkpoint_path,
            dtype_choice="bf16",
            load_ema_to_reg=False,
        )

        b, _c, h, w = baseline_01.shape
        device = "cuda"
        latent_bf16 = samples.to(device=device, dtype=torch.bfloat16)
        baseline_neg1_1 = (baseline_01.to(device=device, dtype=torch.bfloat16) * 2.0) - 1.0

        caption = caption if caption else (prompt or "")
        data_batch = {
            model.config.input_caption_key: [caption] * int(b),
            "LQ_video_or_image": baseline_neg1_1,
            "LQ_latent": latent_bf16,
            "degrade_sigma": torch.full((int(b),), float(sigma), device=device, dtype=torch.float32),
        }
        infer_image_size = (int(h) * int(scale), int(w) * int(scale))

        if model_management is not None:
            try:
                model_management.throw_exception_if_processing_interrupted()
            except Exception:
                pass

        _free_cuda_memory(aggressive=bool(aggressive_cleanup))
        try:
            with torch.inference_mode():
                out = model.generate_samples_from_batch(
                    data_batch,
                    cfg_scale=float(cfg_scale),
                    num_steps=int(pid_steps),
                    seed=int(seed),
                    shift=None,
                    image_size=infer_image_size,
                )
        except Exception as exc:
            # After a CUDA allocator/internal-assert failure, the process may be in a bad state.
            # Still try to free memory and return a useful error to the ComfyUI UI.
            _free_cuda_memory(aggressive=True)
            raise _format_pid_runtime_error(exc, infer_image_size, f"{backbone}/{pid_ckpt_type}", int(scale)) from exc

        out = _normalize_pid_samples(out)
        image = _bchw_neg1_to_comfy_image(out)

        if model_management is not None:
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass

        return (image,)


NODE_CLASS_MAPPINGS = {
    "PiDDecode": PiDDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDDecode": "PiD Decode",
}
