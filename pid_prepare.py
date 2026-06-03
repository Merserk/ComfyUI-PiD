from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    from .pid_decode import (
        PID_BACKBONES,
        BACKBONE_CHOICES,
        PiDNodeError,
        _checkpoint_for,
        _resolve_pid_dir,
        _resolve_pid_model_dir,
        _migrate_legacy_checkpoints,
        _ensure_pid_source,
        _required_pid_source_files_for_backbone,
        _ensure_checkpoint,
        _ensure_backbone_assets,
        _latent_samples,
        _latent_pid_sigma,
        _baseline_cpu_and_size,
        _free_cuda_memory,
    )
except ImportError:  # pragma: no cover
    from pid_decode import (
        PID_BACKBONES,
        BACKBONE_CHOICES,
        PiDNodeError,
        _checkpoint_for,
        _resolve_pid_dir,
        _resolve_pid_model_dir,
        _migrate_legacy_checkpoints,
        _ensure_pid_source,
        _required_pid_source_files_for_backbone,
        _ensure_checkpoint,
        _ensure_backbone_assets,
        _latent_samples,
        _latent_pid_sigma,
        _baseline_cpu_and_size,
        _free_cuda_memory,
    )


PID_PREP_TYPE = "PID_PREP"


@dataclass
class PiDPreparedBatch:
    pid_dir: str
    model_dir: str
    backbone: str
    pid_ckpt_type: str
    checkpoint_path: str
    caption: str
    sigma: float
    scale: int
    infer_image_size: Tuple[int, int]
    latent_cpu: torch.Tensor
    baseline_cpu: Optional[torch.Tensor]
    baseline_size: Tuple[int, int]


class PiDPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "caption": ("STRING", {"multiline": True, "default": ""}),
                "backbone": (BACKBONE_CHOICES, {"default": "zimage"}),
                "pid_ckpt_type": (["2k", "2kto4k"], {"default": "2k"}),
                "scale": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "auto_download": ("BOOLEAN", {"default": True}),
                "cleanup_after_prepare": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "pid_source_dir": ("STRING", {"default": "", "multiline": False}),
                "baseline_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (PID_PREP_TYPE,)
    RETURN_NAMES = ("prepared",)
    FUNCTION = "prepare"
    CATEGORY = "PiD/Staged"

    def prepare(
        self,
        latent,
        caption: str,
        backbone: str,
        pid_ckpt_type: str,
        scale: int,
        sigma: float,
        auto_download: bool,
        cleanup_after_prepare: bool = True,
        vae=None,
        pid_source_dir: str = "",
        baseline_image=None,
    ):
        backbone = str(backbone).strip()
        pid_ckpt_type = str(pid_ckpt_type).strip()
        if backbone not in PID_BACKBONES:
            raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
        backbone_info = PID_BACKBONES[backbone]
        ckpt = _checkpoint_for(backbone, pid_ckpt_type)
        if int(scale) <= 0:
            scale = int(ckpt.scale or backbone_info.default_scale)

        pid_dir = _resolve_pid_dir(pid_source_dir)
        model_dir = _resolve_pid_model_dir()
        _ensure_pid_source(
            pid_dir,
            allow_download=bool(auto_download),
            required_files=_required_pid_source_files_for_backbone(backbone),
        )
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

        samples_cpu = samples.detach().to("cpu").contiguous()
        baseline_cpu, baseline_size = _baseline_cpu_and_size(
            samples,
            backbone,
            vae=vae,
            baseline_image=baseline_image,
        )
        h, w = baseline_size
        infer_image_size = (int(h) * int(scale), int(w) * int(scale))

        if cleanup_after_prepare:
            _free_cuda_memory(aggressive=True)

        prepared = PiDPreparedBatch(
            pid_dir=str(pid_dir),
            model_dir=str(model_dir),
            backbone=backbone,
            pid_ckpt_type=pid_ckpt_type,
            checkpoint_path=str(checkpoint_path),
            caption=caption or "",
            sigma=float(sigma),
            scale=int(scale),
            infer_image_size=infer_image_size,
            latent_cpu=samples_cpu,
            baseline_cpu=baseline_cpu,
            baseline_size=baseline_size,
        )
        return (prepared,)


NODE_CLASS_MAPPINGS = {
    "PiDPrepare": PiDPrepare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDPrepare": "PiD Prepare",
}
