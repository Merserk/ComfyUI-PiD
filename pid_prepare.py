from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
        _validate_latent_source_backbone,
        _infer_lq_size_from_latent,
        _free_cuda_memory,
        _normalize_scale_for_checkpoint,
        _prepare_latent_for_pid_backbone,
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
        _validate_latent_source_backbone,
        _infer_lq_size_from_latent,
        _free_cuda_memory,
        _normalize_scale_for_checkpoint,
        _prepare_latent_for_pid_backbone,
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


class PiDPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "caption": ("STRING", {"forceInput": True}),
                "backbone": (BACKBONE_CHOICES, {"default": "zimage"}),
                "pid_ckpt_type": (["2k", "2kto4k"], {"default": "2k"}),
                "scale": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "auto_download": ("BOOLEAN", {"default": True}),
                "cleanup_after_prepare": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "pid_source_dir": ("STRING", {"default": "", "multiline": False}),
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
        pid_source_dir: str = "",
    ):
        backbone = str(backbone).strip()
        pid_ckpt_type = str(pid_ckpt_type).strip()
        if backbone not in PID_BACKBONES:
            raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
        _validate_latent_source_backbone(latent, backbone)
        backbone_info = PID_BACKBONES[backbone]
        ckpt = _checkpoint_for(backbone, pid_ckpt_type)
        scale = _normalize_scale_for_checkpoint(backbone, ckpt, int(scale))

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
        samples_cpu, sigma = _prepare_latent_for_pid_backbone(samples_cpu, sigma, backbone)
        h, w = _infer_lq_size_from_latent(samples, backbone)
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
        )
        return (prepared,)


NODE_CLASS_MAPPINGS = {
    "PiDPrepare": PiDPrepare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDPrepare": "PiD Prepare",
}
