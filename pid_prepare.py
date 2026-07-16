from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

try:
    from .pid_decode import (
        PID_BACKBONES,
        BACKBONE_CHOICES,
        MODEL_PRECISION_CHOICES,
        PID_VERSION_CHOICES,
        PiDNodeError,
        _checkpoint_for,
        _ensure_native_pid_assets,
        _latent_samples,
        _latent_pid_sigma,
        _validate_latent_source_backbone,
        _infer_lq_size_from_latent,
        _log_pid_decode_plan,
        _free_cuda_memory,
        _normalize_scale_for_checkpoint,
        _prepare_latent_for_pid_backbone,
        _validate_pid_base_resolution,
    )
except ImportError:  # pragma: no cover
    from pid_decode import (
        PID_BACKBONES,
        BACKBONE_CHOICES,
        MODEL_PRECISION_CHOICES,
        PID_VERSION_CHOICES,
        PiDNodeError,
        _checkpoint_for,
        _ensure_native_pid_assets,
        _latent_samples,
        _latent_pid_sigma,
        _validate_latent_source_backbone,
        _infer_lq_size_from_latent,
        _log_pid_decode_plan,
        _free_cuda_memory,
        _normalize_scale_for_checkpoint,
        _prepare_latent_for_pid_backbone,
        _validate_pid_base_resolution,
    )


PID_PREP_TYPE = "PID_PREP"


@dataclass
class PiDPreparedBatch:
    version: str
    backbone: str
    pid_ckpt_type: str
    model_precision: str
    diffusion_model_path: str
    text_encoder_path: str
    diffusion_model_name: str
    text_encoder_name: str
    latent_format: str
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
                "version": (PID_VERSION_CHOICES, {"default": "v1"}),
                "backbone": (BACKBONE_CHOICES, {"default": "zimage"}),
                "pid_ckpt_type": (["2k", "2kto4k"], {"default": "2k"}),
                "scale": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "auto_download": ("BOOLEAN", {"default": True}),
                "model_precision": (MODEL_PRECISION_CHOICES, {"default": "bf16"}),
                "cleanup_after_prepare": ("BOOLEAN", {"default": True}),
                "caption": ("STRING", {"forceInput": True}),
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
        model_precision: str = "bf16",
        version: str = "v1",
        cleanup_after_prepare: bool = True,
        pid_source_dir: str = "",
    ):
        backbone = str(backbone).strip()
        pid_ckpt_type = str(pid_ckpt_type).strip()
        if backbone not in PID_BACKBONES:
            raise PiDNodeError(f"Unknown backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
        _validate_latent_source_backbone(latent, backbone)
        backbone_info = PID_BACKBONES[backbone]
        ckpt = _checkpoint_for(backbone, pid_ckpt_type, model_precision, version=version)
        scale = _normalize_scale_for_checkpoint(backbone, ckpt, int(scale))
        diffusion_model_path, text_encoder_path = _ensure_native_pid_assets(
            ckpt,
            allow_download=bool(auto_download),
        )
        del pid_source_dir

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
        _validate_pid_base_resolution(ckpt, (h, w))
        infer_image_size = (int(h) * int(scale), int(w) * int(scale))
        _log_pid_decode_plan(ckpt, tuple(samples.shape), (h, w), infer_image_size, sigma)

        if cleanup_after_prepare:
            _free_cuda_memory(aggressive=True)

        prepared = PiDPreparedBatch(
            version=ckpt.version,
            backbone=backbone,
            pid_ckpt_type=pid_ckpt_type,
            model_precision=ckpt.model_precision,
            diffusion_model_path=str(diffusion_model_path),
            text_encoder_path=str(text_encoder_path),
            diffusion_model_name=ckpt.diffusion_filename,
            text_encoder_name=ckpt.text_encoder_filename,
            latent_format=ckpt.latent_format,
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
