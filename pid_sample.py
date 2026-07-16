from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

try:
    from .pid_decode import (
        PiDNodeError,
        _checkpoint_for,
        _format_pid_runtime_error,
        _free_cuda_memory,
        _log_cuda_peak_memory,
        _make_pid_progress_bar,
        _reset_cuda_peak_memory_stats,
        _run_native_pid_decode,
        _update_pid_progress_bar,
        _warn_if_non_distilled_step_count,
    )
    from .pid_prepare import PID_PREP_TYPE, PiDPreparedBatch
except ImportError:  # pragma: no cover
    from pid_decode import (
        PiDNodeError,
        _checkpoint_for,
        _format_pid_runtime_error,
        _free_cuda_memory,
        _log_cuda_peak_memory,
        _make_pid_progress_bar,
        _reset_cuda_peak_memory_stats,
        _run_native_pid_decode,
        _update_pid_progress_bar,
        _warn_if_non_distilled_step_count,
    )
    from pid_prepare import PID_PREP_TYPE, PiDPreparedBatch


PID_SAMPLES_TYPE = "PID_SAMPLES"


@dataclass
class PiDSampledBatch:
    tensor_cpu: torch.Tensor
    backbone: str
    pid_ckpt_type: str
    infer_image_size: Tuple[int, int]


class PiDSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepared": (PID_PREP_TYPE,),
                "pid_steps": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (PID_SAMPLES_TYPE,)
    RETURN_NAMES = ("sampled",)
    FUNCTION = "sample"
    CATEGORY = "PiD/Staged"

    def sample(
        self,
        prepared: PiDPreparedBatch,
        pid_steps: int,
        cfg_scale: float,
        seed: int,
        aggressive_cleanup: bool = True,
    ):
        if not isinstance(prepared, PiDPreparedBatch):
            raise PiDNodeError("PiD Sample expected a PID_PREP object from PiD Prepare.")

        _free_cuda_memory(aggressive=bool(aggressive_cleanup))
        _warn_if_non_distilled_step_count(pid_steps)
        pbar = _make_pid_progress_bar(pid_steps)

        def update_progress(current: int, total: int) -> None:
            _update_pid_progress_bar(pbar, current, total)

        infer_image_size = tuple(int(x) for x in prepared.infer_image_size)
        spec = _checkpoint_for(
            prepared.backbone,
            prepared.pid_ckpt_type,
            getattr(prepared, "model_precision", "bf16"),
            version=getattr(prepared, "version", "v1"),
        )

        _reset_cuda_peak_memory_stats()
        try:
            out = _run_native_pid_decode(
                spec,
                prepared.caption or "",
                prepared.latent_cpu.detach().to("cpu").contiguous(),
                float(prepared.sigma),
                infer_image_size,
                int(pid_steps),
                float(cfg_scale),
                int(seed),
                allow_download=False,
                diffusion_model_path=Path(prepared.diffusion_model_path),
                text_encoder_path=Path(prepared.text_encoder_path),
                progress_callback=update_progress if pbar is not None else None,
            )
            _log_cuda_peak_memory("staged native sample")
        except Exception as exc:
            _free_cuda_memory(aggressive=True)
            raise _format_pid_runtime_error(
                exc,
                infer_image_size,
                f"{spec.version}/{prepared.backbone}/{prepared.pid_ckpt_type}/{spec.diffusion_filename}",
                int(prepared.scale),
            ) from exc

        sampled = PiDSampledBatch(
            tensor_cpu=out.detach().to("cpu"),
            backbone=str(prepared.backbone),
            pid_ckpt_type=str(prepared.pid_ckpt_type),
            infer_image_size=infer_image_size,
        )
        del out
        _free_cuda_memory(aggressive=bool(aggressive_cleanup))
        return (sampled,)


NODE_CLASS_MAPPINGS = {
    "PiDSample": PiDSample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDSample": "PiD Sample",
}
