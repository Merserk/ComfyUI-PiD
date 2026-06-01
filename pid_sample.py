from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os
import subprocess
import sys
import tempfile

import torch

try:
    from .pid_decode import (
        PID_WEIGHT_PRECISION_CHOICES,
        SEQUENTIAL_OFFLOAD_CHOICES,
        PiDNodeError,
        _free_cuda_memory,
    )
    from .pid_prepare import PID_PREP_TYPE, PiDPreparedBatch
except ImportError:  # pragma: no cover
    from pid_decode import PID_WEIGHT_PRECISION_CHOICES, SEQUENTIAL_OFFLOAD_CHOICES, PiDNodeError, _free_cuda_memory
    from pid_prepare import PID_PREP_TYPE, PiDPreparedBatch


PID_SAMPLES_TYPE = "PID_SAMPLES"


@dataclass
class PiDSampledBatch:
    tensor_cpu: torch.Tensor
    backbone: str
    pid_ckpt_type: str
    infer_image_size: Tuple[int, int]


def _build_pid_subprocess_command(
    runner: Path,
    input_path: Path,
    output_path: Path,
    pid_steps: int,
    cfg_scale: float,
    seed: int,
    sequential_offload: str,
    pid_weight_precision: str,
    pixel_chunk_patches: int,
    aggressive_cleanup: bool,
):
    cmd = [
        sys.executable or "python",
        str(runner),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--pid-steps",
        str(int(pid_steps)),
        "--cfg-scale",
        str(float(cfg_scale)),
        "--seed",
        str(int(seed)),
        "--sequential-offload",
        str(sequential_offload),
        "--pid-weight-precision",
        str(pid_weight_precision),
        "--pixel-chunk-patches",
        str(int(pixel_chunk_patches)),
    ]
    if aggressive_cleanup:
        cmd.append("--aggressive-cleanup")
    return cmd


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
                "sequential_offload": (SEQUENTIAL_OFFLOAD_CHOICES, {"default": "auto_low_vram"}),
                "pid_weight_precision": (PID_WEIGHT_PRECISION_CHOICES, {"default": "fp32_compatible"}),
                "pixel_chunk_patches": ("INT", {"default": 0, "min": 0, "max": 65536, "step": 1024}),
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
        sequential_offload: str = "auto_low_vram",
        pid_weight_precision: str = "fp32_compatible",
        pixel_chunk_patches: int = 0,
    ):
        if not isinstance(prepared, PiDPreparedBatch):
            raise PiDNodeError("PiD Sample expected a PID_PREP object from PiD Prepare.")
        sequential_offload = str(sequential_offload or "auto_low_vram").strip().lower()
        if sequential_offload not in SEQUENTIAL_OFFLOAD_CHOICES:
            raise PiDNodeError(
                f"Unknown sequential_offload={sequential_offload!r}; expected one of {SEQUENTIAL_OFFLOAD_CHOICES}"
            )
        pid_weight_precision = str(pid_weight_precision or "fp32_compatible").strip().lower()
        if pid_weight_precision not in PID_WEIGHT_PRECISION_CHOICES:
            raise PiDNodeError(
                f"Unknown pid_weight_precision={pid_weight_precision!r}; "
                f"expected one of {PID_WEIGHT_PRECISION_CHOICES}"
            )
        pixel_chunk_patches = int(pixel_chunk_patches)
        if pixel_chunk_patches < 0:
            raise PiDNodeError("pixel_chunk_patches must be zero (automatic) or a positive integer.")

        _free_cuda_memory(aggressive=True)
        runner = Path(__file__).resolve().with_name("pid_subprocess_runner.py")
        if not runner.is_file():
            raise PiDNodeError(f"Missing PiD subprocess runner: {runner}")

        with tempfile.TemporaryDirectory(prefix="comfyui_pid_") as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / "pid_input.pt"
            output_path = tmpdir / "pid_output.pt"
            payload = {
                "pid_dir": prepared.pid_dir,
                "model_dir": prepared.model_dir,
                "backbone": prepared.backbone,
                "pid_ckpt_type": prepared.pid_ckpt_type,
                "checkpoint_path": prepared.checkpoint_path,
                "caption": prepared.caption,
                "sigma": float(prepared.sigma),
                "scale": int(prepared.scale),
                "infer_image_size": tuple(int(x) for x in prepared.infer_image_size),
                "latent_cpu": prepared.latent_cpu.detach().to("cpu").contiguous(),
                "baseline_cpu": (
                    prepared.baseline_cpu.detach().to("cpu").contiguous()
                    if prepared.baseline_cpu is not None
                    else None
                ),
                "baseline_size": tuple(int(x) for x in prepared.baseline_size),
            }
            torch.save(payload, str(input_path))
            del payload
            _free_cuda_memory(aggressive=True)

            cmd = _build_pid_subprocess_command(
                runner=runner,
                input_path=input_path,
                output_path=output_path,
                pid_steps=pid_steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sequential_offload=sequential_offload,
                pid_weight_precision=pid_weight_precision,
                pixel_chunk_patches=pixel_chunk_patches,
                aggressive_cleanup=bool(aggressive_cleanup),
            )

            env = os.environ.copy()
            node_dir = str(Path(__file__).resolve().parent)
            env["PYTHONPATH"] = node_dir + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                cmd,
                cwd=node_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0 or not output_path.is_file():
                tail = "\n".join((proc.stdout or "").splitlines()[-120:])
                raise PiDNodeError(
                    "PiD subprocess sampling failed. This usually means the 4K PiD pass still exceeded VRAM, "
                    "or the subprocess could not import/load PiD.\n\n"
                    f"Command: {' '.join(cmd)}\n\n"
                    f"Subprocess log tail:\n{tail}"
                )

            try:
                result = torch.load(str(output_path), map_location="cpu", weights_only=False)
            except TypeError:
                result = torch.load(str(output_path), map_location="cpu")

        _free_cuda_memory(aggressive=True)
        sampled = PiDSampledBatch(
            tensor_cpu=result["tensor_cpu"].detach().to("cpu"),
            backbone=str(result.get("backbone", prepared.backbone)),
            pid_ckpt_type=str(result.get("pid_ckpt_type", prepared.pid_ckpt_type)),
            infer_image_size=tuple(int(x) for x in result.get("infer_image_size", prepared.infer_image_size)),
        )
        return (sampled,)


NODE_CLASS_MAPPINGS = {
    "PiDSample": PiDSample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDSample": "PiD Sample",
}
