from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os
import subprocess
import sys
import tempfile

import torch

try:
    from .pid_decode import (
        PID_BACKBONES,
        BACKBONE_CHOICES,
        SEQUENTIAL_OFFLOAD_CHOICES,
        PiDNodeError,
        _checkpoint_for,
        _resolve_pid_dir,
        _ensure_pid_source,
        _ensure_checkpoint,
        _ensure_backbone_assets,
        _latent_samples,
        _baseline_cpu_and_size,
        _free_cuda_memory,
        _bchw_neg1_to_comfy_image,
    )
except ImportError:  # pragma: no cover
    from pid_decode import (
        PID_BACKBONES,
        BACKBONE_CHOICES,
        SEQUENTIAL_OFFLOAD_CHOICES,
        PiDNodeError,
        _checkpoint_for,
        _resolve_pid_dir,
        _ensure_pid_source,
        _ensure_checkpoint,
        _ensure_backbone_assets,
        _latent_samples,
        _baseline_cpu_and_size,
        _free_cuda_memory,
        _bchw_neg1_to_comfy_image,
    )


PID_PREP_TYPE = "PID_PREP"
PID_SAMPLES_TYPE = "PID_SAMPLES"


@dataclass
class PiDPreparedBatch:
    pid_dir: str
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


@dataclass
class PiDSampledBatch:
    tensor_cpu: torch.Tensor
    backbone: str
    pid_ckpt_type: str
    infer_image_size: Tuple[int, int]


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
        _ensure_pid_source(pid_dir, allow_download=bool(auto_download))
        checkpoint_path = _ensure_checkpoint(pid_dir, backbone, pid_ckpt_type, allow_download=bool(auto_download))
        _ensure_backbone_assets(pid_dir, backbone, allow_download=bool(auto_download))

        samples = _latent_samples(latent)
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
                "sequential_offload": (SEQUENTIAL_OFFLOAD_CHOICES, {"default": "disabled"}),
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
        sequential_offload: str = "disabled",
    ):
        if not isinstance(prepared, PiDPreparedBatch):
            raise PiDNodeError("PiD Sample expected a PID_PREP object from PiD Prepare.")
        sequential_offload = str(sequential_offload or "disabled").strip().lower()
        if sequential_offload not in SEQUENTIAL_OFFLOAD_CHOICES:
            raise PiDNodeError(
                f"Unknown sequential_offload={sequential_offload!r}; expected one of {SEQUENTIAL_OFFLOAD_CHOICES}"
            )

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
                sequential_offload,
            ]
            if aggressive_cleanup:
                cmd.append("--aggressive-cleanup")

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


class PiDFinalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampled": (PID_SAMPLES_TYPE,)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "finalize"
    CATEGORY = "PiD/Staged"

    def finalize(self, sampled: PiDSampledBatch):
        if not isinstance(sampled, PiDSampledBatch):
            raise PiDNodeError("PiD Finalize expected a PID_SAMPLES object from PiD Sample.")
        image = _bchw_neg1_to_comfy_image(sampled.tensor_cpu)
        return (image,)


class PiDDecodeStaged:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "caption": ("STRING", {"forceInput": True}),
                "backbone": (BACKBONE_CHOICES, {"default": "zimage"}),
                "pid_ckpt_type": (["2k", "2kto4k"], {"default": "2k"}),
                "pid_steps": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "scale": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "auto_download": ("BOOLEAN", {"default": True}),
                "cleanup_after_prepare": ("BOOLEAN", {"default": True}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
                "sequential_offload": (SEQUENTIAL_OFFLOAD_CHOICES, {"default": "disabled"}),
            },
            "optional": {
                "vae": ("VAE",),
                "pid_source_dir": ("STRING", {"default": "", "multiline": False}),
                "baseline_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "PiD/Staged"

    def decode(self, **kwargs):
        prep = PiDPrepare().prepare(
            latent=kwargs["latent"],
            caption=kwargs.get("caption", ""),
            backbone=kwargs["backbone"],
            pid_ckpt_type=kwargs["pid_ckpt_type"],
            scale=kwargs["scale"],
            sigma=kwargs["sigma"],
            auto_download=kwargs["auto_download"],
            cleanup_after_prepare=kwargs.get("cleanup_after_prepare", True),
            vae=kwargs.get("vae"),
            pid_source_dir=kwargs.get("pid_source_dir", ""),
            baseline_image=kwargs.get("baseline_image"),
        )[0]
        sampled = PiDSample().sample(
            prepared=prep,
            pid_steps=kwargs["pid_steps"],
            cfg_scale=kwargs["cfg_scale"],
            seed=kwargs["seed"],
            aggressive_cleanup=kwargs.get("aggressive_cleanup", True),
            sequential_offload=kwargs.get("sequential_offload", "disabled"),
        )[0]
        return PiDFinalize().finalize(sampled)


NODE_CLASS_MAPPINGS = {
    "PiDPrepare": PiDPrepare,
    "PiDSample": PiDSample,
    "PiDFinalize": PiDFinalize,
    "PiDDecodeStaged": PiDDecodeStaged,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDPrepare": "PiD Prepare",
    "PiDSample": "PiD Sample",
    "PiDFinalize": "PiD Finalize",
    "PiDDecodeStaged": "PiD Decode (Staged)",
}
