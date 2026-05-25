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
    from comfy import model_management
except Exception:  # pragma: no cover
    model_management = None

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
        _decode_baseline_with_comfy_vae,
        _comfy_image_to_bchw_01,
        _free_cuda_memory,
        _load_pid_model,
        _normalize_pid_samples,
        _bchw_neg1_to_comfy_image,
        _unload_pid_model,
        _SequentialBlockOffloader,
        _format_pid_runtime_error,
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
        _decode_baseline_with_comfy_vae,
        _comfy_image_to_bchw_01,
        _free_cuda_memory,
        _load_pid_model,
        _normalize_pid_samples,
        _bchw_neg1_to_comfy_image,
        _unload_pid_model,
        _SequentialBlockOffloader,
        _format_pid_runtime_error,
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
    baseline_cpu: torch.Tensor


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

        if baseline_image is None:
            if vae is None:
                raise PiDNodeError(
                    "PiD Prepare needs a baseline image. Connect either a matching ComfyUI VAE "
                    "or a pre-decoded baseline_image."
                )
            baseline_01 = _decode_baseline_with_comfy_vae(vae, samples, backbone)
        else:
            baseline_01 = _comfy_image_to_bchw_01(baseline_image)

        if baseline_01.shape[0] != samples.shape[0]:
            raise PiDNodeError(
                f"Batch mismatch: latent batch={samples.shape[0]}, baseline batch={baseline_01.shape[0]}"
            )

        samples_cpu = samples.detach().to("cpu").contiguous()
        baseline_cpu = baseline_01.detach().to("cpu").contiguous()
        b, _c, h, w = baseline_cpu.shape
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
                "unload_comfy_before_pid": ("BOOLEAN", {"default": True}),
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
        unload_comfy_before_pid: bool = True,
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
        if not torch.cuda.is_available():
            raise PiDNodeError("CUDA GPU is required for PiD.")

        if unload_comfy_before_pid:
            _free_cuda_memory(aggressive=bool(aggressive_cleanup))

        pid_dir = Path(prepared.pid_dir)
        checkpoint_path = Path(prepared.checkpoint_path)
        model = _load_pid_model(
            pid_dir=pid_dir,
            backbone=prepared.backbone,
            ckpt_type=prepared.pid_ckpt_type,
            checkpoint_path=checkpoint_path,
            dtype_choice="bf16",
            load_ema_to_reg=False,
        )

        device = "cuda"
        latent_inputs = prepared.latent_cpu.to(device=device, dtype=torch.bfloat16)
        baseline_neg1_1 = (prepared.baseline_cpu.to(device=device, dtype=torch.bfloat16) * 2.0) - 1.0
        batch = int(prepared.baseline_cpu.shape[0])

        data_batch = {
            model.config.input_caption_key: [prepared.caption] * batch,
            "LQ_video_or_image": baseline_neg1_1,
            "LQ_latent": latent_inputs,
            "degrade_sigma": torch.full((batch,), float(prepared.sigma), device=device, dtype=torch.float32),
        }

        offloader = None
        if sequential_offload != "disabled":
            offloader = _SequentialBlockOffloader(model, sequential_offload, device=device)

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
                    image_size=prepared.infer_image_size,
                )
        except Exception as exc:
            if offloader is not None:
                offloader.cleanup()
            del data_batch
            del latent_inputs
            del baseline_neg1_1
            _unload_pid_model(model, aggressive=True)
            del model
            raise _format_pid_runtime_error(
                exc,
                prepared.infer_image_size,
                f"{prepared.backbone}/{prepared.pid_ckpt_type}",
                int(prepared.scale),
            ) from exc

        if offloader is not None:
            offloader.cleanup()

        out = _normalize_pid_samples(out)
        out_cpu = out.detach().to("cpu")

        del out
        del data_batch
        del latent_inputs
        del baseline_neg1_1
        _unload_pid_model(model, aggressive=bool(aggressive_cleanup))
        del model

        sampled = PiDSampledBatch(
            tensor_cpu=out_cpu,
            backbone=prepared.backbone,
            pid_ckpt_type=prepared.pid_ckpt_type,
            infer_image_size=prepared.infer_image_size,
        )
        return (sampled,)


class PiDSampleSubprocess:
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
            raise PiDNodeError("PiD Sample Subprocess expected a PID_PREP object from PiD Prepare.")
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
                "baseline_cpu": prepared.baseline_cpu.detach().to("cpu").contiguous(),
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
                "unload_comfy_before_pid": ("BOOLEAN", {"default": True}),
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
            unload_comfy_before_pid=kwargs.get("unload_comfy_before_pid", True),
            aggressive_cleanup=kwargs.get("aggressive_cleanup", True),
            sequential_offload=kwargs.get("sequential_offload", "disabled"),
        )[0]
        return PiDFinalize().finalize(sampled)


NODE_CLASS_MAPPINGS = {
    "PiDPrepare": PiDPrepare,
    "PiDSample": PiDSample,
    "PiDSampleSubprocess": PiDSampleSubprocess,
    "PiDFinalize": PiDFinalize,
    "PiDDecodeStaged": PiDDecodeStaged,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDPrepare": "PiD Prepare",
    "PiDSample": "PiD Sample",
    "PiDSampleSubprocess": "PiD Sample (Subprocess)",
    "PiDFinalize": "PiD Finalize",
    "PiDDecodeStaged": "PiD Decode (Staged)",
}
