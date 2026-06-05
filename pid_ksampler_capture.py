from __future__ import annotations

from typing import Optional

import torch

try:
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview
except Exception:  # pragma: no cover - only available inside ComfyUI
    comfy = None
    latent_preview = None


def _sampler_names():
    if comfy is not None:
        try:
            return comfy.samplers.KSampler.SAMPLERS
        except Exception:
            pass
    return ["euler"]


EXTRA_SCHEDULERS = ["flowmatch_euler_discrete"]


def _scheduler_names():
    names = []
    if comfy is not None:
        try:
            names = list(comfy.samplers.KSampler.SCHEDULERS)
        except Exception:
            names = []
    if not names:
        names = ["beta"]
    for name in EXTRA_SCHEDULERS:
        if name not in names:
            names.append(name)
    return names


def _is_extra_scheduler(scheduler: str) -> bool:
    return str(scheduler) in EXTRA_SCHEDULERS


def _comfy_scheduler_fallback(scheduler: str) -> str:
    """Return a scheduler name Comfy's KSampler can instantiate.

    Extra PiD schedules are injected later through KSampler.sample(sigmas=...).
    KSampler still needs a valid scheduler during construction, so use a stable
    built-in fallback that is not used when custom sigmas are supplied.
    """
    if comfy is not None:
        try:
            schedulers = list(comfy.samplers.KSampler.SCHEDULERS)
            if str(scheduler) in schedulers:
                return str(scheduler)
            if "simple" in schedulers:
                return "simple"
            if schedulers:
                return schedulers[0]
        except Exception:
            pass
    return "simple"


def _flowmatch_euler_discrete_sigmas(
    steps: int,
    *,
    shift: float = 3.0,
    denoise: float = 1.0,
    device=None,
) -> torch.Tensor:
    """Diffusers FlowMatchEulerDiscreteScheduler sigma schedule.

    This mirrors diffusers' default set_timesteps() path for
    FlowMatchEulerDiscreteScheduler with use_dynamic_shifting=False and appends
    a terminal zero sigma. Z-Image and Z-Image-Turbo ship scheduler configs with
    shift=3.0, which is the default here.
    """
    steps = max(1, int(steps))
    denoise = float(denoise)
    if denoise <= 0.0:
        return torch.empty(0, dtype=torch.float32, device=device)

    # Match Comfy's partial-denoise behavior: build a longer schedule, then take
    # the tail containing the requested number of visible sampling steps.
    schedule_steps = steps if denoise > 0.9999 else max(1, int(steps / denoise))
    num_train_timesteps = 1000.0
    shift = max(float(shift), 1e-6)

    # The diffusers scheduler first initializes shifted train sigmas and stores
    # the shifted sigma_min/sigma_max, then set_timesteps() linearly interpolates
    # between those endpoints and applies the configured static shift again.
    train = torch.linspace(
        num_train_timesteps,
        1.0,
        int(num_train_timesteps),
        dtype=torch.float32,
        device=device,
    ) / num_train_timesteps
    train = shift * train / (1.0 + (shift - 1.0) * train)
    sigma_max = train[0]
    sigma_min = train[-1]

    sigmas = torch.linspace(
        sigma_max * num_train_timesteps,
        sigma_min * num_train_timesteps,
        schedule_steps,
        dtype=torch.float32,
        device=device,
    ) / num_train_timesteps
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float32, device=device)])

    if denoise <= 0.9999:
        sigmas = sigmas[-(steps + 1):]
    return sigmas


def _infer_pid_source_backbone(model) -> Optional[str]:
    candidates = [model, getattr(model, "model", None)]
    base_model = candidates[-1]
    candidates.extend(
        [
            getattr(base_model, "model_config", None),
            getattr(model, "model_config", None),
            getattr(base_model, "diffusion_model", None),
        ]
    )
    for obj in candidates:
        if obj is None:
            continue
        class_name = obj.__class__.__name__.lower()
        if "qwenimage" in class_name or "qwen_image" in class_name:
            return "qwenimage"
        unet_config = getattr(obj, "unet_config", None)
        if isinstance(unet_config, dict):
            image_model = str(unet_config.get("image_model", "")).strip().lower()
            if image_model == "qwen_image":
                return "qwenimage"
        if "zimage" in class_name or "z_image" in class_name:
            return "zimage"
    return None


def _validate_qwen_loader_options(model, source_backbone: Optional[str]) -> None:
    if source_backbone != "qwenimage":
        return
    model_options = getattr(model, "model_options", None)
    if isinstance(model_options, dict) and model_options.get("fp8_optimizations"):
        raise RuntimeError(
            "Qwen-Image PiD capture is not compatible with UNETLoader "
            "weight_dtype='fp8_e4m3fn_fast'; it can produce speckled/artifact latents. "
            "Set the Qwen diffusion model loader weight_dtype to 'default' and queue again."
        )


def _copy_latent(
    latent: dict,
    samples: torch.Tensor,
    sigma: Optional[float] = None,
    source_backbone: Optional[str] = None,
) -> dict:
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    if sigma is not None:
        out["pid_sigma"] = float(sigma)
    if source_backbone:
        out["pid_source_backbone"] = str(source_backbone)
    return out


class PiDKSamplerCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (_sampler_names(), {"default": "euler"}),
                "scheduler": (_scheduler_names(), {"default": "flowmatch_euler_discrete"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # NVIDIA's PiD save_xt_steps are counted after K completed LDM denoising
                # passes. 0 captures the initial noisy latent; values >= the effective
                # step count return the final clean latent with sigma=0.
                "capture_step": ("INT", {"default": 46, "min": 0, "max": 10000, "step": 1}),
                "flowmatch_shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01, "round": 0.001}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT")
    RETURN_NAMES = ("final_latent", "pid_latent", "pid_sigma")
    FUNCTION = "sample"
    CATEGORY = "PiD/Staged"

    def sample(
        self,
        model,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        latent_image,
        denoise: float = 1.0,
        capture_step: int = 46,
        flowmatch_shift: float = 3.0,
    ):
        if comfy is None:
            raise RuntimeError("PiD KSampler Capture must run inside ComfyUI.")

        source_backbone = _infer_pid_source_backbone(model)
        _validate_qwen_loader_options(model, source_backbone)
        latent_samples = latent_image["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent_samples,
            latent_image.get("downscale_ratio_spacial", None),
        )

        batch_inds = latent_image.get("batch_index")
        noise = comfy.sample.prepare_noise(latent_samples, int(seed), batch_inds)
        noise_mask = latent_image.get("noise_mask")

        use_extra_scheduler = _is_extra_scheduler(scheduler)
        comfy_scheduler = _comfy_scheduler_fallback(scheduler)
        sampler = comfy.samplers.KSampler(
            model,
            steps=int(steps),
            device=model.load_device,
            sampler=str(sampler_name),
            scheduler=comfy_scheduler,
            denoise=float(denoise),
            model_options=model.model_options,
        )
        if use_extra_scheduler:
            sigmas = _flowmatch_euler_discrete_sigmas(
                int(steps),
                shift=float(flowmatch_shift),
                denoise=float(denoise),
                device=getattr(model, "load_device", None),
            )
        else:
            sigmas = sampler.sigmas
        effective_steps = max(0, int(sigmas.shape[0]) - 1)
        target_step = min(max(0, int(capture_step)), effective_steps)

        captured = {"samples": None, "sigma": None}
        preview_callback = None
        if latent_preview is not None:
            try:
                preview_callback = latent_preview.prepare_callback(model, effective_steps)
            except Exception:
                preview_callback = None

        def callback(step, x0, x, total_steps):
            if preview_callback is not None:
                preview_callback(step, x0, x, total_steps)
            # Comfy/k-diffusion calls this callback before sampler step `step`. At
            # that point, `x` is already the latent after `step` completed denoising
            # passes and corresponds to sigmas[step]. NVIDIA PiD's K means "after K
            # LDM forward passes", so capture when step == K, not K-1.
            step_index = int(step)
            if captured["samples"] is None and step_index == target_step and target_step < effective_steps:
                captured["samples"] = x.detach().to("cpu").contiguous()
                try:
                    captured["sigma"] = float(sigmas[step_index].detach().float().cpu().item())
                except Exception:
                    captured["sigma"] = 0.0

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        if use_extra_scheduler:
            samples = sampler.sample(
                noise,
                positive,
                negative,
                float(cfg),
                latent_image=latent_samples,
                denoise_mask=noise_mask,
                sigmas=sigmas,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=int(seed),
            )
        else:
            samples = comfy.sample.sample(
                model,
                noise,
                int(steps),
                float(cfg),
                str(sampler_name),
                comfy_scheduler,
                positive,
                negative,
                latent_samples,
                denoise=float(denoise),
                disable_noise=False,
                noise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=int(seed),
            )

        if captured["samples"] is None:
            # capture_step >= effective_steps means final clean x0.
            captured["samples"] = samples.detach().to("cpu").contiguous()
            captured["sigma"] = 0.0

        final_latent = _copy_latent(latent_image, samples, source_backbone=source_backbone)
        pid_latent = _copy_latent(
            latent_image,
            captured["samples"],
            captured["sigma"],
            source_backbone=source_backbone,
        )
        return (final_latent, pid_latent, float(captured["sigma"]))


NODE_CLASS_MAPPINGS = {
    "PiDKSamplerCapture": PiDKSamplerCapture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDKSamplerCapture": "PiD KSampler Capture",
}
