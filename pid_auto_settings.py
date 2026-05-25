from __future__ import annotations

from typing import Dict, Tuple

try:
    from .pid_decode import BACKBONE_CHOICES, PID_BACKBONES, _latent_samples, _vram_total_gb
except ImportError:  # Allows `python -c "import pid_auto_settings"` from this folder.
    from pid_decode import BACKBONE_CHOICES, PID_BACKBONES, _latent_samples, _vram_total_gb


AUTO_BACKBONE_CHOICES = ["auto"] + BACKBONE_CHOICES
PRESET_CHOICES = ["safe", "balanced", "quality", "4k"]

_COMPRESSION_BY_BACKBONE = {
    "zimage": 8,
    "flux": 8,
    "sd3": 8,
    "flux2": 16,
    "dinov2": 16,
    "siglip": 16,
}

_BACKBONE_BY_CHANNELS = {
    16: "zimage",
    128: "flux2",
    768: "dinov2",
    1152: "siglip",
}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _scale_for_target(base_w: int, base_h: int, target_max: int, max_scale: int) -> int:
    base_max = max(1, int(max(base_w, base_h)))
    return _clamp(target_max // base_max, 1, max_scale)


def _choose_settings(backbone: str, base_w: int, base_h: int, preset: str) -> Tuple[str, int]:
    info = PID_BACKBONES[backbone]
    base_max = max(base_w, base_h)
    default_scale = int(info.default_scale)
    supports_4k_ckpt = "2kto4k" in info.ckpt_types

    if preset == "safe":
        return "2k", _scale_for_target(base_w, base_h, 1536, default_scale)

    if preset == "balanced":
        return "2k", _scale_for_target(base_w, base_h, 2048, default_scale)

    if preset == "quality":
        if supports_4k_ckpt and base_max >= 896:
            return "2kto4k", _scale_for_target(base_w, base_h, 3072, default_scale)
        return "2k", _scale_for_target(base_w, base_h, 2048, default_scale)

    if preset == "4k":
        if supports_4k_ckpt:
            return "2kto4k", default_scale
        return "2k", default_scale

    return "2k", _scale_for_target(base_w, base_h, 2048, default_scale)


class PiDAutoSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "backbone": (AUTO_BACKBONE_CHOICES, {"default": "auto"}),
                "preset": (PRESET_CHOICES, {"default": "balanced"}),
                "base_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "base_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("PID_SETTINGS", "STRING", "STRING", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "auto_settings",
        "backbone",
        "pid_ckpt_type",
        "pid_steps",
        "scale",
        "cfg_scale",
        "sigma",
        "status",
    )
    FUNCTION = "build"
    CATEGORY = "PiD"

    def build(
        self,
        latent,
        backbone: str,
        preset: str,
        base_width: int = 0,
        base_height: int = 0,
    ):
        samples = _latent_samples(latent)
        channels = int(samples.shape[1])

        if backbone == "auto":
            backbone = _BACKBONE_BY_CHANNELS.get(channels, "zimage")

        info = PID_BACKBONES[backbone]
        compression = _COMPRESSION_BY_BACKBONE[backbone]
        inferred_w = int(samples.shape[-1]) * compression
        inferred_h = int(samples.shape[-2]) * compression
        base_w = int(base_width) if int(base_width) > 0 else inferred_w
        base_h = int(base_height) if int(base_height) > 0 else inferred_h

        ckpt_type, scale = _choose_settings(backbone, base_w, base_h, preset)
        out_w = base_w * scale
        out_h = base_h * scale

        settings: Dict[str, object] = {
            "backbone": backbone,
            "pid_ckpt_type": ckpt_type,
            "pid_steps": 4,
            "scale": scale,
            "cfg_scale": 1.0,
            "sigma": 0.0,
            "unload_comfy_before_pid": True,
            "aggressive_cleanup": True,
            "base_width": base_w,
            "base_height": base_h,
            "output_width": out_w,
            "output_height": out_h,
            "preset": preset,
        }

        notes = []
        if channels != info.latent_channels:
            notes.append(f"warning: {info.label} expects {info.latent_channels} channels, latent has {channels}")
        vram_gb = _vram_total_gb()
        if vram_gb and max(out_w, out_h) >= 3840 and vram_gb < 24:
            notes.append(f"warning: estimated {out_w}x{out_h} output may be heavy on {vram_gb:.1f}GB VRAM")

        status = (
            f"PiD Auto Settings: backbone={backbone}, preset={preset}, "
            f"checkpoint={ckpt_type}, scale={scale}, base={base_w}x{base_h}, output={out_w}x{out_h}."
        )
        if notes:
            status += "\n" + "\n".join(notes)
        return (
            settings,
            str(settings["backbone"]),
            str(settings["pid_ckpt_type"]),
            int(settings["pid_steps"]),
            int(settings["scale"]),
            float(settings["cfg_scale"]),
            float(settings["sigma"]),
            status,
        )


NODE_CLASS_MAPPINGS = {
    "PiDAutoSettings": PiDAutoSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDAutoSettings": "PiD Auto Settings",
}
