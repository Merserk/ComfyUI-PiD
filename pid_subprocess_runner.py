from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

# Make local imports work when executed as a script.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pid_decode import (  # noqa: E402
    _log_cuda_peak_memory,
    _log_pid_memory_plan,
    _load_pid_model,
    _normalize_pid_samples,
    _reset_cuda_peak_memory_stats,
    _unload_pid_model,
    _free_cuda_memory,
    _make_pid_data_batch,
    _generate_samples_low_vram,
    configure_pid_runtime,
    resolve_pid_memory_plan,
)


def _torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PiD sampling in a separate Python process.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid-steps", type=int, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sequential-offload", default="auto_low_vram")
    parser.add_argument("--pid-weight-precision", default="fp32_compatible")
    parser.add_argument("--pixel-chunk-patches", type=int, default=0)
    parser.add_argument("--aggressive-cleanup", action="store_true")
    args = parser.parse_args()

    try:
        payload = _torch_load(Path(args.input))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for PiD subprocess sampling.")

        _free_cuda_memory(aggressive=bool(args.aggressive_cleanup))
        infer_image_size = tuple(int(x) for x in payload["infer_image_size"])
        plan = resolve_pid_memory_plan(
            sequential_offload=args.sequential_offload,
            pid_weight_precision=args.pid_weight_precision,
            pixel_chunk_patches=args.pixel_chunk_patches,
            infer_image_size=infer_image_size,
        )
        _log_pid_memory_plan(plan)

        model = _load_pid_model(
            pid_dir=Path(payload["pid_dir"]),
            model_dir=Path(payload["model_dir"]),
            backbone=str(payload["backbone"]),
            ckpt_type=str(payload["pid_ckpt_type"]),
            checkpoint_path=Path(payload["checkpoint_path"]),
            dtype_choice="bf16",
            load_ema_to_reg=False,
            skip_vae_encoder=plan.skip_vae_encoder,
            pid_weight_precision=plan.pid_weight_precision,
        )

        device = "cuda"
        data_batch = _make_pid_data_batch(
            model,
            str(payload.get("caption", "")),
            float(payload["sigma"]),
            payload["latent_cpu"],
            payload.get("baseline_cpu"),
            device,
        )
        offloader = configure_pid_runtime(model, plan, device=device)

        _free_cuda_memory(aggressive=bool(args.aggressive_cleanup))
        _reset_cuda_peak_memory_stats()
        with torch.inference_mode():
            out = _generate_samples_low_vram(
                model,
                data_batch,
                cfg_scale=float(args.cfg_scale),
                num_steps=int(args.pid_steps),
                seed=int(args.seed),
                shift=None,
                image_size=infer_image_size,
            )
        _log_cuda_peak_memory("subprocess decode")

        if offloader is not None:
            offloader.cleanup()

        out = _normalize_pid_samples(out)
        out_cpu = out.detach().to("cpu")
        torch.save(
            {
                "tensor_cpu": out_cpu,
                "backbone": str(payload["backbone"]),
                "pid_ckpt_type": str(payload["pid_ckpt_type"]),
                "infer_image_size": infer_image_size,
            },
            str(args.output),
        )

        del out
        del out_cpu
        del data_batch
        _unload_pid_model(model, aggressive=bool(args.aggressive_cleanup))
        del model
        _free_cuda_memory(aggressive=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
