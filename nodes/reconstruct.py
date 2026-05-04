"""HYWM2Reconstruct - WorldMirror 2.0 multi-view reconstruction.

Takes a ComfyUI IMAGE batch (>= 1 view) and runs the WorldMirror 2.0 forward
pass to produce depth, surface normals, dense point clouds, camera params,
and 3D Gaussian Splatting attributes — all in a single dict output.

The vendored upstream pipeline expects file paths, so we materialize the
input batch as PNGs in a temp dir; this is the same approach the upstream
gradio app uses.
"""

import gc
import logging
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
import comfy.model_management as mm
import comfy.model_patcher
from comfy_api.latest import io

log = logging.getLogger("hywm2")

# Empirical: ViT tokens we can afford per GB of *free* VRAM during a
# single forward pass (bf16 + FA-2). The FFN expansion is the binding
# constraint.
_TOKENS_PER_GB = 1500
_PATCH = 14
_MIN_EFFECTIVE = 224  # absolute floor (matches DINO pretraining min)


def _auto_target_size(num_views: int, requested: int, free_gb: float) -> int:
    """Snap `requested` down so total tokens stay inside the VRAM budget.

    Returns a multiple of 14 in [_MIN_EFFECTIVE, requested].
    """
    if num_views <= 0 or free_gb <= 0:
        return requested
    budget_tokens = max(2000, int(free_gb * _TOKENS_PER_GB))
    patches_per_view = budget_tokens / num_views
    eff = int(_PATCH * math.sqrt(max(patches_per_view, 1)))
    eff = (eff // _PATCH) * _PATCH
    eff = max(_MIN_EFFECTIVE, min(eff, requested))
    # Snap requested too in case caller didn't
    eff = (eff // _PATCH) * _PATCH
    return max(_MIN_EFFECTIVE, eff)


class HYWM2Reconstruct(io.ComfyNode):
    """Run WorldMirror 2.0 on a multi-view IMAGE batch.

    Outputs the raw predictions dict containing depth/normals/pts3d/camera
    poses/intrinsics/3DGS attributes; downstream nodes can decode/export
    individual fields.
    """

    # Cache a single pipeline instance across invocations.
    # Key tracks (model_dir, enable_bf16, disable_heads_tuple).
    _pipeline = None
    _pipeline_key = None
    _patcher = None  # comfy.model_patcher.ModelPatcher wrapping pipeline.model

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2Reconstruct",
            display_name="HYWM2 Reconstruct",
            category="HYWM2",
            description=(
                "Run WorldMirror 2.0 forward pass on a multi-view image batch. "
                "Provide >= 1 view in the IMAGE batch (>= 2 strongly recommended)."
            ),
            inputs=[
                io.Custom("HYWM2_MODEL").Input(
                    "model",
                    tooltip="Model handle from LoadHYWM2Model",
                ),
                io.Image.Input(
                    "images",
                    tooltip="Multi-view image batch (S >= 1; >= 2 recommended).",
                ),
                io.String.Input(
                    "prior_camera_json",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Optional path to a camera_params.json with prior extrinsics/intrinsics "
                        "(OpenCV c2w + 3x3 K). Leave blank to let the model predict cameras."
                    ),
                ),
            ],
            outputs=[
                io.Custom("HYWM2_PREDICTIONS").Output(
                    display_name="predictions",
                    tooltip=(
                        "WorldMirror predictions dict: depth, normals, pts3d, "
                        "camera_poses, camera_intrs, splats, plus the input imgs tensor."
                    ),
                ),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        model: Any,
        images: torch.Tensor,
        prior_camera_json: str,
    ):
        if not isinstance(model, dict) or "model_dir" not in model:
            raise ValueError(
                f"HYWM2Reconstruct: invalid model handle (expected dict from "
                f"LoadHYWM2Model, got {type(model).__name__})"
            )
        if images is None or images.numel() == 0:
            raise ValueError("HYWM2Reconstruct: images batch is empty")

        log.info(
            "HYWM2Reconstruct: model_dir=%s, images shape=%s, prior_camera_json=%r",
            model["model_dir"], tuple(images.shape), prior_camera_json,
        )

        # Materialize the IMAGE batch as PNGs so we can reuse the upstream
        # file-path-based preprocessing (resize + center-crop + 1/14 snap).
        with tempfile.TemporaryDirectory(prefix="hywm2_") as tmp_str:
            tmp_dir = Path(tmp_str)
            img_paths = cls._dump_image_batch(images, tmp_dir)
            log.info("HYWM2Reconstruct: wrote %d frame(s) to %s", len(img_paths), tmp_dir)

            pipeline = cls._get_pipeline(model)

            # Hand control of GPU residency to ComfyUI's model manager.
            if cls._patcher is not None:
                mm.load_models_gpu([cls._patcher])

            # Resolution policy:
            #   1. Snap requested target down to image's actual longest edge
            #      (compute_adaptive_target_size handles 1/14 snap + clamp).
            #   2. Auto-shrink further based on view count + free VRAM so a
            #      21-view 952 panorama doesn't OOM on an 8 GB card.
            from .hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
                compute_adaptive_target_size,
            )
            target_size = int(model.get("target_size", 952))
            adaptive = compute_adaptive_target_size(img_paths, target_size)
            free_gb = 0.0
            if torch.cuda.is_available():
                try:
                    free_gb = mm.get_free_memory(pipeline.device) / (1024 ** 3)
                except Exception:
                    free_gb = torch.cuda.mem_get_info(pipeline.device)[0] / (1024 ** 3)
            effective = _auto_target_size(len(img_paths), adaptive, free_gb)
            log.info(
                "HYWM2Reconstruct: views=%d requested=%d adaptive=%d free=%.1fGB -> effective=%d",
                len(img_paths), target_size, adaptive, free_gb, effective,
            )

            prior_cam = prior_camera_json.strip() or None
            if prior_cam and not Path(prior_cam).is_file():
                raise FileNotFoundError(f"prior_camera_json not found: {prior_cam}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            predictions, imgs, infer_time = pipeline._run_inference(
                img_paths=img_paths,
                target_size=effective,
                prior_cam_path=prior_cam,
                prior_depth_path=None,
            )

        log.info("HYWM2Reconstruct: forward pass done in %.2fs", infer_time)

        out = {
            "predictions": predictions,
            "imgs": imgs,
            "infer_time": infer_time,
            "target_size": effective,
            "model_handle": model,
        }
        return io.NodeOutput(out)

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def _get_pipeline(cls, model_handle: dict):
        """Lazy-load and cache the WorldMirrorPipeline + a ComfyUI ModelPatcher.

        Reloads if the loader handle changed (e.g. user toggled bf16 or
        disabled a head between runs). Caller is responsible for
        ``mm.load_models_gpu([cls._patcher])`` before forward.
        """
        key = (
            model_handle["model_dir"],
            bool(model_handle.get("enable_bf16", True)),
            tuple(sorted(model_handle.get("disable_heads", []) or [])),
        )
        if cls._pipeline is not None and cls._pipeline_key == key:
            return cls._pipeline

        # Free any previous instance before loading a new one.
        if cls._patcher is not None:
            try:
                cls._patcher.unpatch_model(device_to=cls._patcher.offload_device)
                cls._patcher.cleanup()
            except Exception:
                pass
            cls._patcher = None
        if cls._pipeline is not None:
            try:
                cls._pipeline.model.cpu()
            except Exception:
                pass
            cls._pipeline = None
            cls._pipeline_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info("HYWM2Reconstruct: loading WorldMirror pipeline...")
        from .hyworld2.worldrecon.pipeline import WorldMirrorPipeline

        pipeline = WorldMirrorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_handle["model_dir"],
            subfolder="",
            enable_bf16=bool(model_handle.get("enable_bf16", True)),
            disable_heads=list(model_handle.get("disable_heads") or []) or None,
        )

        # Move to CPU so ComfyUI ModelPatcher owns GPU residency policy.
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        try:
            pipeline.model.to(offload_device)
            pipeline.device = load_device  # pipeline.forward will see this
        except Exception as e:
            log.warning("HYWM2Reconstruct: failed to offload pipeline.model to CPU: %s", e)

        # Wrap in ModelPatcher so subsequent load_models_gpu cooperates with
        # the rest of ComfyUI's VRAM budget (other workflows can evict us).
        cls._patcher = comfy.model_patcher.ModelPatcher(
            pipeline.model,
            load_device=load_device,
            offload_device=offload_device,
        )

        cls._pipeline = pipeline
        cls._pipeline_key = key
        log.info("HYWM2Reconstruct: pipeline + patcher ready (load=%s offload=%s)",
                 load_device, offload_device)
        return pipeline

    # ------------------------------------------------------------------
    # ComfyUI IMAGE -> file paths
    # ------------------------------------------------------------------
    @staticmethod
    def _dump_image_batch(images: torch.Tensor, out_dir: Path) -> list:
        """Save a [B,H,W,C] float[0,1] IMAGE batch to PNGs and return paths."""
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4 or images.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"HYWM2Reconstruct: expected IMAGE shape [B,H,W,C], got {tuple(images.shape)}"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        np_imgs = images.detach().cpu().clamp(0, 1).numpy()
        for i, frame in enumerate(np_imgs):
            arr = (frame * 255.0 + 0.5).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            path = out_dir / f"frame_{i:04d}.png"
            Image.fromarray(arr).save(path)
            paths.append(str(path))
        return paths
