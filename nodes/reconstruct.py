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
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
import comfy.model_management as mm
from comfy_api.latest import io

log = logging.getLogger("hywm2")


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

            # Adaptive resolution: snap to multiples of 14, capped at target_size.
            from .hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
                compute_adaptive_target_size,
            )
            target_size = int(model.get("target_size", 952))
            effective = compute_adaptive_target_size(img_paths, target_size)
            log.info("HYWM2Reconstruct: target_size=%d effective=%d", target_size, effective)

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
        """Lazy-load and cache the WorldMirrorPipeline.

        Reloads if the loader handle changed (e.g. user toggled bf16 or
        disabled a head between runs).
        """
        key = (
            model_handle["model_dir"],
            bool(model_handle.get("enable_bf16", True)),
            tuple(sorted(model_handle.get("disable_heads", []) or [])),
        )
        if cls._pipeline is not None and cls._pipeline_key == key:
            # Make sure the cached pipeline is on GPU (ComfyUI may have
            # offloaded it between runs to free VRAM for other models).
            try:
                cls._pipeline.model.to(cls._pipeline.device)
                cls._pipeline.model.eval()
            except Exception:
                pass
            return cls._pipeline

        # Free any previous instance before loading a new one.
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
        cls._pipeline = pipeline
        cls._pipeline_key = key
        log.info("HYWM2Reconstruct: pipeline ready")
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
