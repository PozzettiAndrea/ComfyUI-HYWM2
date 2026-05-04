"""HYWM2Reconstruct — WorldMirror 2.0 multi-view reconstruction.

Takes a ComfyUI IMAGE batch (>= 1 view) and emits FIVE typed outputs:
  - images       (IMAGE)              — depth viz batch
  - normals      (IMAGE)              — surface-normals viz batch
  - points       (HYWM2_POINTS)       — colored point cloud (means + colors)
  - gaussians    (HYWM2_GAUSSIANS)    — 3DGS attributes (means/quats/scales/...)
  - predictions  (HYWM2_PREDICTIONS)  — full wrapper dict for power users

The vendored upstream pipeline expects file paths, so we materialize the
input batch as PNGs in a temp dir; same approach the upstream gradio app uses.
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

    Emits depth-viz / normals-viz IMAGE batches, a HYWM2_POINTS colored
    point cloud, a HYWM2_GAUSSIANS 3DGS dict, and the full predictions
    wrapper as a fallback for downstream introspection.
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
                io.Custom("EXTRINSICS").Input(
                    "prior_extrinsics",
                    optional=True,
                    tooltip=(
                        "Optional camera extrinsics tensor [N,4,4] (world-to-camera, "
                        "CameraPack convention). Wired automatically by HYWM2SamplePanorama."
                    ),
                ),
                io.Custom("INTRINSICS").Input(
                    "prior_intrinsics",
                    optional=True,
                    tooltip=(
                        "Optional camera intrinsics tensor [N,3,3] (or [3,3] broadcast to all views). "
                        "Wired automatically by HYWM2SamplePanorama."
                    ),
                ),
                io.String.Input(
                    "prior_camera_json",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Optional path to a camera_params.json with prior extrinsics/intrinsics. "
                        "Leave blank if you used the EXTRINSICS / INTRINSICS inputs above."
                    ),
                ),
                # ----- decode parameters (formerly Decode* nodes) -----
                io.Combo.Input(
                    "depth_colormap",
                    options=["viridis", "grayscale"],
                    default="viridis",
                    tooltip="Colormap for the `images` (depth-viz) output.",
                ),
                io.Boolean.Input(
                    "apply_mask",
                    default=True,
                    tooltip="Mask invalid pixels in depth/normals viz and drop them from the point cloud.",
                ),
                io.Int.Input(
                    "view_index",
                    default=-1, min=-1, max=64,
                    tooltip="Which view to emit in `images` / `normals`. -1 = all.",
                ),
                io.Float.Input(
                    "points_conf_percentile",
                    default=0.0, min=0.0, max=99.0, step=1.0,
                    tooltip="Drop bottom N%% of points by pts3d_conf (0 = keep all).",
                ),
                io.Int.Input(
                    "points_max",
                    default=2_000_000, min=10_000, max=20_000_000,
                    tooltip="Max points after filtering; randomly subsampled if exceeded.",
                ),
                io.Float.Input(
                    "gaussians_weight_threshold",
                    default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Drop Gaussians below this per-point weight (0 = keep all).",
                ),
                io.Float.Input(
                    "gaussians_downsample",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Random keep ratio for Gaussians (1.0 = keep all, 0.1 = keep 10%%).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="Depth visualization as IMAGE batch (one frame per view).",
                ),
                io.Image.Output(
                    display_name="normals",
                    tooltip="Surface normals visualization (RGB = 0.5·(n+1)).",
                ),
                io.Custom("HYWM2_POINTS").Output(
                    display_name="points",
                    tooltip="Colored point cloud (means + per-vertex RGB).",
                ),
                io.Custom("HYWM2_GAUSSIANS").Output(
                    display_name="gaussians",
                    tooltip="3DGS attributes (means/quats/scales/opacities/rgbs).",
                ),
                io.Custom("HYWM2_PREDICTIONS").Output(
                    display_name="predictions",
                    tooltip=(
                        "Full WorldMirror predictions wrapper: depth, normals, pts3d, "
                        "camera_poses, camera_intrs, splats, imgs, infer_time, target_size."
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
        prior_camera_json: str = "",
        prior_extrinsics: torch.Tensor | None = None,
        prior_intrinsics: torch.Tensor | None = None,
        depth_colormap: str = "viridis",
        apply_mask: bool = True,
        view_index: int = -1,
        points_conf_percentile: float = 0.0,
        points_max: int = 2_000_000,
        gaussians_weight_threshold: float = 0.0,
        gaussians_downsample: float = 1.0,
    ):
        if not isinstance(model, dict) or "model_dir" not in model:
            raise ValueError(
                f"HYWM2Reconstruct: invalid model handle (expected dict from "
                f"LoadHYWM2Model, got {type(model).__name__})"
            )
        if images is None or images.numel() == 0:
            raise ValueError("HYWM2Reconstruct: images batch is empty")

        log.info(
            "HYWM2Reconstruct: model_dir=%s, images shape=%s, prior_camera_json=%r, "
            "prior_extrinsics=%s, prior_intrinsics=%s",
            model["model_dir"], tuple(images.shape), prior_camera_json,
            None if prior_extrinsics is None else tuple(prior_extrinsics.shape),
            None if prior_intrinsics is None else tuple(prior_intrinsics.shape),
        )

        # Materialize the IMAGE batch as PNGs so we can reuse the upstream
        # file-path-based preprocessing (resize + center-crop + 1/14 snap).
        with tempfile.TemporaryDirectory(prefix="hywm2_") as tmp_str:
            tmp_dir = Path(tmp_str)
            img_paths = cls._dump_image_batch(images, tmp_dir)
            log.info("HYWM2Reconstruct: wrote %d frame(s) to %s", len(img_paths), tmp_dir)

            # If the user passed in tensors, materialize them as the prior
            # JSON the upstream pipeline expects.
            if (prior_extrinsics is not None) or (prior_intrinsics is not None):
                if prior_camera_json.strip():
                    log.warning(
                        "HYWM2Reconstruct: both tensor priors and prior_camera_json provided; "
                        "tensor priors win.")
                prior_camera_json = cls._dump_camera_priors_json(
                    img_paths, tmp_dir, prior_extrinsics, prior_intrinsics
                )

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

        wrapper = {
            "predictions": predictions,
            "imgs": imgs,
            "infer_time": infer_time,
            "target_size": effective,
            "model_handle": model,
        }

        # ----- inline decode (formerly Decode* nodes) -----
        from .decode_export import (
            decode_depth_image, decode_normals_image,
            decode_points, decode_gaussians,
        )

        depth_img = decode_depth_image(
            predictions, view_index=view_index,
            apply_mask=apply_mask, colormap=depth_colormap,
        )
        normals_img = decode_normals_image(
            predictions, view_index=view_index, apply_mask=apply_mask,
        )
        points = decode_points(
            predictions, imgs,
            apply_mask=apply_mask,
            conf_percentile=points_conf_percentile,
            max_points=points_max,
        )
        try:
            gaussians = decode_gaussians(
                predictions,
                weight_threshold=gaussians_weight_threshold,
                downsample=gaussians_downsample,
            )
        except RuntimeError as e:
            # Happens if the gs head was disabled at load time.
            log.warning("HYWM2Reconstruct: gaussians decode skipped: %s", e)
            empty = torch.zeros((0, 3))
            gaussians = {
                "means": empty, "quats": torch.zeros((0, 4)),
                "scales": empty, "opacities": torch.zeros((0,)),
                "rgbs": empty,
            }

        return io.NodeOutput(depth_img, normals_img, points, gaussians, wrapper)

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

    @staticmethod
    def _dump_camera_priors_json(
        img_paths: list,
        out_dir: Path,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
    ) -> str:
        """Convert in-memory EXTRINSICS (w2c) + INTRINSICS (3x3) tensors to the
        JSON format that upstream `load_prior_camera` expects.

        - EXTRINSICS in: [N,4,4] world-to-camera (CameraPack convention).
        - WorldMirror wants camera-to-world, so we invert per-frame.
        - INTRINSICS in: [N,3,3] or [3,3] (broadcast). We pass through; upstream
          applies the resize+crop transform to align with the preprocessed image.
        """
        import json

        N = len(img_paths)
        stems = [Path(p).stem for p in img_paths]
        out: dict = {}

        if extrinsics is not None:
            ext = extrinsics.detach().float().cpu()
            if ext.dim() == 4 and ext.shape[0] == 1:
                ext = ext[0]                              # [N,4,4]
            if ext.dim() == 3 and ext.shape[1:] == (3, 4):  # accept 3x4
                pad = torch.tensor([0, 0, 0, 1], dtype=ext.dtype).expand(ext.shape[0], 1, 4)
                ext = torch.cat([ext, pad], dim=1)
            if ext.dim() != 3 or ext.shape[1:] != (4, 4):
                raise ValueError(
                    f"prior_extrinsics: expected [N,4,4], got {tuple(ext.shape)}"
                )
            if ext.shape[0] != N:
                raise ValueError(
                    f"prior_extrinsics N={ext.shape[0]} != number of images {N}"
                )
            # w2c -> c2w
            c2w = torch.linalg.inv(ext)
            out["extrinsics"] = [
                {"camera_id": stems[i], "matrix": c2w[i].tolist()} for i in range(N)
            ]

        if intrinsics is not None:
            intr = intrinsics.detach().float().cpu()
            if intr.dim() == 2:
                intr = intr.unsqueeze(0).expand(N, 3, 3).contiguous()
            elif intr.dim() == 3 and intr.shape[0] == 1 and N > 1:
                intr = intr.expand(N, 3, 3).contiguous()
            if intr.dim() == 4 and intr.shape[0] == 1:
                intr = intr[0]
            # Allow 4x4 fall-through (drop homogenous row/col)
            if intr.dim() == 3 and intr.shape[1:] == (4, 4):
                intr = intr[:, :3, :3].contiguous()
            if intr.dim() != 3 or intr.shape[1:] != (3, 3):
                raise ValueError(
                    f"prior_intrinsics: expected [N,3,3] (or [3,3]), got {tuple(intr.shape)}"
                )
            if intr.shape[0] != N:
                raise ValueError(
                    f"prior_intrinsics N={intr.shape[0]} != number of images {N}"
                )
            out["intrinsics"] = [
                {"camera_id": stems[i], "matrix": intr[i].tolist()} for i in range(N)
            ]

        json_path = out_dir / "prior_camera.json"
        with json_path.open("w") as f:
            json.dump(out, f)
        return str(json_path)
