"""HYWM2Reconstruct — WorldMirror 2.0 multi-view reconstruction.

Takes a ComfyUI IMAGE batch (>= 1 view) and emits SIX typed outputs:
  - images                 (IMAGE)             — depth viz batch
  - normals                (IMAGE)             — surface-normals viz batch
  - points                 (HYWM2_POINTS)      — colored point cloud
  - gaussians              (HYWM2_GAUSSIANS)   — 3DGS attributes
  - predicted_extrinsics   (EXTRINSICS)        — w2c [N,4,4] (CameraPack)
  - predicted_intrinsics   (INTRINSICS)        — K     [N,3,3]

The vendored upstream pipeline expects file paths, so we materialize the
input batch as PNGs in a temp dir; same approach the upstream gradio app uses.
"""

import gc
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io
# NOTE: `comfy.model_management` and `comfy.model_patcher` are imported
# lazily inside the methods that use them. Importing them at module load
# triggers torch.cuda.current_device(), which crashes on CPU-only CI
# hosts where torch is built without CUDA.

log = logging.getLogger("hywm2")
# In the comfy-env subprocess worker the root logger isn't wired to
# stderr, so plain log.info() silently dies. Attach a stderr handler
# (matching the _p()/stderr-print pattern used by other nodes in the
# pack). Idempotent so reconstruct.py + decode_export.py can both call.
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[hywm2] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

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
                    "gaussians_conf_percentile",
                    default=10.0, min=0.0, max=99.0, step=1.0,
                    tooltip="Drop bottom N%% of Gaussians by confidence "
                            "(0 = keep all). Matches upstream's default "
                            "(confidence_percentile=10.0 in pipeline.py). "
                            "Confidence source priority: pts3d_conf > "
                            "depth_conf > splats['weights']. If pts_head / "
                            "depth_head are disabled (only gs enabled), "
                            "falls back to the gs_head's per-Gaussian "
                            "`weights` channel.",
                ),
                io.Float.Input(
                    "gaussians_voxel_size",
                    default=0.002, min=0.0, max=1.0, step=0.001,
                    tooltip="Voxel-merge Gaussians whose means land in the "
                            "same voxel-size cube. Weighted average via "
                            "splats['weights'] (means/scales/colors/opacities) "
                            "+ summed-weighted-then-renormalized quaternions. "
                            "Matches upstream default (0.002 world units, "
                            "~2mm if scene is in meters). 0 = disabled. "
                            "Dramatic dedup in flat regions; preserves "
                            "geometric detail in textured ones. Applied "
                            "AFTER conf filter, BEFORE random downsample.",
                ),
                io.Float.Input(
                    "gaussians_downsample",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Random keep ratio for Gaussians (1.0 = keep all, 0.1 = keep 10%%).",
                ),
                io.Int.Input(
                    "target_size_override",
                    default=0, min=0, max=2048, step=14, optional=True,
                    tooltip="EXPLICIT inference resolution (longest edge in "
                            "pixels, snapped to multiples of 14). 0 = use "
                            "LoadHYWM2Model.target_size + VRAM auto-shrink "
                            "(current default behaviour). >0 = use this exact "
                            "value, bypass both the model default AND the "
                            "VRAM auto-shrink heuristic. Useful when you know "
                            "your image's native size and want to run there "
                            "(e.g., 832 -> set 826 = 14*59).",
                ),
                io.Boolean.Input(
                    "bypass_vram_shrink",
                    default=False, optional=True,
                    tooltip="When True, skip the VRAM-based auto-shrink "
                            "(`_auto_target_size`) and use the adaptive "
                            "size (min of model.target_size and image's "
                            "longest-edge-snapped-to-14). Will OOM if there "
                            "isn't enough VRAM for that resolution at the "
                            "given view count -- you take responsibility. "
                            "Ignored if `target_size_override` > 0 (override "
                            "already bypasses)."),
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
                io.Custom("EXTRINSICS").Output(
                    display_name="predicted_extrinsics",
                    tooltip=(
                        "Predicted camera extrinsics tensor [N,4,4] in world-to-camera "
                        "(CameraPack) convention — i.e. the inverse of the model's c2w "
                        "camera_poses. First view is normalized to identity by upstream."
                    ),
                ),
                io.Custom("INTRINSICS").Output(
                    display_name="predicted_intrinsics",
                    tooltip=(
                        "Predicted camera intrinsics tensor [N,3,3] (K matrix per view, "
                        "in pixel units of the effective inference resolution)."
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
        gaussians_conf_percentile: float = 10.0,
        gaussians_voxel_size: float = 0.002,
        gaussians_downsample: float = 1.0,
        target_size_override: int = 0,
        bypass_vram_shrink: bool = False,
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

            # Lazy import — comfy.model_management probes CUDA at load time
            # and crashes on CPU-only CI hosts.
            import comfy.model_management as mm

            # Hand control of GPU residency to ComfyUI's model manager.
            # When free VRAM is below model_size, ModelPatcher.partially_load
            # will stream the manual_cast-wrapped Linear/LN/Conv layers per
            # forward via comfy.ops.cast_bias_weight (see _rewrite_to_manual_cast).
            if cls._patcher is not None:
                if torch.cuda.is_available():
                    cls._log_vram("pre-load_models_gpu", pipeline.device, mm)
                mm.load_models_gpu([cls._patcher])
                if torch.cuda.is_available():
                    cls._log_vram("post-load_models_gpu", pipeline.device, mm)

            # Resolution policy:
            #   1. Snap requested target down to image's actual longest edge
            #      (compute_adaptive_target_size handles 1/14 snap + clamp).
            #   2. Auto-shrink further based on view count + free VRAM so a
            #      21-view 952 panorama doesn't OOM on an 8 GB card.
            from .hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
                compute_adaptive_target_size,
            )
            # Resolution policy.
            # Priority order:
            #   1. `target_size_override > 0`  -> use that exact value (snap to /14),
            #      bypass both the model default AND the VRAM auto-shrink.
            #   2. `bypass_vram_shrink=True`   -> compute `adaptive` from model
            #      default + image native size; skip the VRAM auto-shrink.
            #   3. Default                     -> model.target_size -> adaptive
            #      -> _auto_target_size (current behaviour).
            free_gb = 0.0
            if torch.cuda.is_available():
                try:
                    free_gb = mm.get_free_memory(pipeline.device) / (1024 ** 3)
                except Exception:
                    free_gb = torch.cuda.mem_get_info(pipeline.device)[0] / (1024 ** 3)

            if int(target_size_override) > 0:
                # Snap user value to multiple of 14, clamp to >= _MIN_EFFECTIVE.
                snapped = (int(target_size_override) // _PATCH) * _PATCH
                effective = max(_MIN_EFFECTIVE, snapped)
                log.info(
                    "HYWM2Reconstruct: target_size_override=%d -> effective=%d "
                    "(bypassing both model default and VRAM auto-shrink; views=%d, free=%.1fGB)",
                    target_size_override, effective, len(img_paths), free_gb,
                )
            else:
                target_size = int(model.get("target_size", 952))
                adaptive = compute_adaptive_target_size(img_paths, target_size)
                if bypass_vram_shrink:
                    effective = adaptive
                    log.info(
                        "HYWM2Reconstruct: bypass_vram_shrink=True -> "
                        "views=%d requested=%d adaptive=%d free=%.1fGB -> effective=%d "
                        "(VRAM auto-shrink skipped; OOM is on you)",
                        len(img_paths), target_size, adaptive, free_gb, effective,
                    )
                else:
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
                cls._log_vram("pre-forward", pipeline.device, mm)
                torch.cuda.reset_peak_memory_stats(pipeline.device)

            try:
                predictions, imgs, infer_time = pipeline._run_inference(
                    img_paths=img_paths,
                    target_size=effective,
                    prior_cam_path=prior_cam,
                    prior_depth_path=None,
                )
            except torch.cuda.OutOfMemoryError:
                cls._log_oom_diagnostic(pipeline.device, effective, len(img_paths), mm)
                raise

            if torch.cuda.is_available():
                cls._log_vram("post-forward", pipeline.device, mm)

        log.info("HYWM2Reconstruct: forward pass done in %.2fs", infer_time)

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
        gaussians = decode_gaussians(
            predictions,
            weight_threshold=gaussians_weight_threshold,
            conf_percentile=gaussians_conf_percentile,
            voxel_size=gaussians_voxel_size,
            downsample=gaussians_downsample,
        )

        # ----- predicted cameras (CameraPack convention: w2c [N,4,4], K [N,3,3]) -----
        c2w = predictions.get("camera_poses")          # [B,S,4,4] (OpenCV c2w)
        intr = predictions.get("camera_intrs")         # [B,S,3,3]
        if c2w is not None:
            c2w_t = c2w.detach().float().cpu()
            if c2w_t.dim() == 4 and c2w_t.shape[0] == 1:
                c2w_t = c2w_t[0]                       # [S,4,4]
            extrinsics_w2c = torch.linalg.inv(c2w_t)
        else:
            extrinsics_w2c = torch.eye(4).unsqueeze(0)
        if intr is not None:
            intr_t = intr.detach().float().cpu()
            if intr_t.dim() == 4 and intr_t.shape[0] == 1:
                intr_t = intr_t[0]                     # [S,3,3]
        else:
            intr_t = torch.eye(3).unsqueeze(0)

        # Summary so the user can see at a glance which outputs are populated.
        empty = []
        if "depth" not in predictions:    empty.append("images(depth)")
        if "normals" not in predictions:  empty.append("normals")
        if "pts3d" not in predictions:    empty.append("points")
        if "splats" not in predictions:   empty.append("gaussians")
        if c2w is None or intr is None:   empty.append("cameras")
        if empty:
            log.info("HYWM2Reconstruct: empty outputs (head disabled): %s", ", ".join(empty))

        return io.NodeOutput(depth_img, normals_img, points, gaussians, extrinsics_w2c, intr_t)

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

        # Lazy imports — comfy.model_management + comfy.model_patcher probe
        # CUDA at module load and crash on CPU-only CI hosts.
        import comfy.model_management as mm
        import comfy.model_patcher

        # ComfyUI-native partial-offload setup. The model is already built
        # from comfy.ops.disable_weight_init.* leaves (Linear / LayerNorm /
        # Conv*d / ConvTranspose*d), so ModelPatcher.partially_load() can
        # stream any layer it can't fit -- just by flipping the layer's
        # `comfy_cast_weights` flag. We only need to:
        #   1. Move the model to offload_device (so ModelPatcher.load()
        #      owns every device transfer from this point on).
        #   2. Pin tiny buffers (_resnet_mean/std, RoPE tables, etc.) to
        #      load_device. ModelPatcher only walks parameters; buffers
        #      stay wherever .to() left them. Without this, the very first
        #      forward op `(images - _resnet_mean) / _resnet_std` would
        #      crash with a CPU vs CUDA device-mismatch.
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        try:
            pipeline.model.to(offload_device)
            pipeline.device = load_device  # pipeline.forward will see this
        except Exception as e:
            log.warning("HYWM2Reconstruct: failed to offload pipeline.model to CPU: %s", e)
        cls._pin_buffers_to_device(pipeline.model, load_device)

        # Wrap in ModelPatcher so load_models_gpu can budget against the
        # rest of ComfyUI's VRAM (other workflows can evict us; we'll
        # auto-stream layers that don't fit via cast_bias_weight).
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
    # Buffer pinning helper (ModelPatcher only touches parameters)
    # ------------------------------------------------------------------
    @classmethod
    def _pin_buffers_to_device(cls, model, device) -> int:
        """Move all module buffers to `device` and keep them there.

        ModelPatcher streams *parameters* via cast_bias_weight, but never
        touches buffers (they're not in `_load_list`). The HYWM2 backbone
        has tiny but critical buffers like `_resnet_mean` / `_resnet_std`
        that the very first op of the forward (`(images - mean) / std`)
        reads. If we leave them on CPU after `.to(offload_device)`, the
        input (on GPU) hits a device-mismatch crash. Solution: keep all
        buffers resident on `device` permanently. They're tiny -- the
        whole model has < 10 MB of buffer data.
        """
        n_moved = 0
        for module in model.modules():
            for name, buf in list(module._buffers.items()):
                if buf is None or buf.device == device:
                    continue
                module._buffers[name] = buf.to(device)
                n_moved += 1
        log.info("HYWM2Reconstruct: pinned %d buffers to %s", n_moved, device)
        return n_moved

    # ------------------------------------------------------------------
    # VRAM accounting helpers (so the user can see the actual numbers)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_vram(label: str, device, mm) -> None:
        """Snapshot allocated / reserved / free / total VRAM on `device`."""
        if not torch.cuda.is_available():
            return
        GB = 1024 ** 3
        try:
            free_b = mm.get_free_memory(device)
        except Exception:
            free_b = torch.cuda.mem_get_info(device)[0]
        total_b = torch.cuda.get_device_properties(device).total_memory
        alloc_b = torch.cuda.memory_allocated(device)
        reserv_b = torch.cuda.memory_reserved(device)
        peak_b = torch.cuda.max_memory_allocated(device)
        log.info(
            "HYWM2 VRAM[%s] alloc=%.2fGB peak=%.2fGB reserved=%.2fGB free=%.2fGB / total=%.2fGB",
            label, alloc_b / GB, peak_b / GB, reserv_b / GB, free_b / GB, total_b / GB,
        )

    @classmethod
    def _log_oom_diagnostic(cls, device, target_size: int, n_views: int, mm) -> None:
        """When the forward OOMs, dump a full VRAM picture + a back-of-the-
        envelope estimate of the worst single activation in the head path so
        the user can see HOW close they are to fitting.
        """
        cls._log_vram("OOM", device, mm)
        # Activation cost estimate at the input resolution. Patch dims are
        # target_size / 14 on the longer side; we assume an aspect close to
        # what we saw (476x826) for the shorter side. The biggest single
        # intermediate is the gs_head conv1 output: [B*S, 256, H, W] in the
        # model's compute dtype (bf16 by default).
        # NOTE: this is a coarse estimate; real peak includes residuals,
        # backbone activations, and grad-free workspace.
        # Use a 476:826 ratio (the observed working shape) as a proxy.
        long_edge = target_size
        short_edge = int(round(target_size * 476 / 826 / 14)) * 14
        gs_conv1_bytes = n_views * 256 * long_edge * short_edge * 2  # bf16
        gs_feats_bytes = n_views * 128 * long_edge * short_edge * 2
        GB = 1024 ** 3
        log.info(
            "HYWM2 OOM est: forward shape ~ [%d, *, %d, %d]; "
            "gs_head conv1 activation alone ~ %.2f GB bf16, "
            "gs_feats input ~ %.2f GB bf16. "
            "Most likely OOM cause is one of these (head activation, not weights). "
            "Mitigations: (a) reduce target_size_override, (b) reduce view count, "
            "(c) chunk gs_head along view dim (see rasterization.py:186).",
            n_views, short_edge, long_edge,
            gs_conv1_bytes / GB, gs_feats_bytes / GB,
        )

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
