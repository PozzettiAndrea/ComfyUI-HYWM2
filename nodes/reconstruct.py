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

_PATCH = 14
_MIN_EFFECTIVE = 224  # absolute floor (matches DINO pretraining min)


# ---------------------------------------------------------------------------
# Camera-frame rebasing helpers (MIRROR of train_gaussians.py — keep in sync)
#
# HYWM2's reconstruction pipeline anchors view 0 to identity in its internal
# world frame for numerical stability of the pose head. As a side effect, the
# output gaussians + predicted_extrinsics live in this rebased frame, NOT in
# the prior_extrinsics frame the user supplied. We undo the rebase at the
# output boundary so downstream consumers (training, preview, export) see
# everything in the original prior frame — "just works" instead of needing
# special handling.
# ---------------------------------------------------------------------------

def _matrix_to_quaternion_wxyz(R: torch.Tensor) -> torch.Tensor:
    """3×3 rotation matrix -> wxyz unit quaternion (Shepperd's method)."""
    R = R.float().cpu()
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = float((1.0 + trace).sqrt() * 2.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]).item() / s
        y = (R[0, 2] - R[2, 0]).item() / s
        z = (R[1, 0] - R[0, 1]).item() / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = float((1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2.0)
        w = (R[2, 1] - R[1, 2]).item() / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]).item() / s
        z = (R[0, 2] + R[2, 0]).item() / s
    elif R[1, 1] >= R[2, 2]:
        s = float((1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2.0)
        w = (R[0, 2] - R[2, 0]).item() / s
        x = (R[0, 1] + R[1, 0]).item() / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]).item() / s
    else:
        s = float((1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2.0)
        w = (R[1, 0] - R[0, 1]).item() / s
        x = (R[0, 2] + R[2, 0]).item() / s
        y = (R[1, 2] + R[2, 1]).item() / s
        z = 0.25 * s
    q = torch.tensor([w, x, y, z], dtype=torch.float32)
    return q / q.norm()


def _quat_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2 in wxyz convention. Broadcasts on leading dims."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _rebase_outputs_to_prior_frame(
    gaussians: dict | None,
    predicted_w2c: torch.Tensor,
    prior_w2c: torch.Tensor,
) -> tuple[dict | None, torch.Tensor]:
    """Transform HYWM2 outputs from its rebased world frame back to the
    user's `prior_extrinsics` frame.

    Both `predicted_w2c` and `prior_w2c` describe the SAME 12 physical
    cameras, just in two different world conventions. The rigid transform
    T_h2p = inv(W_h_0) @ W_p_0 ... actually:

        g_p = inv(W_p_0) @ W_h_0 @ g_h   (point in prior world)
              ───────────────────
              T_h2p

    With HYWM2 anchoring `W_h_0 ≈ I`, T_h2p ≈ W_p_0 — i.e. apply prior's
    view-0 w2c to the HYWM2 gaussians and predicted-extrinsics get a
    matching post-multiplication by inv(T_h2p).

    Transforms:
      - gaussians["means"]: homogeneous transform by T_h2p.
      - gaussians["quats"]: pre-multiply by rotation quat of T_h2p.
      - predicted_w2c[i]:   W_h_i @ inv(T_h2p)  →  becomes prior-frame w2c.
      - scales / opacities / rgbs / sh0: unchanged (rotation-invariant
        within the band-0 SH approximation).

    No-op fast path when T ≈ I (within 1e-4).
    """
    W_h_0 = predicted_w2c[0].float().cpu()
    W_p_0 = prior_w2c[0].float().cpu()
    if W_h_0.shape == (3, 4):
        W_h_0 = torch.cat([W_h_0, torch.tensor([[0., 0., 0., 1.]])], dim=0)
    if W_p_0.shape == (3, 4):
        W_p_0 = torch.cat([W_p_0, torch.tensor([[0., 0., 0., 1.]])], dim=0)

    T_h2p = torch.linalg.inv(W_p_0) @ W_h_0       # HYWM2 world -> prior world
    if torch.allclose(T_h2p, torch.eye(4), atol=1e-4):
        log.info("HYWM2Reconstruct: predicted frame already aligned to prior "
                 "frame (no rebasing needed)")
        return gaussians, predicted_w2c

    # Two transforms in opposite directions:
    #
    #   Gaussians: g_p = T_h2p @ g_h         (apply T_h2p to means/quats)
    #   Extrinsics: new_W_p_i = W_h_i @ inv(T_h2p)
    #
    # Derivation: keep the same physical view across the frame change:
    #     c_i = W_h_i @ g_h               (HYWM2 frame)
    #         = W_h_i @ inv(T_h2p) @ g_p   (substituting g_h)
    #         = new_W_p_i @ g_p            (prior frame, by construction)
    # So new_W_p_i = W_h_i @ inv(T_h2p). When HYWM2 doesn't refine the
    # poses (predicted is a pure rebase of prior), this reduces to
    # new_W_p_i = W_p_i exactly.
    R = T_h2p[:3, :3]
    t = T_h2p[:3, 3]
    inv_T = torch.linalg.inv(T_h2p)
    log.info(
        f"HYWM2Reconstruct: rebasing outputs to prior frame "
        f"(|R_T_h2p - I|_F={float((R - torch.eye(3)).norm()):.4f}, "
        f"|t_T_h2p|={float(t.norm()):.4f})"
    )

    # Transform gaussians by T_h2p (HYWM2 world -> prior world).
    if gaussians is not None and isinstance(gaussians, dict) and "means" in gaussians:
        means = gaussians["means"].float().cpu()
        gaussians = dict(gaussians)
        gaussians["means"] = (R @ means.T).T + t
        if "quats" in gaussians:
            quats = gaussians["quats"].float().cpu()
            q_R = _matrix_to_quaternion_wxyz(R)
            q_R_b = q_R.unsqueeze(0).expand(quats.shape[0], 4)
            gaussians["quats"] = _quat_multiply_wxyz(q_R_b, quats)

    # Post-multiply each predicted w2c by inv(T_h2p) so projected views
    # match the now-rebased gaussians.
    new_predicted = torch.matmul(predicted_w2c.float().cpu(), inv_T)

    return gaussians, new_predicted


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
                io.Image.Output(
                    display_name="depth_raw",
                    tooltip=(
                        "Per-view raw float depth as IMAGE [N,H,W,3] (the depth value "
                        "broadcast across 3 channels — mirrors MoGe2's depth_raw "
                        "convention, so it composes with depth-viz nodes and pipes "
                        "directly into WorldStereoAlignMemoryBankDepths / "
                        "WorldStereoAlignDepthAndGrowPCD. Returns 1×1 zeros if the "
                        "depth head is disabled."
                    ),
                ),
                io.Image.Output(
                    display_name="points_raw",
                    tooltip=(
                        "Per-view raw 3D point map IMAGE [N,H,W,3] with channels "
                        "(X, Y, Z) in CAMERA SPACE (meters). Computed from depth + "
                        "predicted_intrinsics via the pinhole unprojection "
                        "(X = (u-cx)/fx · Z, Y = (v-cy)/fy · Z). Drop-in for "
                        "PanoramaDepthMerge.face_points and MoGe2-style "
                        "points_raw consumers. Returns 1×1 zeros if the depth "
                        "head is disabled."
                    ),
                ),
                io.Image.Output(
                    display_name="depth_conf",
                    tooltip=(
                        "Per-pixel depth confidence IMAGE [N,H,W,3] from the "
                        "depth head (broadcast across 3 channels). Higher = "
                        "model is more certain about that pixel's depth. "
                        "Typically dips at corners, edges, occlusion "
                        "boundaries — useful signal for plane-snapping "
                        "post-processing (chain into a confidence-guided "
                        "edge sharpener to mitigate chamfered corners). "
                        "Returns 1×1 zeros if the depth head is disabled."
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
    ):
        if not isinstance(model, dict) or "model_dir" not in model:
            raise ValueError(
                f"HYWM2Reconstruct: invalid model handle (expected dict from "
                f"LoadHYWM2Model, got {type(model).__name__})"
            )
        if images is None or images.numel() == 0:
            raise ValueError("HYWM2Reconstruct: images batch is empty")

        # 6-stage progress bar for the ComfyUI per-node chrome.
        # Stages: 1=preprocess inputs, 2=load pipeline, 3=forward pass,
        # 4=decode predictions, 5=rebase to prior frame, 6=points_raw + emit.
        # Stage 3 is the long one — bar will appear "stuck" there during
        # the actual model forward, which is the expected/honest signal.
        try:
            from comfy.utils import ProgressBar
            _pbar = ProgressBar(6)
        except Exception:
            class _NoPbar:
                def update(self, *_a, **_kw): pass
            _pbar = _NoPbar()

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
            log.info("stage 1/6: preprocess inputs (dump images + prior cam JSON)")
            img_paths = cls._dump_image_batch(images, tmp_dir)
            log.info("HYWM2Reconstruct: wrote %d frame(s) to %s", len(img_paths), tmp_dir)
            _pbar.update(1)

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

            log.info("stage 2/6: load pipeline (cached if config unchanged)")
            pipeline = cls._get_pipeline(model)
            _pbar.update(1)

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

            # Resolution policy (no VRAM auto-shrink):
            #   1. `target_size_override > 0`  -> use that exact value, snapped
            #      to a multiple of 14 (ViT patch size).
            #   2. Default                     -> compute_adaptive_target_size:
            #      min(image_longest_edge, model.target_size) snapped to /14.
            #      Keeps the image at its native resolution when smaller than
            #      the model's training resolution (default 952); caps at the
            #      training resolution for larger inputs (positional encoding
            #      doesn't extrapolate well past that).
            # If the resulting resolution OOMs, `_log_oom_diagnostic` prints
            # a breakdown; user picks a smaller `target_size_override`.
            from .hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
                compute_adaptive_target_size,
            )
            free_gb = 0.0
            # Free-VRAM read kept for logging only.
            if torch.cuda.is_available():
                try:
                    free_gb = mm.get_free_memory(pipeline.device) / (1024 ** 3)
                except Exception:
                    free_gb = torch.cuda.mem_get_info(pipeline.device)[0] / (1024 ** 3)

            # Predictable, non-VRAM-based effective resolution. No auto-
            # shrink: the user gets exactly the size they asked for (or
            # the input-aware adaptive fallback when no override). OOM is
            # surfaced via `_log_oom_diagnostic` if it happens — the user
            # picks a smaller `target_size_override` next time.
            if int(target_size_override) > 0:
                snapped = (int(target_size_override) // _PATCH) * _PATCH
                effective = max(_MIN_EFFECTIVE, snapped)
                log.info(
                    "HYWM2Reconstruct: target_size_override=%d -> effective=%d "
                    "(views=%d, free=%.1fGB)",
                    target_size_override, effective, len(img_paths), free_gb,
                )
            else:
                target_size = int(model.get("target_size", 952))
                effective = compute_adaptive_target_size(img_paths, target_size)
                log.info(
                    "HYWM2Reconstruct: views=%d requested=%d -> effective=%d "
                    "(input-aware; free=%.1fGB)",
                    len(img_paths), target_size, effective, free_gb,
                )


            prior_cam = prior_camera_json.strip() or None
            if prior_cam and not Path(prior_cam).is_file():
                raise FileNotFoundError(f"prior_camera_json not found: {prior_cam}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                cls._log_vram("pre-forward", pipeline.device, mm)
                torch.cuda.reset_peak_memory_stats(pipeline.device)

            log.info("stage 3/6: forward pass @ effective=%d, views=%d",
                     effective, len(img_paths))
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
            _pbar.update(1)

            if torch.cuda.is_available():
                cls._log_vram("post-forward", pipeline.device, mm)

        log.info("HYWM2Reconstruct: forward pass done in %.2fs", infer_time)

        # ----- inline decode (formerly Decode* nodes) -----
        log.info("stage 4/6: decode predictions (depth/normals/points/gaussians)")
        from .decode_export import (
            decode_depth_image, decode_normals_image,
            decode_points, decode_gaussians,
            select_views, _empty_image,
        )

        depth_img = decode_depth_image(
            predictions, view_index=view_index,
            apply_mask=apply_mask, colormap=depth_colormap,
        )

        # Raw float depth for downstream alignment / metric consumers.
        # Mirrors decode_depth_image's view-selection + masking but
        # skips the per-frame normalize + colormap step.
        if "depth" in predictions:
            d = predictions["depth"].detach().float().cpu()             # [B,S,H,W,1] or [B,S,H,W]
            if d.dim() == 5:
                d = d.squeeze(-1)
            d = select_views(d, view_index)                              # [S,H,W]
            if apply_mask and "depth_mask" in predictions:
                dm = predictions["depth_mask"].detach().float().cpu()
                dm = select_views(dm, view_index)
                d = torch.where(dm >= 0.5, d, torch.zeros_like(d))

            # Per-frame depth distribution stats. Helps diagnose alignment
            # PASS1 SKIPs (often caused by HYWM2 producing few valid pixels
            # on out-of-coverage views).
            S_, H_, W_ = d.shape
            total_px = int(H_ * W_)
            valid_pcts: list = []
            medians: list = []
            mins: list = []
            maxs: list = []
            per_frame_lines: list = []
            for s in range(S_):
                ds = d[s]
                valid = ds > 0
                n_valid = int(valid.sum().item())
                if n_valid > 0:
                    dv = ds[valid]
                    dmin = float(dv.min().item())
                    dmed = float(dv.median().item())
                    dmax = float(dv.max().item())
                else:
                    dmin = dmed = dmax = float("nan")
                pct = 100.0 * n_valid / max(total_px, 1)
                valid_pcts.append(pct)
                medians.append(dmed)
                mins.append(dmin)
                maxs.append(dmax)
                per_frame_lines.append(
                    f"  frame {s:>3}: min={dmin:.3f} median={dmed:.3f} "
                    f"max={dmax:.3f} valid={pct:.1f}% ({n_valid}/{total_px})"
                )

            if S_ <= 20:
                log.info("HYWM2Reconstruct: depth stats per frame:")
                for line in per_frame_lines:
                    log.info(line)
            else:
                vp = np.asarray(valid_pcts, dtype=np.float64)
                mds = np.asarray(medians, dtype=np.float64)
                mds_valid = mds[~np.isnan(mds)]
                vp_min_idx = int(np.argmin(vp))
                p10, p50, p90 = np.percentile(vp, [10, 50, 90])
                if mds_valid.size > 0:
                    med_summary = float(np.median(mds_valid))
                else:
                    med_summary = float("nan")
                log.info(
                    "HYWM2Reconstruct: depth distribution across %d frames: "
                    "p50_of_per_frame_median=%.3f | valid_pct: min=%.1f%% "
                    "(frame %d) p10=%.1f%% median=%.1f%% p90=%.1f%%",
                    S_, med_summary, float(vp.min()), vp_min_idx,
                    float(p10), float(p50), float(p90),
                )

            depth_raw_img = d.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()  # [S,H,W,3]
        else:
            depth_raw_img = _empty_image()
            d = None  # signal points_raw to short-circuit below

        # Per-pixel depth confidence from the depth head. Useful for
        # detecting where the model is uncertain — chamfered corners,
        # occlusion edges — so downstream nodes can confidence-guide a
        # plane-snap or edge-sharpen post-process.
        if "depth_conf" in predictions:
            dc = predictions["depth_conf"].detach().float().cpu()           # [B,S,H,W,1] or [B,S,H,W]
            if dc.dim() == 5:
                dc = dc.squeeze(-1)
            dc = select_views(dc, view_index)                                # [S,H,W]
            depth_conf_img = dc.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()  # [S,H,W,3]
            try:
                log.info(
                    "HYWM2Reconstruct: depth_conf [%d,%d,%d], range=[%.3f, %.3f] "
                    "median=%.3f",
                    dc.shape[0], dc.shape[1], dc.shape[2],
                    float(dc.min()), float(dc.max()), float(dc.median()),
                )
            except Exception:
                pass
        else:
            depth_conf_img = _empty_image()
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
        _pbar.update(1)

        # ----- predicted cameras (CameraPack convention: w2c [N,4,4], K [N,3,3]) -----
        # When `predict_camera=True`, use the camera head's outputs (HYWM2
        # may refine the priors slightly). When it's disabled, fall back to
        # the user's `prior_extrinsics` / `prior_intrinsics` so downstream
        # consumers (HYWM2GaussianTrain.gaussians_extrinsics,
        # PreviewGaussianCamera, points_raw unprojection) get valid camera
        # data instead of a 1×4×4 / 1×3×3 identity placeholder. The
        # frame-rebase block below is still gated on `c2w is not None`
        # (only meaningful when HYWM2 produced its own poses), so the
        # fallback skips rebasing — outputs are already in prior frame.
        c2w = predictions.get("camera_poses")          # [B,S,4,4] (OpenCV c2w)
        intr = predictions.get("camera_intrs")         # [B,S,3,3]
        if c2w is not None:
            c2w_t = c2w.detach().float().cpu()
            if c2w_t.dim() == 4 and c2w_t.shape[0] == 1:
                c2w_t = c2w_t[0]                       # [S,4,4]
            extrinsics_w2c = torch.linalg.inv(c2w_t)
        elif prior_extrinsics is not None:
            extrinsics_w2c = prior_extrinsics.detach().float().cpu()
            if extrinsics_w2c.dim() == 4 and extrinsics_w2c.shape[0] == 1:
                extrinsics_w2c = extrinsics_w2c[0]
            log.info(
                "HYWM2Reconstruct: predicted_extrinsics fallback — using "
                "prior_extrinsics [%s] (predict_camera=False)",
                "×".join(str(s) for s in extrinsics_w2c.shape),
            )
        else:
            extrinsics_w2c = torch.eye(4).unsqueeze(0)

        if intr is not None:
            intr_t = intr.detach().float().cpu()
            if intr_t.dim() == 4 and intr_t.shape[0] == 1:
                intr_t = intr_t[0]                     # [S,3,3]
        elif prior_intrinsics is not None:
            # Fallback: user-supplied K. Sniff normalized vs pixel-K and
            # rescale to the effective inference resolution (depth's W if
            # available, else `effective`) so downstream consumers expect-
            # ing inference-res intrinsics get a matching K.
            pi = prior_intrinsics.detach().float().cpu()
            if pi.dim() == 4 and pi.shape[0] == 1:
                pi = pi[0]
            if pi.dim() == 2:
                pi = pi.unsqueeze(0)
            W_inf = int(d.shape[-1]) if d is not None else int(effective)
            if float(pi[0, 0, 0]) < 2.0:
                pi[..., :2, :] = pi[..., :2, :] * W_inf
                log.info(
                    "HYWM2Reconstruct: predicted_intrinsics fallback — "
                    "prior_intrinsics normalized; rescaled to pixel-K at "
                    "inference res %d (predict_camera=False)", W_inf,
                )
            else:
                W_input = int(images.shape[2]) if images is not None else W_inf
                if W_input > 0 and W_input != W_inf:
                    pi[..., :2, :] = pi[..., :2, :] * (W_inf / W_input)
                    log.info(
                        "HYWM2Reconstruct: predicted_intrinsics fallback — "
                        "prior_intrinsics at %d → rescaled to inference res %d "
                        "(×%.4f, predict_camera=False)",
                        W_input, W_inf, W_inf / W_input,
                    )
                else:
                    log.info(
                        "HYWM2Reconstruct: predicted_intrinsics fallback — "
                        "using prior_intrinsics as-is at res %d "
                        "(predict_camera=False)", W_inf,
                    )
            intr_t = pi
        else:
            intr_t = torch.eye(3).unsqueeze(0)

        # ----- points_raw: per-view 3D point map in camera space -----
        # Compute via pinhole unprojection: (X, Y, Z) = ((u-cx)/fx · Z,
        # (v-cy)/fy · Z, Z). By construction ||points|| equals the
        # ray distance that PanoramaDepthMerge.face_points consumes via
        # np.linalg.norm. Predicted intrinsics are already in pixel units
        # of the effective inference resolution — they match d's H,W
        # exactly, so no rescale needed (unlike SharpPredictMetricDepth
        # which has to bridge face-image-K to internal 1536² grid).
        #
        # `intr_t` is now always populated (predicted camera head → first;
        # prior_intrinsics → fallback; identity placeholder → last resort).
        # If we hit the identity placeholder (no head + no prior), the
        # unprojection produces garbage; skip via the dim/shape check below.
        intr_valid = intr_t is not None and intr_t.shape[0] >= 1 and not torch.equal(
            intr_t[0], torch.eye(3)
        )
        if d is not None and intr_valid:
            S_p, H_p, W_p = d.shape
            uu = torch.arange(W_p, dtype=torch.float32)
            vv = torch.arange(H_p, dtype=torch.float32)
            uu_g, vv_g = torch.meshgrid(uu, vv, indexing="xy")  # (H_p, W_p)
            n_intr = int(intr_t.shape[0])
            points_per_view = []
            for s in range(S_p):
                K_s = intr_t[min(s, n_intr - 1)]
                fx = float(K_s[0, 0])
                fy = float(K_s[1, 1])
                cx = float(K_s[0, 2])
                cy = float(K_s[1, 2])
                x_cam = (uu_g - cx) / fx
                y_cam = (vv_g - cy) / fy
                Z_s = d[s]
                points_per_view.append(
                    torch.stack([x_cam * Z_s, y_cam * Z_s, Z_s], dim=-1)
                )
            points_raw_img = torch.stack(points_per_view, dim=0).contiguous()  # [S,H,W,3]
            try:
                pn = points_raw_img.norm(dim=-1)
                log.info(
                    "HYWM2Reconstruct: points_raw [%d,%d,%d,3], ||v||=%.2f-%.2fm",
                    S_p, H_p, W_p, float(pn.min()), float(pn.max()),
                )
            except Exception:
                pass
        else:
            points_raw_img = _empty_image()

        # Rebase outputs back to the user's `prior_extrinsics` frame.
        # HYWM2's pose head anchors view 0 to identity in its internal
        # world; without this rebase, gaussians + predicted_extrinsics
        # come out in HYWM2's rebased frame, which doesn't match the
        # frame the caller supplied via `prior_extrinsics`. Result was
        # "each rendered view shows the wrong image" downstream.
        # See `_rebase_outputs_to_prior_frame` for the math.
        log.info("stage 5/6: rebase outputs to prior frame (+ predicted cameras + points_raw)")
        _pbar.update(1)
        if (prior_extrinsics is not None
                and c2w is not None
                and isinstance(extrinsics_w2c, torch.Tensor)
                and extrinsics_w2c.dim() == 3
                and extrinsics_w2c.shape[0] >= 1):
            try:
                prior_t = prior_extrinsics
                if prior_t.dim() == 4 and prior_t.shape[0] == 1:
                    prior_t = prior_t[0]
                if prior_t.shape[0] == extrinsics_w2c.shape[0]:
                    gaussians, extrinsics_w2c = _rebase_outputs_to_prior_frame(
                        gaussians, extrinsics_w2c, prior_t,
                    )
                else:
                    log.info(
                        "HYWM2Reconstruct: skipping prior-frame rebase — "
                        "N mismatch (prior=%d, predicted=%d).",
                        int(prior_t.shape[0]), int(extrinsics_w2c.shape[0]),
                    )
            except Exception as e:
                log.info(
                    "HYWM2Reconstruct: prior-frame rebase failed (%s); "
                    "returning outputs in HYWM2's internal frame.", e,
                )

        # Summary so the user can see at a glance which outputs are populated.
        empty = []
        if "depth" not in predictions:    empty.append("images(depth)")
        if "normals" not in predictions:  empty.append("normals")
        if "pts3d" not in predictions:    empty.append("points")
        if "splats" not in predictions:   empty.append("gaussians")
        if c2w is None or intr is None:   empty.append("cameras")
        if empty:
            log.info("HYWM2Reconstruct: empty outputs (head disabled): %s", ", ".join(empty))

        log.info("stage 6/6: emit outputs")
        _pbar.update(1)

        # Option-B (hybrid) ComfyUI-native cleanup: drop our class-level
        # reference to the ModelPatcher so ComfyUI's LRU (current_loaded_models
        # / free_memory) can fully evict it when another node needs VRAM.
        # The `_pipeline` cache stays so warm re-runs are still fast — re-
        # wrapping `pipeline.model` in a fresh patcher on the next call is
        # cheap when the weights are already where they need to be.
        cls._patcher = None

        return io.NodeOutput(depth_img, normals_img, points, gaussians, extrinsics_w2c, intr_t, depth_raw_img, points_raw_img, depth_conf_img)

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
