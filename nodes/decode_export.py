"""Decode + export nodes for HYWM2 predictions.

These consume the wrapper dict produced by HYWM2Reconstruct (with keys
``predictions``, ``imgs``, ...) and emit:
  - IMAGE batches (depth, normals, mask) for inline preview
  - HYWM2_POINTS / HYWM2_GAUSSIANS custom dicts for chaining
  - PLY filepaths for export to SuperSplat / Blender / Unity
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import folder_paths
from comfy_api.latest import io

log = logging.getLogger("hywm2")

_C0 = 0.28209479177387814  # SH degree-0 normalization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(predictions: Any) -> dict:
    """Accept either the full HYWM2Reconstruct wrapper or a raw predictions dict."""
    if not isinstance(predictions, dict):
        raise ValueError(
            f"Expected dict from HYWM2Reconstruct, got {type(predictions).__name__}"
        )
    return predictions["predictions"] if "predictions" in predictions else predictions


def _imgs_tensor(predictions: Any) -> torch.Tensor | None:
    if isinstance(predictions, dict) and "imgs" in predictions:
        return predictions["imgs"]
    return None


def _select_views(t: torch.Tensor, view_index: int) -> torch.Tensor:
    """Slice [B,S,...] -> [S,...] (all) or [1,...] (single)."""
    if t.dim() < 2:
        return t
    if view_index < 0:
        return t[0]
    return t[0, view_index : view_index + 1]


def _viridis_colormap(values: torch.Tensor) -> torch.Tensor:
    """Map a [...,H,W] float tensor in [0,1] to RGB [...,H,W,3] using viridis."""
    # 16-stop viridis lookup, linear-interpolated.
    stops = torch.tensor([
        [0.267, 0.005, 0.329], [0.282, 0.094, 0.418], [0.279, 0.175, 0.483],
        [0.262, 0.243, 0.521], [0.232, 0.300, 0.542], [0.198, 0.353, 0.553],
        [0.166, 0.404, 0.557], [0.137, 0.453, 0.558], [0.114, 0.502, 0.555],
        [0.119, 0.554, 0.546], [0.165, 0.602, 0.527], [0.246, 0.648, 0.493],
        [0.358, 0.692, 0.444], [0.491, 0.731, 0.378], [0.642, 0.762, 0.295],
        [0.795, 0.789, 0.196], [0.993, 0.906, 0.144],
    ], dtype=values.dtype, device=values.device)
    n = stops.shape[0] - 1
    x = values.clamp(0, 1) * n
    lo = x.floor().long().clamp(max=n - 1)
    t_frac = (x - lo.float()).unsqueeze(-1)
    a = stops[lo]
    b = stops[lo + 1]
    return a + (b - a) * t_frac


def _to_image_batch(t: torch.Tensor) -> torch.Tensor:
    """Force [B,H,W,C] float in [0,1] for ComfyUI IMAGE."""
    t = t.detach().float().cpu()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
    elif t.dim() == 3:
        if t.shape[-1] in (1, 3):
            if t.shape[-1] == 1:
                t = t.repeat(1, 1, 3).unsqueeze(0)
            else:
                t = t.unsqueeze(0)
        else:
            t = t.unsqueeze(-1).repeat(1, 1, 1, 3)
    elif t.dim() == 4 and t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    return t.clamp(0, 1).contiguous()


def _resolve_output_path(filename: str, output_dir: str, ext: str) -> Path:
    if output_dir and output_dir.strip():
        save_dir = Path(output_dir.strip())
    else:
        save_dir = Path(folder_paths.get_output_directory()) / "hywm2"
    save_dir.mkdir(parents=True, exist_ok=True)
    name = filename.strip() or "output"
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    name = name.strip(". ")
    if not name.lower().endswith(ext):
        name = name + ext
    return save_dir / name


# ---------------------------------------------------------------------------
# DecodeDepth
# ---------------------------------------------------------------------------

class HYWM2DecodeDepth(io.ComfyNode):
    """Predictions → IMAGE (per-view depth, normalized + optionally colormapped)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2DecodeDepth",
            display_name="HYWM2 Decode Depth",
            category="HYWM2",
            description="Convert depth predictions into a previewable IMAGE batch (one frame per view).",
            inputs=[
                io.Custom("HYWM2_PREDICTIONS").Input("predictions"),
                io.Combo.Input("colormap", options=["grayscale", "viridis"], default="viridis"),
                io.Boolean.Input("apply_mask", default=True,
                                 tooltip="Mask out invalid depth pixels (depth_mask < 0.5)."),
                io.Int.Input("view_index", default=-1, min=-1, max=64,
                             tooltip="Single view (>=0) or all views (-1)."),
            ],
            outputs=[io.Image.Output(display_name="depth")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, predictions, colormap, apply_mask, view_index):
        preds = _unwrap(predictions)
        depth = preds["depth"].detach().float().cpu()  # [B,S,H,W,1]
        if depth.dim() == 5:
            depth = depth.squeeze(-1)
        depth = _select_views(depth, view_index)  # [S,H,W]

        if apply_mask and "depth_mask" in preds:
            mask = preds["depth_mask"].detach().float().cpu()
            mask = _select_views(mask, view_index)
            valid = mask >= 0.5
        else:
            valid = torch.ones_like(depth, dtype=torch.bool)

        # Per-frame normalization over valid pixels
        norm = torch.zeros_like(depth)
        for i in range(depth.shape[0]):
            m = valid[i]
            if m.any():
                d = depth[i][m]
                lo, hi = d.min(), d.max()
                if hi > lo:
                    norm[i] = ((depth[i] - lo) / (hi - lo)).clamp(0, 1)
                norm[i][~m] = 0.0

        if colormap == "viridis":
            rgb = _viridis_colormap(norm)
        else:
            rgb = norm.unsqueeze(-1).repeat(1, 1, 1, 3)

        return io.NodeOutput(_to_image_batch(rgb))


# ---------------------------------------------------------------------------
# DecodeNormals
# ---------------------------------------------------------------------------

class HYWM2DecodeNormals(io.ComfyNode):
    """Predictions → IMAGE (per-view surface normals as RGB = 0.5*(n+1))."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2DecodeNormals",
            display_name="HYWM2 Decode Normals",
            category="HYWM2",
            description="Convert surface normal predictions into a previewable IMAGE batch.",
            inputs=[
                io.Custom("HYWM2_PREDICTIONS").Input("predictions"),
                io.Boolean.Input("apply_mask", default=True),
                io.Int.Input("view_index", default=-1, min=-1, max=64),
            ],
            outputs=[io.Image.Output(display_name="normals")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, predictions, apply_mask, view_index):
        preds = _unwrap(predictions)
        normals = preds["normals"].detach().float().cpu()  # [B,S,H,W,3]
        normals = _select_views(normals, view_index)
        rgb = (0.5 * (normals + 1.0)).clamp(0, 1)

        if apply_mask and "depth_mask" in preds:
            mask = _select_views(preds["depth_mask"].detach().float().cpu(), view_index)
            rgb = rgb * (mask >= 0.5).float().unsqueeze(-1)

        return io.NodeOutput(_to_image_batch(rgb))


# ---------------------------------------------------------------------------
# DecodePoints
# ---------------------------------------------------------------------------

class HYWM2DecodePoints(io.ComfyNode):
    """Predictions → HYWM2_POINTS (means + per-vertex RGB from input imgs)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2DecodePoints",
            display_name="HYWM2 Decode Points",
            category="HYWM2",
            description="Extract a colored point cloud from pts3d + imgs, optionally filtered by confidence/mask.",
            inputs=[
                io.Custom("HYWM2_PREDICTIONS").Input("predictions"),
                io.Boolean.Input("apply_mask", default=True),
                io.Float.Input("conf_percentile", default=0.0, min=0.0, max=99.0, step=1.0,
                               tooltip="Drop bottom N%% of points by pts3d_conf (0 = keep all)."),
                io.Int.Input("max_points", default=2_000_000, min=10_000, max=20_000_000),
            ],
            outputs=[io.Custom("HYWM2_POINTS").Output(display_name="points")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, predictions, apply_mask, conf_percentile, max_points):
        preds = _unwrap(predictions)
        pts = preds["pts3d"].detach().float().cpu()      # [B,S,H,W,3]
        if pts.dim() == 5:
            pts = pts[0]                                  # [S,H,W,3]
        S, H, W, _ = pts.shape

        imgs = _imgs_tensor(predictions)
        if imgs is not None:
            colors = imgs.detach().float().cpu()
            # imgs shape from upstream: [B,S,3,H,W]
            if colors.dim() == 5:
                colors = colors[0]
            if colors.shape[1] == 3:
                colors = colors.permute(0, 2, 3, 1).contiguous()  # [S,H,W,3]
        else:
            colors = torch.full_like(pts, 0.5)

        means = pts.reshape(-1, 3)
        cols = colors.reshape(-1, 3).clamp(0, 1)

        keep = torch.ones(means.shape[0], dtype=torch.bool)
        if apply_mask and "depth_mask" in preds:
            m = preds["depth_mask"].detach().float().cpu()
            if m.dim() == 4:
                m = m[0]
            keep &= (m.reshape(-1) >= 0.5)
        if conf_percentile > 0 and "pts3d_conf" in preds:
            c = preds["pts3d_conf"].detach().float().cpu()
            if c.dim() == 4:
                c = c[0]
            cv = c.reshape(-1)
            thresh = torch.quantile(cv, conf_percentile / 100.0)
            keep &= (cv >= thresh)

        means = means[keep]
        cols = cols[keep]

        if means.shape[0] > max_points and means.shape[0] > 0:
            idx = torch.randperm(means.shape[0])[:max_points]
            means = means[idx]
            cols = cols[idx]

        log.info("HYWM2DecodePoints: %d points", means.shape[0])
        return io.NodeOutput({"means": means, "colors": cols})


# ---------------------------------------------------------------------------
# DecodeGaussians
# ---------------------------------------------------------------------------

class HYWM2DecodeGaussians(io.ComfyNode):
    """Predictions → HYWM2_GAUSSIANS (means/quats/scales/opacities/sh + RGB)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2DecodeGaussians",
            display_name="HYWM2 Decode Gaussians",
            category="HYWM2",
            description="Extract 3DGS attributes (with SH→RGB conversion) for export.",
            inputs=[
                io.Custom("HYWM2_PREDICTIONS").Input("predictions"),
                io.Float.Input("weight_threshold", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Drop Gaussians below this per-point weight (0 = keep all)."),
            ],
            outputs=[io.Custom("HYWM2_GAUSSIANS").Output(display_name="gaussians")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, predictions, weight_threshold):
        preds = _unwrap(predictions)
        if "splats" not in preds:
            raise RuntimeError("HYWM2DecodeGaussians: predictions has no 'splats' key (gs head disabled?)")
        s = preds["splats"]

        means = s["means"].detach().float().cpu()
        quats = s["quats"].detach().float().cpu()
        scales = s["scales"].detach().float().cpu()
        opacities = s["opacities"].detach().float().cpu()
        sh = s["sh"].detach().float().cpu()        # [B,N,1,3]
        weights = s.get("weights", None)
        if weights is not None:
            weights = weights.detach().float().cpu()

        if means.dim() == 3:
            means, quats, scales, opacities = means[0], quats[0], scales[0], opacities[0]
            sh = sh[0]
            if weights is not None:
                weights = weights[0]

        sh_dc = sh[..., 0, :]                       # [N,3]
        rgbs = (sh_dc * _C0 + 0.5).clamp(0, 1)

        if weight_threshold > 0 and weights is not None:
            keep = weights >= weight_threshold
            means, quats, scales, opacities, rgbs = (
                means[keep], quats[keep], scales[keep], opacities[keep], rgbs[keep]
            )
            log.info("HYWM2DecodeGaussians: kept %d/%d gaussians (weight>=%.2f)",
                     keep.sum().item(), keep.numel(), weight_threshold)

        return io.NodeOutput({
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "rgbs": rgbs,
        })


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class HYWM2ExportPointsPLY(io.ComfyNode):
    """HYWM2_POINTS → PLY filepath."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2ExportPointsPLY",
            display_name="HYWM2 Export Points PLY",
            category="HYWM2",
            description="Write a point cloud PLY (x,y,z + r,g,b) to disk.",
            inputs=[
                io.Custom("HYWM2_POINTS").Input("points"),
                io.String.Input("filename", default="hywm2_points", multiline=False),
                io.String.Input("output_dir", default="", multiline=False),
            ],
            outputs=[io.String.Output(display_name="filepath")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, points, filename, output_dir):
        if not isinstance(points, dict) or "means" not in points:
            raise ValueError("HYWM2ExportPointsPLY: expected HYWM2_POINTS dict with 'means'")

        from .hyworld2.worldrecon.hyworldmirror.utils.save_utils import save_points_ply

        out = _resolve_output_path(filename, output_dir, ".ply")
        means = points["means"].detach().cpu().numpy().astype(np.float32)
        cols = points.get("colors")
        if cols is None:
            cols_np = np.full_like(means, 0.5, dtype=np.float32)
        else:
            cols_np = cols.detach().cpu().numpy().astype(np.float32)
        cols_np = (cols_np.clip(0, 1) * 255.0 + 0.5).astype(np.uint8)

        save_points_ply(out, means, cols_np)
        log.info("HYWM2ExportPointsPLY: wrote %s (%d pts, %.1f MB)",
                 out, means.shape[0], out.stat().st_size / 1e6)
        return io.NodeOutput(str(out))


class HYWM2ExportGaussiansPLY(io.ComfyNode):
    """HYWM2_GAUSSIANS → 3DGS PLY filepath (SuperSplat / Blender compatible)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2ExportGaussiansPLY",
            display_name="HYWM2 Export Gaussians PLY",
            category="HYWM2",
            description="Write a 3D Gaussian Splatting PLY (means/scales/rotations/RGB/opacity).",
            inputs=[
                io.Custom("HYWM2_GAUSSIANS").Input("gaussians"),
                io.String.Input("filename", default="hywm2_gaussians", multiline=False),
                io.String.Input("output_dir", default="", multiline=False),
            ],
            outputs=[io.String.Output(display_name="filepath")],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, gaussians, filename, output_dir):
        if not isinstance(gaussians, dict) or "means" not in gaussians:
            raise ValueError("HYWM2ExportGaussiansPLY: expected HYWM2_GAUSSIANS dict")

        from .hyworld2.worldrecon.hyworldmirror.utils.save_utils import save_gs_ply

        out = _resolve_output_path(filename, output_dir, ".ply")
        save_gs_ply(
            out,
            means=gaussians["means"].float(),
            scales=gaussians["scales"].float(),
            rotations=gaussians["quats"].float(),
            rgbs=gaussians["rgbs"].float(),
            opacities=gaussians["opacities"].float(),
        )
        log.info("HYWM2ExportGaussiansPLY: wrote %s (%d gaussians, %.1f MB)",
                 out, gaussians["means"].shape[0], out.stat().st_size / 1e6)
        return io.NodeOutput(str(out))


# ---------------------------------------------------------------------------
# Preview (3D viewer)
# ---------------------------------------------------------------------------

class HYWM2PreviewPointCloud(io.ComfyNode):
    """Browser preview of a PLY file via the bundled pointcloud_vtk viewer."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2PreviewPointCloud",
            display_name="HYWM2 Preview Point Cloud",
            category="HYWM2",
            description="Preview a PLY file in 3D using the VTK.js viewer.",
            is_output_node=True,
            inputs=[io.String.Input("file_path", default="")],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return f"{kwargs.get('file_path', '')}"

    @classmethod
    @torch.no_grad()
    def execute(cls, file_path: str = ""):
        return {"ui": {"file_path": [file_path or ""]}}
