"""Export + preview nodes for HYWM2 outputs.

The Decode* nodes (Depth / Normals / Points / Gaussians) used to live here
but are now inlined into HYWM2Reconstruct's outputs. This module keeps:

  - shared decode helpers (used by reconstruct.py inline)
  - HYWM2ExportPointsPLY  (HYWM2_POINTS    → STRING filepath)
  - HYWM2ExportGaussiansPLY (HYWM2_GAUSSIANS → STRING filepath)
  - HYWM2PreviewPointCloud (STRING filepath → VTK.js viewer)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import folder_paths
from comfy_api.latest import io

log = logging.getLogger("hywm2")

C0 = 0.28209479177387814  # SH degree-0 normalization


# ---------------------------------------------------------------------------
# Decode helpers — exported for reconstruct.py
# ---------------------------------------------------------------------------

def unwrap_predictions(predictions: Any) -> dict:
    """Accept either the full HYWM2Reconstruct wrapper or a raw predictions dict."""
    if not isinstance(predictions, dict):
        raise ValueError(
            f"Expected dict from HYWM2Reconstruct, got {type(predictions).__name__}"
        )
    return predictions["predictions"] if "predictions" in predictions else predictions


def imgs_tensor(predictions: Any) -> torch.Tensor | None:
    if isinstance(predictions, dict) and "imgs" in predictions:
        return predictions["imgs"]
    return None


def select_views(t: torch.Tensor, view_index: int) -> torch.Tensor:
    """Slice [B,S,...] -> [S,...] (all) or [1,...] (single)."""
    if t.dim() < 2:
        return t
    if view_index < 0:
        return t[0]
    return t[0, view_index : view_index + 1]


def viridis_colormap(values: torch.Tensor) -> torch.Tensor:
    """Map a [...,H,W] float tensor in [0,1] to RGB [...,H,W,3] using viridis."""
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


def to_image_batch(t: torch.Tensor) -> torch.Tensor:
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
# Decode primitives — used inline by reconstruct.py
# ---------------------------------------------------------------------------

def _empty_image() -> torch.Tensor:
    """1×1 black IMAGE used as the typed-empty sentinel for disabled head outputs."""
    return torch.zeros((1, 1, 1, 3))


def decode_depth_image(preds: dict, *, view_index: int = -1,
                       apply_mask: bool = True, colormap: str = "viridis") -> torch.Tensor:
    """Per-view depth → IMAGE batch (per-frame normalized + colormapped + masked).

    Returns a 1×1 black IMAGE if the depth head was disabled.
    """
    if "depth" not in preds:
        log.info("HYWM2 decode_depth_image: depth head disabled → empty IMAGE")
        return _empty_image()
    depth = preds["depth"].detach().float().cpu()                # [B,S,H,W,1]
    if depth.dim() == 5:
        depth = depth.squeeze(-1)
    depth = select_views(depth, view_index)                      # [S,H,W]

    if apply_mask and "depth_mask" in preds:
        mask = preds["depth_mask"].detach().float().cpu()
        mask = select_views(mask, view_index)
        valid = mask >= 0.5
    else:
        valid = torch.ones_like(depth, dtype=torch.bool)

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
        rgb = viridis_colormap(norm)
    else:
        rgb = norm.unsqueeze(-1).repeat(1, 1, 1, 3)
    return to_image_batch(rgb)


def decode_normals_image(preds: dict, *, view_index: int = -1,
                         apply_mask: bool = True) -> torch.Tensor:
    """Per-view normals → IMAGE batch (RGB = 0.5·(n+1), masked).

    Returns a 1×1 black IMAGE if the normals head was disabled.
    """
    if "normals" not in preds:
        log.info("HYWM2 decode_normals_image: normals head disabled → empty IMAGE")
        return _empty_image()
    normals = preds["normals"].detach().float().cpu()            # [B,S,H,W,3]
    normals = select_views(normals, view_index)
    rgb = (0.5 * (normals + 1.0)).clamp(0, 1)
    if apply_mask and "depth_mask" in preds:
        mask = select_views(preds["depth_mask"].detach().float().cpu(), view_index)
        rgb = rgb * (mask >= 0.5).float().unsqueeze(-1)
    return to_image_batch(rgb)


def decode_points(preds: dict, imgs: torch.Tensor | None, *,
                  apply_mask: bool = True, conf_percentile: float = 0.0,
                  max_points: int = 2_000_000) -> dict:
    """Predictions → HYWM2_POINTS (means + RGB-from-imgs, optionally filtered).

    Returns an empty HYWM2_POINTS dict if the points head was disabled.
    """
    if "pts3d" not in preds:
        # 1-row sentinel rather than (0,3): comfy-env's worker shm serializer
        # rejects 0-element tensors via `rebuild_storage_empty`.
        log.info("HYWM2 decode_points: points head disabled → 1-row sentinel HYWM2_POINTS")
        return {"means": torch.zeros((1, 3)), "colors": torch.zeros((1, 3))}
    pts = preds["pts3d"].detach().float().cpu()                  # [B,S,H,W,3]
    if pts.dim() == 5:
        pts = pts[0]                                              # [S,H,W,3]
    S, H, W, _ = pts.shape

    if imgs is not None:
        colors = imgs.detach().float().cpu()
        if colors.dim() == 5:
            colors = colors[0]
        if colors.shape[1] == 3:
            colors = colors.permute(0, 2, 3, 1).contiguous()      # [S,H,W,3]
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

    return {"means": means, "colors": cols}


def decode_gaussians(preds: dict, *, weight_threshold: float = 0.0,
                     downsample: float = 1.0) -> dict:
    """Predictions → HYWM2_GAUSSIANS (means/quats/scales/opacities/rgbs).

    Returns an empty HYWM2_GAUSSIANS dict if the gs head was disabled.
    """
    if "splats" not in preds:
        # 1-row sentinel rather than (0,…): comfy-env's worker shm serializer
        # rejects 0-element tensors via `rebuild_storage_empty`.
        log.info("HYWM2 decode_gaussians: gs head disabled → 1-row sentinel HYWM2_GAUSSIANS")
        return {
            "means": torch.zeros((1, 3)),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # identity quat
            "scales": torch.full((1, 3), 1e-6),              # near-zero ellipsoid
            "opacities": torch.zeros((1,)),
            "rgbs": torch.zeros((1, 3)),
        }
    s = preds["splats"]

    means = s["means"].detach().float().cpu()
    quats = s["quats"].detach().float().cpu()
    scales = s["scales"].detach().float().cpu()
    opacities = s["opacities"].detach().float().cpu()
    sh = s["sh"].detach().float().cpu()                          # [B,N,1,3]
    weights = s.get("weights", None)
    if weights is not None:
        weights = weights.detach().float().cpu()

    if means.dim() == 3:
        means, quats, scales, opacities = means[0], quats[0], scales[0], opacities[0]
        sh = sh[0]
        if weights is not None:
            weights = weights[0]

    sh_dc = sh[..., 0, :]                                         # [N,3]
    rgbs = (sh_dc * C0 + 0.5).clamp(0, 1)

    if weight_threshold > 0 and weights is not None:
        keep = weights >= weight_threshold
        means, quats, scales, opacities, rgbs = (
            means[keep], quats[keep], scales[keep], opacities[keep], rgbs[keep]
        )

    ratio = max(0.0, min(1.0, float(downsample)))
    if ratio < 1.0 and means.shape[0] > 0:
        n = means.shape[0]
        k = max(1, int(round(n * ratio))) if ratio > 0 else 0
        if k < n:
            idx = torch.randperm(n)[:k]
            means, quats, scales, opacities, rgbs = (
                means[idx], quats[idx], scales[idx], opacities[idx], rgbs[idx]
            )

    return {
        "means": means, "quats": quats, "scales": scales,
        "opacities": opacities, "rgbs": rgbs,
    }


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


class HYWM2ExportGaussiansSplat(io.ComfyNode):
    """HYWM2_GAUSSIANS → ``.splat`` (Antimatter15 binary, 32 B/Gaussian).

    Format per Gaussian (sorted by visibility): position [12 B float32] +
    linear scales [12 B float32] + color RGBA8 [4 B uchar] + quaternion
    quantized to uchar [4 B] = 32 B total. Eaten by SuperSplat,
    PlayCanvas viewer, Antimatter15's WebGL viewer, etc. Mirrors
    upstream ``process_ply_to_splat`` but skips the PLY round-trip.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2ExportGaussiansSplat",
            display_name="HYWM2 Export Gaussians .splat",
            category="HYWM2",
            description=(
                "Write the compact 32 B/Gaussian .splat binary used by "
                "SuperSplat / PlayCanvas / Antimatter15's viewer. Sorted "
                "by visibility (largest most-opaque first)."
            ),
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
            raise ValueError("HYWM2ExportGaussiansSplat: expected HYWM2_GAUSSIANS dict")

        out = _resolve_output_path(filename, output_dir, ".splat")

        means = gaussians["means"].detach().float().cpu().numpy().astype(np.float32)
        scales = gaussians["scales"].detach().float().cpu().numpy().astype(np.float32)
        quats = gaussians["quats"].detach().float().cpu().numpy().astype(np.float32)
        opacities = gaussians["opacities"].detach().float().cpu().numpy().astype(np.float32)
        rgbs = gaussians["rgbs"].detach().float().cpu().numpy().astype(np.float32)

        n = means.shape[0]
        if n == 0:
            out.write_bytes(b"")
            log.info("HYWM2ExportGaussiansSplat: 0 gaussians; wrote empty %s", out)
            return io.NodeOutput(str(out))

        # Visibility sort: −Σexp(scale) · sigmoid(opacity).
        # If opacities are logits, sigmoid them; if already in 0..1 (which
        # happens when the model emits pre-sigmoided alphas), pass through.
        if (opacities.min() >= 0.0) and (opacities.max() <= 1.0):
            alpha = opacities
        else:
            alpha = 1.0 / (1.0 + np.exp(-opacities))

        # `gaussians["scales"]` from decode_gaussians is the model's raw scale
        # output. WorldMirror stores them so that save_gs_ply's `.log()`
        # produces standard 3DGS log-scales — i.e. these are LINEAR. Use
        # them directly.
        scales_lin = scales

        sort_key = -(np.exp(scales_lin.sum(axis=-1)) * alpha)
        order = np.argsort(sort_key, kind="stable")

        means = means[order]
        scales_lin = scales_lin[order]
        quats = quats[order]
        rgbs = rgbs[order]
        alpha = alpha[order]

        # Quaternion quantization: normalize, then rescale to [0, 255].
        qnorm = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-9)
        qquant = np.clip(qnorm * 128.0 + 128.0, 0, 255).astype(np.uint8)

        rgba = np.empty((n, 4), dtype=np.uint8)
        rgba[:, :3] = np.clip(rgbs * 255.0 + 0.5, 0, 255).astype(np.uint8)
        rgba[:, 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)

        # Pack: [pos(3)][scale(3)][rgba(4)][quat(4)] = 12+12+4+4 = 32 B
        rec = np.empty(n, dtype=[
            ("pos", np.float32, 3),
            ("scale", np.float32, 3),
            ("rgba", np.uint8, 4),
            ("quat", np.uint8, 4),
        ])
        rec["pos"] = means
        rec["scale"] = scales_lin
        rec["rgba"] = rgba
        rec["quat"] = qquant
        with open(out, "wb") as f:
            f.write(rec.tobytes())

        log.info("HYWM2ExportGaussiansSplat: wrote %s (%d gaussians, %.1f MB)",
                 out, n, out.stat().st_size / 1e6)
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
