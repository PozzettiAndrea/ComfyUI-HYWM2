"""HYWM2 .splat Advanced Viewer node.

Browser-side viewer for the 32 B/Gaussian Antimatter15 ``.splat`` binary
format. Same mkkellogg renderer as the PLY viewer, but with a side panel
that introspects the fixed splat record layout (position 0..11,
scale 12..23, rgba8 24..27, quat-quantized 28..31) and runs the same
sanity checks (alpha 0..255, quat dequantization round-trip ~ unit
quaternion, scales positive, etc.).
"""

import logging
import os

import folder_paths
from comfy_api.latest import io

log = logging.getLogger("hywm2")


class HYWM2SplatAdvancedViewer(io.ComfyNode):
    """Advanced .splat viewer — visualizes 3DGS and surfaces record-level metadata."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2SplatAdvancedViewer",
            display_name="Splat Viewer",
            category="HYWM2",
            description=(
                "Preview a 3DGS .splat binary in 3D and inspect its 32 B "
                "record layout (pos / scale / rgba8 / quat-quantized) plus "
                "per-attribute stats. Real-time mkkellogg/gaussian-splats-3d "
                "rasterization on the left."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "splat_path",
                    default="",
                    multiline=False,
                    force_input=True,
                    tooltip="Path to a .splat file (typically the output of HYWM2ExportGaussiansSplat).",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        path = kwargs.get("splat_path", "") or ""
        try:
            mtime = os.path.getmtime(path) if path and os.path.exists(path) else 0.0
            size = os.path.getsize(path) if path and os.path.exists(path) else 0
        except OSError:
            mtime, size = 0.0, 0
        return f"{path}|{mtime}|{size}"

    @classmethod
    def execute(cls, splat_path: str = ""):
        splat_path = (splat_path or "").strip()
        if not splat_path:
            return io.NodeOutput(ui={"error": ["No .splat path provided"]})
        if not os.path.exists(splat_path):
            return io.NodeOutput(ui={"error": [f"File not found: {splat_path}"]})

        url = cls._build_view_url(splat_path)
        size = os.path.getsize(splat_path)
        n_gaussians = size // 32
        log.info(
            "[Splat Viewer] %s -> %s (%.2f MB, %d gaussians)",
            splat_path, url, size / (1024 * 1024), n_gaussians,
        )

        return io.NodeOutput(ui={
            "splat_path": [splat_path],
            "splat_url": [url],
            "filename": [os.path.basename(splat_path)],
            "file_size_bytes": [size],
            "gaussian_count": [n_gaussians],
        })

    @staticmethod
    def _build_view_url(splat_path: str) -> str:
        normalized = splat_path.replace("\\", "/")
        for kind, root in (
            ("output", folder_paths.get_output_directory()),
            ("input", folder_paths.get_input_directory()),
            ("temp", folder_paths.get_temp_directory()),
        ):
            try:
                rel = os.path.relpath(normalized, root)
            except ValueError:
                continue
            if rel.startswith(".."):
                continue
            rel = rel.replace("\\", "/")
            parts = rel.split("/")
            filename = parts[-1]
            subfolder = "/".join(parts[:-1])
            return f"/view?filename={filename}&type={kind}&subfolder={subfolder}"
        return f"/view?filename={os.path.basename(splat_path)}&type=output&subfolder="
