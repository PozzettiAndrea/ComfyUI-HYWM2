"""HYWM2 PLY Advanced Gaussian Viewer node.

Browser-side viewer with a 3D canvas (left) and a tabbed metadata panel
(right) that introspects the actual fields present in the PLY file —
their dtypes, counts, and how each is interpreted at render time
(SH->RGB, exp(scale), normalized quaternion, sigmoid opacity, etc.).

The same per-field stats / interpretation / sanity-check / byte-layout
table is also dumped to the **browser DevTools console** by
viewer.html (open the iframe's console; tag = "[HYWM2 PLY Advanced
Viewer]"). Server-side terminal printing was attempted earlier but
comfy-env's subprocess worker swallows non-print stdout, so the JS
side is the source of truth for the debug dump.
"""

import logging
import os

import folder_paths
from comfy_api.latest import io

log = logging.getLogger("hywm2")


class HYWM2PLYAdvancedGaussianViewer(io.ComfyNode):
    """Advanced 3DGS PLY viewer — visualizes splats and surfaces field-level metadata."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2PLYAdvancedGaussianViewer",
            display_name="PLY Advanced Gaussian Viewer",
            category="HYWM2",
            description=(
                "Preview a 3DGS PLY in 3D and inspect its raw field layout "
                "(names, dtypes, counts) plus how each field is interpreted "
                "by the viewer (SH->RGB, exp scales, normalized quats, "
                "sigmoid opacity). Open the browser DevTools console for a "
                "full per-field stats / byte-layout / sanity-check dump on "
                "every reload."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "ply_path",
                    default="",
                    multiline=False,
                    force_input=True,
                    tooltip="Path to a Gaussian Splatting PLY file (typically the output of HYWM2ExportGaussiansPLY).",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        path = kwargs.get("ply_path", "") or ""
        try:
            mtime = os.path.getmtime(path) if path and os.path.exists(path) else 0.0
            size = os.path.getsize(path) if path and os.path.exists(path) else 0
        except OSError:
            mtime, size = 0.0, 0
        return f"{path}|{mtime}|{size}"

    @classmethod
    def execute(cls, ply_path: str = ""):
        ply_path = (ply_path or "").strip()
        if not ply_path:
            return io.NodeOutput(ui={"error": ["No PLY path provided"]})
        if not os.path.exists(ply_path):
            return io.NodeOutput(ui={"error": [f"File not found: {ply_path}"]})

        url = cls._build_view_url(ply_path)
        size = os.path.getsize(ply_path)
        log.info(
            "[PLY Advanced Viewer] %s -> %s (%.2f MB)",
            ply_path, url, size / (1024 * 1024),
        )

        return io.NodeOutput(ui={
            "ply_path": [ply_path],
            "ply_url": [url],
            "filename": [os.path.basename(ply_path)],
            "file_size_bytes": [size],
        })

    @staticmethod
    def _build_view_url(ply_path: str) -> str:
        """Map an on-disk path -> ComfyUI /view URL preserving subfolder."""
        normalized = ply_path.replace("\\", "/")
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
        return f"/view?filename={os.path.basename(ply_path)}&type=output&subfolder="
