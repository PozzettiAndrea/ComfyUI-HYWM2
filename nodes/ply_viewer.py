"""HYWM2 PLY Advanced Gaussian Viewer node.

Browser-side viewer with a 3D canvas (left) and a tabbed metadata panel
(right) that introspects the actual fields present in the PLY file —
their dtypes, counts, and how each is interpreted at render time
(SH→RGB, exp(scale), normalized quaternion, sigmoid opacity, etc.).

The same per-field stats / interpretation / sanity-check / byte-layout
table is also printed to the terminal so the result can be diffed
against the in-browser panel.
"""

import logging
import os
import re
from pathlib import Path

import folder_paths
from comfy_api.latest import io

log = logging.getLogger("hywm2")


# ---------------------------------------------------------------------------
# Field interpretation (mirrors the JS rules in viewer.html)
# ---------------------------------------------------------------------------

_INTERP_RULES = [
    (re.compile(r"^x$|^y$|^z$"),         "position (world coords)",                                       None),
    (re.compile(r"^nx$|^ny$|^nz$"),      "stored normal (unused at render — 3DGS PLYs zero them)",        (-1.001, 1.001)),
    (re.compile(r"^f_dc_[012]$"),        "SH₀ DC band → RGB = clamp(0.5 + C₀·v, 0, 1), C₀ = 0.282",       (-5, 8)),
    (re.compile(r"^f_rest_"),            "higher-order SH coefficient (band ≥ 1)",                        (-5, 5)),
    (re.compile(r"^opacity$"),           "logit → α = sigmoid(opacity)",                                  (-15, 15)),
    (re.compile(r"^scale_[012]$"),       "log-scale → σᵢ = exp(scaleᵢ)",                                  (-15, 2)),
    (re.compile(r"^rot_[0123]$"),        "quaternion component (normalized at render: [w,x,y,z])",        (-1.001, 1.001)),
    (re.compile(r"^red$|^green$|^blue$"),"vertex color (uchar 0..255 → /255)",                            (0, 255)),
]

_TYPE_BYTES = {
    "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
    "int": 4, "uint": 4, "int32": 4, "uint32": 4,
    "float": 4, "float32": 4, "double": 8, "float64": 8,
}


def _interpret(name: str):
    for rx, label, expected in _INTERP_RULES:
        if rx.match(name):
            return label, expected
    return "raw value (no built-in interpretation)", None


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:                                      # NaN
        return "nan"
    af = abs(f)
    if af >= 1000 or (af > 0 and af < 1e-3):
        return f"{f:.3e}"
    return f"{f:.4f}"


def _print_debug_stats(ply_path: str) -> None:
    """Parse the PLY and print everything-we-need-to-debug to stdout."""
    try:
        from plyfile import PlyData
        import numpy as np
    except ImportError as e:
        log.warning("[PLY Advanced Viewer] cannot print stats — missing dep: %s", e)
        return

    try:
        data = PlyData.read(ply_path)
    except Exception as e:
        log.warning("[PLY Advanced Viewer] failed to parse %s: %s", ply_path, e)
        return

    file_bytes = os.path.getsize(ply_path)
    fmt_name = "ascii" if data.text else (
        "binary_big_endian" if data.byte_order == ">" else "binary_little_endian"
    )

    bar = "═" * 100
    print(f"\n{bar}")
    print(f"[HYWM2 PLY Advanced Viewer] {ply_path}")
    print(f"  Format     : {fmt_name}")
    print(f"  Size       : {file_bytes / (1024 * 1024):.2f} MB ({file_bytes:,} bytes)")
    if data.comments:
        print(f"  Comments   : {' / '.join(data.comments)}")
    print(f"  Elements   : {', '.join(f'{el.name} ({el.count:,})' for el in data.elements)}")

    # Vertex element only — the rest of 3DGS attrs live there.
    vertex = next((el for el in data.elements if el.name == "vertex"), None)
    if vertex is None:
        print(f"{bar}\n  (no 'vertex' element — nothing else to introspect)\n{bar}\n", flush=True)
        return

    props = list(vertex.properties)
    stride = sum(_TYPE_BYTES.get(_dtype_name(p), 0) for p in props)
    n = vertex.count
    print(f"  Vertices   : {n:,}")
    print(f"  Stride     : {stride} bytes ({len(props)} properties)")
    print(bar)

    # Per-field stats + sanity check
    arr = vertex.data                                   # structured numpy array
    rows = []
    sanity_warnings = []
    for p in props:
        name = p.name
        dtype_str = _dtype_name(p)
        col = np.asarray(arr[name])
        if col.size == 0:
            mn = mx = mean = float("nan")
        else:
            mn = float(col.min())
            mx = float(col.max())
            mean = float(col.mean())
        label, expected = _interpret(name)
        if expected is None:
            sanity = "·"
        else:
            lo, hi = expected
            ok = mn >= lo and mx <= hi
            sanity = "✓" if ok else "⚠"
            if not ok:
                sanity_warnings.append((name, mn, mx, lo, hi, label))
        rows.append((name, dtype_str, mn, mx, mean, sanity, label))

    name_w = max(len("Field"), max(len(r[0]) for r in rows))
    type_w = max(len("Type"),  max(len(r[1]) for r in rows))
    val_w = 12
    print(
        f"  {'Field':<{name_w}}  {'Type':<{type_w}}  "
        f"{'min':>{val_w}}  {'max':>{val_w}}  {'mean':>{val_w}}  ✓  Interpretation"
    )
    print("  " + "─" * (name_w + type_w + val_w * 3 + 22))
    for name, dtype_str, mn, mx, mean, sanity, label in rows:
        print(
            f"  {name:<{name_w}}  {dtype_str:<{type_w}}  "
            f"{_fmt(mn):>{val_w}}  {_fmt(mx):>{val_w}}  {_fmt(mean):>{val_w}}  "
            f"{sanity}  {label}"
        )

    # Byte layout + first two vertices' decoded values
    print(bar)
    print(f"  Byte layout (stride = {stride} bytes per vertex)")
    print(f"  {'Off':>4}  {'Field':<{name_w}}  {'Type':<{type_w}}  {'v[0]':>{val_w}}  {'v[1]':>{val_w}}")
    print("  " + "─" * (4 + name_w + type_w + val_w * 2 + 8))
    off = 0
    for p in props:
        dtype_str = _dtype_name(p)
        v0 = arr[p.name][0] if n >= 1 else None
        v1 = arr[p.name][1] if n >= 2 else None
        print(
            f"  {off:>4}  {p.name:<{name_w}}  {dtype_str:<{type_w}}  "
            f"{_fmt(v0):>{val_w}}  {_fmt(v1):>{val_w}}"
        )
        off += _TYPE_BYTES.get(dtype_str, 0)

    # Sanity-check summary
    print(bar)
    if sanity_warnings:
        print(f"  ⚠ {len(sanity_warnings)} field(s) outside expected 3DGS range "
              f"(parser bug or non-standard PLY):")
        for name, mn, mx, lo, hi, label in sanity_warnings:
            print(f"     {name}: observed [{_fmt(mn)}, {_fmt(mx)}], "
                  f"expected [{_fmt(lo)}, {_fmt(hi)}]  ({label})")
    else:
        print("  ✓ All fields with known 3DGS expected ranges look sane.")
    print(bar + "\n", flush=True)


def _dtype_name(prop) -> str:
    """Best-effort dtype string for a PlyProperty (handles plyfile API drift)."""
    for attr in ("val_dtype", "dtype"):
        v = getattr(prop, attr, None)
        if isinstance(v, str):
            return v
        if callable(v):
            try:
                return str(v())
            except Exception:
                pass
    return getattr(prop, "type", "?")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

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
                "by the viewer (SH→RGB, exp scales, normalized quats, "
                "sigmoid opacity). Also prints the same debug table to the "
                "terminal for diffing against the in-browser panel."
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
                io.Boolean.Input(
                    "print_stats",
                    default=True,
                    tooltip="Also dump per-field stats / interpretation / byte layout / sanity check to the terminal.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Force re-execution when the path changes (and on stat tweaks).
        path = kwargs.get("ply_path", "") or ""
        try:
            mtime = os.path.getmtime(path) if path and os.path.exists(path) else 0.0
            size = os.path.getsize(path) if path and os.path.exists(path) else 0
        except OSError:
            mtime, size = 0.0, 0
        return f"{path}|{mtime}|{size}|{kwargs.get('print_stats', True)}"

    @classmethod
    def execute(cls, ply_path: str = "", print_stats: bool = True):
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

        if print_stats:
            try:
                _print_debug_stats(ply_path)
            except Exception as e:
                log.warning("[PLY Advanced Viewer] stats dump failed: %s", e)

        return io.NodeOutput(ui={
            "ply_path": [ply_path],
            "ply_url": [url],
            "filename": [os.path.basename(ply_path)],
            "file_size_bytes": [size],
        })

    @staticmethod
    def _build_view_url(ply_path: str) -> str:
        """Map an on-disk path → ComfyUI /view URL preserving subfolder."""
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
