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
import sys
import traceback

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

# Map both numpy ('<f4') and PLY ASCII ('float') to byte sizes.
_TYPE_BYTES = {
    # PLY ASCII names
    "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
    "int": 4, "uint": 4, "int32": 4, "uint32": 4,
    "float": 4, "float32": 4, "double": 8, "float64": 8,
    # numpy dtype kind+size strings
    "i1": 1, "u1": 1, "i2": 2, "u2": 2, "i4": 4, "u4": 4, "i8": 8, "u8": 8,
    "f4": 4, "f8": 8,
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
    if f != f:                                                     # NaN
        return "nan"
    af = abs(f)
    if af >= 1000 or (af > 0 and af < 1e-3):
        return f"{f:.3e}"
    return f"{f:.4f}"


def _dtype_name(prop) -> str:
    """Return a normalized PLY ASCII type name ('float', 'uchar', ...)."""
    # plyfile stores `val_dtype` as a numpy-style code like 'f4', '<f4', 'u1'.
    raw = getattr(prop, "val_dtype", None) or getattr(prop, "dtype", None) or ""
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = ""
    raw = str(raw).lstrip("<>=|")
    np_to_ply = {
        "f4": "float", "f8": "double",
        "i1": "char", "u1": "uchar",
        "i2": "short", "u2": "ushort",
        "i4": "int", "u4": "uint",
        "i8": "int64", "u8": "uint64",
    }
    return np_to_ply.get(raw, raw or getattr(prop, "type", "?"))


# ---------------------------------------------------------------------------
# Cache: (path, mtime, size) -> parsed PlyData
# Lets a fan-out of viewers on the same PLY pay parse cost once.
# ---------------------------------------------------------------------------

_PARSE_CACHE: dict = {}
_PARSE_CACHE_MAX = 4


def _read_ply_cached(ply_path: str):
    from plyfile import PlyData
    try:
        st = os.stat(ply_path)
    except OSError:
        return None
    key = (os.path.abspath(ply_path), int(st.st_mtime), st.st_size)
    cached = _PARSE_CACHE.get(key)
    if cached is not None:
        return cached
    data = PlyData.read(ply_path)
    if len(_PARSE_CACHE) >= _PARSE_CACHE_MAX:
        _PARSE_CACHE.pop(next(iter(_PARSE_CACHE)))
    _PARSE_CACHE[key] = data
    return data


def _print_debug_stats(ply_path: str) -> None:
    """Parse the PLY and print everything-we-need-to-debug to stdout.

    Failures inside this function are surfaced (with traceback) to stdout
    rather than being silently swallowed — debugging the debug printer is
    the user's primary use case.
    """
    # NOTE: comfy-env's subprocess worker forwards builtins.print over its
    # IPC socket; sys.stdout.write() is silently dropped. So we go through
    # print() unconditionally here.
    def write(s):
        print(s, end="", flush=True)

    def flush():
        return None
    bar = "═" * 100

    write(f"\n{bar}\n[HYWM2 PLY Advanced Viewer] {ply_path}\n")
    flush()

    try:
        from plyfile import PlyData                      # noqa: F401
        import numpy as np
    except ImportError as e:
        write(f"  ⚠ missing dep: {e}\n{bar}\n")
        flush()
        return

    try:
        data = _read_ply_cached(ply_path)
        if data is None:
            write(f"  ⚠ could not stat / read {ply_path}\n{bar}\n")
            flush()
            return
    except Exception:
        write("  ⚠ failed to parse PLY:\n")
        write(traceback.format_exc())
        write(bar + "\n")
        flush()
        return

    try:
        file_bytes = os.path.getsize(ply_path)
        fmt_name = "ascii" if data.text else (
            "binary_big_endian" if data.byte_order == ">" else "binary_little_endian"
        )
        write(f"  Format     : {fmt_name}\n")
        write(f"  Size       : {file_bytes / (1024 * 1024):.2f} MB ({file_bytes:,} bytes)\n")
        if data.comments:
            write(f"  Comments   : {' / '.join(data.comments)}\n")
        write(f"  Elements   : {', '.join(f'{el.name} ({el.count:,})' for el in data.elements)}\n")

        vertex = next((el for el in data.elements if el.name == "vertex"), None)
        if vertex is None:
            write(f"  (no 'vertex' element — nothing else to introspect)\n{bar}\n")
            flush()
            return

        props = list(vertex.properties)
        prop_widths = [(p, _dtype_name(p), _TYPE_BYTES.get(_dtype_name(p), 0)) for p in props]
        stride = sum(w for _, _, w in prop_widths)
        n = vertex.count
        write(f"  Vertices   : {n:,}\n")
        write(f"  Stride     : {stride} bytes ({len(props)} properties)\n")
        write(bar + "\n")

        arr = vertex.data
        rows = []
        sanity_warnings = []
        for p, dtype_str, _ in prop_widths:
            name = p.name
            try:
                col = np.asarray(arr[name])
                if col.size == 0:
                    mn = mx = mean = float("nan")
                else:
                    mn = float(col.min()); mx = float(col.max()); mean = float(col.mean())
            except Exception as e:
                mn = mx = mean = float("nan")
                write(f"  ⚠ stat failure on field {name}: {e}\n")

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

        name_w = max(5, max(len(r[0]) for r in rows))
        type_w = max(4, max(len(r[1]) for r in rows))
        val_w = 12
        write(
            f"  {'Field':<{name_w}}  {'Type':<{type_w}}  "
            f"{'min':>{val_w}}  {'max':>{val_w}}  {'mean':>{val_w}}  ✓  Interpretation\n"
        )
        write("  " + "─" * (name_w + type_w + val_w * 3 + 22) + "\n")
        for name, dtype_str, mn, mx, mean, sanity, label in rows:
            write(
                f"  {name:<{name_w}}  {dtype_str:<{type_w}}  "
                f"{_fmt(mn):>{val_w}}  {_fmt(mx):>{val_w}}  {_fmt(mean):>{val_w}}  "
                f"{sanity}  {label}\n"
            )
        flush()

        write(bar + "\n")
        write(f"  Byte layout (stride = {stride} bytes per vertex)\n")
        write(f"  {'Off':>4}  {'Field':<{name_w}}  {'Type':<{type_w}}  {'v[0]':>{val_w}}  {'v[1]':>{val_w}}\n")
        write("  " + "─" * (4 + name_w + type_w + val_w * 2 + 8) + "\n")
        off = 0
        for p, dtype_str, w in prop_widths:
            v0 = arr[p.name][0] if n >= 1 else None
            v1 = arr[p.name][1] if n >= 2 else None
            write(
                f"  {off:>4}  {p.name:<{name_w}}  {dtype_str:<{type_w}}  "
                f"{_fmt(v0):>{val_w}}  {_fmt(v1):>{val_w}}\n"
            )
            off += w
        flush()

        write(bar + "\n")
        if sanity_warnings:
            write(f"  ⚠ {len(sanity_warnings)} field(s) outside expected 3DGS range "
                  f"(parser bug or non-standard PLY):\n")
            for name, mn, mx, lo, hi, label in sanity_warnings:
                write(f"     {name}: observed [{_fmt(mn)}, {_fmt(mx)}], "
                      f"expected [{_fmt(lo)}, {_fmt(hi)}]  ({label})\n")
        else:
            write("  ✓ All fields with known 3DGS expected ranges look sane.\n")
        write(bar + "\n\n")
        flush()
    except Exception:
        write("  ⚠ stats dump failed mid-way:\n")
        write(traceback.format_exc())
        write(bar + "\n")
        flush()


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
                "sigmoid opacity). Optional terminal dump of the same debug "
                "table for diffing against the in-browser panel."
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
                    default=False,
                    tooltip="Dump per-field stats / interpretation / byte layout / sanity check to the terminal. Off by default to keep the queue snappy when fan-out is large.",
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
        return f"{path}|{mtime}|{size}|{kwargs.get('print_stats', False)}"

    @classmethod
    def execute(cls, ply_path: str = "", print_stats: bool = False):
        ply_path = (ply_path or "").strip()
        if not ply_path:
            return io.NodeOutput(ui={"error": ["No PLY path provided"]})
        if not os.path.exists(ply_path):
            return io.NodeOutput(ui={"error": [f"File not found: {ply_path}"]})

        url = cls._build_view_url(ply_path)
        size = os.path.getsize(ply_path)
        # One-line trace via print() so the comfy-env worker forwards it.
        print(
            f"[HYWM2 PLY Advanced Viewer] {os.path.basename(ply_path)} "
            f"({size / (1024*1024):.2f} MB) -> {url} "
            f"[print_stats={print_stats}]",
            flush=True,
        )

        if print_stats:
            _print_debug_stats(ply_path)

        return io.NodeOutput(ui={
            "ply_path": [ply_path],
            "ply_url": [url],
            "filename": [os.path.basename(ply_path)],
            "file_size_bytes": [size],
        })

    @staticmethod
    def _build_view_url(ply_path: str) -> str:
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
