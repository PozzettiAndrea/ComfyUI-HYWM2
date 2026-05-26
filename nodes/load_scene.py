"""HYWM2LoadScene -- one-shot loader for a WorldStereo scene folder.

A "scene" is a folder produced by WorldStereo's panorama-anchored
multi-trajectory generation flow. Layout:

    <scene_dir>/
        memorybanks/
            memorybank_traj_0/ ... memorybank_traj_N/
                cameras.json + frames/ + (optional) depths/
        pointclouds/
            pointcloud_traj_0.ply ... pointcloud_traj_N.ply
                (or pointcloud_initial.ply / pointcloud_final.ply)
        panorama.png                  (optional, informational)

This node emits everything downstream nodes need to run the depth-
correction / PCD-growth flow in one go: the latest bank's frames +
cameras + the latest PCD. No need for a separate
WorldStereoLoadPointCloud or a folder-string-typed in two places.

For scene discovery, this node scans both ComfyUI/input/ AND
ComfyUI/output/ for top-level folders that contain BOTH a
`memorybanks/` and a `pointclouds/` subdirectory.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[HYWM2LoadScene] {msg}", file=sys.stderr, flush=True)


def _looks_like_scene(p: Path) -> bool:
    return p.is_dir() and (p / "memorybanks").is_dir() and (p / "pointclouds").is_dir()


def _list_scene_folders() -> list[str]:
    """Scan ComfyUI/input/ and ComfyUI/output/ for scene folders.

    Returns labels of the form "<root_tag>/<scene_name>" where root_tag is
    'input' or 'output'. Sorted. Returns ['<none>'] if nothing found.
    """
    try:
        import folder_paths
        roots = [
            ("input", Path(folder_paths.get_input_directory())),
            ("output", Path(folder_paths.get_output_directory())),
        ]
    except Exception:
        return ["<none>"]

    out: list[str] = []
    for tag, root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if _looks_like_scene(child):
                out.append(f"{tag}/{child.name}")
    return out or ["<none>"]


def _resolve_scene_path(label_or_path: str) -> Path:
    """Turn a dropdown label or override string into an absolute scene path."""
    import folder_paths

    raw = (label_or_path or "").strip()
    if not raw or raw == "<none>":
        raise ValueError("HYWM2LoadScene: no scene selected.")

    # If it's a dropdown label like "input/scene_000" or "output/scene_000",
    # split off the root tag.
    if raw.startswith("input/") or raw.startswith("output/"):
        tag, rest = raw.split("/", 1)
        root = (Path(folder_paths.get_input_directory()) if tag == "input"
                else Path(folder_paths.get_output_directory()))
        return root / rest

    # Else: treat as absolute or relative-to-output path.
    p = Path(raw)
    if not p.is_absolute():
        # Try output/, then input/, then cwd.
        for cand in (Path(folder_paths.get_output_directory()) / raw,
                     Path(folder_paths.get_input_directory()) / raw,
                     Path.cwd() / raw):
            if _looks_like_scene(cand):
                return cand
        # Fall through with the relative-to-output guess so error message
        # is informative.
        p = Path(folder_paths.get_output_directory()) / raw
    return p


def _pick_latest(dir_: Path, suffix: str) -> Path | None:
    """Pick the file in `dir_` with the most recent mtime (tiebreak: name)."""
    if not dir_.is_dir():
        return None
    files = [p for p in dir_.iterdir() if p.is_file() and p.suffix.lower() == suffix]
    if not files:
        return None
    files.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return files[-1]


def _pick_latest_dir(dir_: Path) -> Path | None:
    """Pick the subdir with the most recent mtime (tiebreak: name)."""
    if not dir_.is_dir():
        return None
    subs = [p for p in dir_.iterdir() if p.is_dir()]
    if not subs:
        return None
    subs.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return subs[-1]


class HYWM2LoadScene(io.ComfyNode):
    """Load a full WorldStereo scene (latest bank + latest PCD)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2LoadScene",
            display_name="HYWM2 Load Scene",
            category="HYWM2",
            description=(
                "Load a WorldStereo scene folder in one node. Picks the "
                "latest memorybank subdir + the latest pointcloud .ply by "
                "mtime and emits everything the depth-correction flow "
                "needs (images, extrinsics, intrinsics, fov, pcd).\n\n"
                "Scene folder layout:\n"
                "  <scene>/memorybanks/<bank_name>/cameras.json + frames/\n"
                "  <scene>/pointclouds/*.ply\n\n"
                "The dropdown scans both ComfyUI/input/ and ComfyUI/output/ "
                "for any top-level folder containing BOTH memorybanks/ + "
                "pointclouds/. Restart ComfyUI to refresh after copying new "
                "scenes in."
            ),
            inputs=[
                io.Combo.Input(
                    "scene",
                    options=_list_scene_folders(),
                    tooltip="<root>/<scene_name> from input/ or output/. "
                            "Root prefix tells the node which directory to "
                            "resolve relative to."),
                io.String.Input(
                    "scene_override", default="", multiline=False,
                    optional=True,
                    tooltip="Absolute path (or path relative to "
                            "ComfyUI/output/) overriding the dropdown. Use "
                            "for scenes outside the standard input/output "
                            "directories."),
                io.String.Input(
                    "bank_override", default="", multiline=False,
                    optional=True,
                    tooltip="Override which memorybank subdir to load. "
                            "Default = latest by mtime. Set to a bank name "
                            "(e.g. 'memorybank_traj_5') to pin a specific "
                            "one."),
                io.String.Input(
                    "pointcloud_override", default="", multiline=False,
                    optional=True,
                    tooltip="Override which .ply to load. Default = latest "
                            "by mtime. Set to a filename (e.g. "
                            "'pointcloud_initial.ply') to pin a specific "
                            "one. Absolute paths also accepted."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="[N, H, W, 3] float in [0,1] from the bank's frames/."),
                io.Custom("EXTRINSICS").Output(
                    display_name="extrinsics",
                    tooltip="[N, 4, 4] w2c from the bank's cameras.json."),
                io.Custom("INTRINSICS").Output(
                    display_name="intrinsics",
                    tooltip="[N, 3, 3] K from the bank's cameras.json."),
                io.Int.Output(
                    display_name="num_entries",
                    tooltip="N — number of frames + cameras loaded."),
                io.Float.Output(
                    display_name="fov_x_deg",
                    tooltip="Horizontal FoV derived from median fx in the "
                            "bank's intrinsics. Wires into MoGe2Inference."),
                io.Custom("TRIMESH").Output(
                    display_name="pointcloud",
                    tooltip="The selected .ply loaded as a trimesh.PointCloud. "
                            "Wire into WorldStereoAlignDepthAndGrowPCD."),
                io.String.Output(
                    display_name="scene_path",
                    tooltip="Absolute path to the loaded scene folder. "
                            "Useful for routing to downstream save nodes."),
            ],
        )

    @classmethod
    def execute(cls, scene: str, scene_override: str = "",
                bank_override: str = "", pointcloud_override: str = ""):
        # ---- Resolve scene folder -----------------------------------
        raw = (scene_override or "").strip() or scene
        scene_dir = _resolve_scene_path(raw)
        if not _looks_like_scene(scene_dir):
            raise FileNotFoundError(
                f"HYWM2LoadScene: {scene_dir} doesn't look like a scene "
                f"(needs memorybanks/ + pointclouds/ subdirs)."
            )
        _p(f"scene: {scene_dir}")

        # ---- Pick bank ----------------------------------------------
        banks_root = scene_dir / "memorybanks"
        if bank_override.strip():
            bank_dir = banks_root / bank_override.strip()
            if not bank_dir.is_dir():
                raise FileNotFoundError(
                    f"HYWM2LoadScene: bank override '{bank_override}' not "
                    f"found in {banks_root}"
                )
        else:
            bank_dir = _pick_latest_dir(banks_root)
            if bank_dir is None:
                raise FileNotFoundError(
                    f"HYWM2LoadScene: no subdirs in {banks_root}"
                )
        if not (bank_dir / "cameras.json").is_file() or not (bank_dir / "frames").is_dir():
            raise FileNotFoundError(
                f"HYWM2LoadScene: {bank_dir} missing cameras.json or frames/"
            )
        _p(f"bank: {bank_dir.name}")

        # ---- Load bank (cameras + frames) ---------------------------
        cameras = json.loads((bank_dir / "cameras.json").read_text())
        ext_np = np.asarray(cameras.get("extrinsics", []), dtype=np.float32)
        K_np = np.asarray(cameras.get("intrinsics", []), dtype=np.float32)
        if ext_np.ndim != 3 or ext_np.shape[1:] != (4, 4):
            raise ValueError(
                f"cameras.json extrinsics shape {ext_np.shape} not [N, 4, 4]"
            )
        if K_np.ndim != 3 or K_np.shape[1:] != (3, 3):
            raise ValueError(
                f"cameras.json intrinsics shape {K_np.shape} not [N, 3, 3]"
            )
        N = int(ext_np.shape[0])
        if K_np.shape[0] != N:
            raise ValueError(
                f"cameras.json count mismatch: {ext_np.shape[0]} ext vs "
                f"{K_np.shape[0]} K"
            )

        frames_dir = bank_dir / "frames"
        frame_paths = sorted(
            p for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
            and not p.name.startswith(".")
        )
        if len(frame_paths) != N:
            raise ValueError(
                f"frame count mismatch: {len(frame_paths)} images in "
                f"{frames_dir} vs {N} cameras"
            )
        frames_arr = np.stack(
            [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
             for p in frame_paths],
            axis=0,
        )
        H, W = int(frames_arr.shape[1]), int(frames_arr.shape[2])
        images_t = torch.from_numpy(frames_arr.astype(np.float32) / 255.0)
        extrinsics_t = torch.from_numpy(ext_np)
        intrinsics_t = torch.from_numpy(K_np)

        # ---- fov_x_deg from intrinsics ------------------------------
        fx_values = K_np[:, 0, 0].astype(np.float64)
        fx_med = float(np.median(fx_values))
        if fx_med <= 0:
            _p(f"WARNING: median fx={fx_med} non-positive; fov_x_deg=0")
            fov_x_deg = 0.0
        else:
            rel_dev = np.abs(fx_values - fx_med) / fx_med
            if rel_dev.max() > 0.01:
                worst = int(np.argmax(rel_dev))
                _p(f"WARNING: intrinsics fx not uniform across {N} entries — "
                   f"median={fx_med:.2f}, worst entry {worst} "
                   f"fx={fx_values[worst]:.2f} (dev={rel_dev.max()*100:.2f}%)")
            fov_x_deg = math.degrees(2.0 * math.atan(W / (2.0 * fx_med)))

        # ---- Pick + load pointcloud --------------------------------
        import trimesh
        pcd_dir = scene_dir / "pointclouds"
        if pointcloud_override.strip():
            raw_pcd = pointcloud_override.strip()
            pcd_path = Path(raw_pcd)
            if not pcd_path.is_absolute():
                pcd_path = pcd_dir / raw_pcd
            if not pcd_path.is_file():
                raise FileNotFoundError(
                    f"HYWM2LoadScene: pointcloud override {pcd_path} not found"
                )
        else:
            pcd_path = _pick_latest(pcd_dir, ".ply")
            if pcd_path is None:
                raise FileNotFoundError(
                    f"HYWM2LoadScene: no .ply files in {pcd_dir}. Either "
                    f"copy a pointcloud .ply in or pass pointcloud_override."
                )
        pcd = trimesh.load(str(pcd_path))
        if not hasattr(pcd, "vertices") or pcd.vertices.shape[0] == 0:
            raise ValueError(
                f"HYWM2LoadScene: {pcd_path} loaded but has zero vertices"
            )
        _p(f"pointcloud: {pcd_path.name} ({pcd.vertices.shape[0]} vertices)")

        _p(f"loaded N={N}, image_size=({W},{H}), fov_x={fov_x_deg:.2f} deg")
        return io.NodeOutput(
            images_t, extrinsics_t, intrinsics_t, N, fov_x_deg, pcd,
            str(scene_dir),
        )
