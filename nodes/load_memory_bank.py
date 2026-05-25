"""HYWM2LoadMemoryBank -- read a WorldStereo-style memorybank folder from
disk and emit IMAGE + EXTRINSICS + INTRINSICS for HYWM2Reconstruct.

The folder layout matches what `WorldStereoSaveMemoryBank` writes (and what
upstream HY-World's `apply_worldmirror` produces):

    <folder>/
        frames/
            0000.png  0001.png  ...   (uint8 RGB, one per bank entry)
        cameras.json                  {"extrinsics": [...], "intrinsics": [...]}
        meta.json                     bank-wide metadata (informational)
        depths/                       (optional; ignored -- HYWM2 makes its own)

Folder is resolved relative to ComfyUI/input/ (so the prestartup-copied
assets show up in the dropdown), or accepts an absolute path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[HYWM2LoadMemoryBank] {msg}", file=sys.stderr, flush=True)


def _looks_like_bank(p: Path) -> bool:
    """True if `p` is a directory containing the bank's required files."""
    return (p.is_dir()
            and (p / "cameras.json").is_file()
            and (p / "frames").is_dir())


def _list_bank_folders_under_input() -> list[str]:
    """Scan ComfyUI/input/ for folders that look like memorybanks.

    Returns folder NAMES (relative to input/), sorted. Used to populate
    the Combo dropdown. Returns ['<none>'] if no candidates found, so the
    Combo still has at least one option (otherwise ComfyUI sometimes
    rejects the schema).
    """
    try:
        import folder_paths
        input_dir = Path(folder_paths.get_input_directory())
    except Exception:
        return ["<none>"]
    if not input_dir.is_dir():
        return ["<none>"]
    out = []
    for child in sorted(input_dir.iterdir()):
        if _looks_like_bank(child):
            out.append(child.name)
    return out or ["<none>"]


class HYWM2LoadMemoryBank(io.ComfyNode):
    """Load a memorybank folder -> IMAGE + EXTRINSICS + INTRINSICS for HYWM2."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2LoadMemoryBank",
            display_name="HYWM2 Load Memory Bank",
            category="HYWM2",
            description=(
                "Load a WorldStereo-style memorybank folder from disk and emit "
                "(images, extrinsics, intrinsics) ready to wire into "
                "HYWM2Reconstruct.\n\n"
                "Expects the folder layout written by WorldStereoSaveMemoryBank "
                "(and what upstream HY-World's apply_worldmirror produces):\n"
                "  <folder>/\n"
                "    frames/0000.png 0001.png ...     (uint8 RGB per entry)\n"
                "    cameras.json                     ({extrinsics, intrinsics})\n"
                "    meta.json                        (bank metadata)\n"
                "    depths/                          (optional; ignored)\n\n"
                "The `folder` dropdown lists candidate memorybanks detected "
                "under ComfyUI/input/. To load from an absolute path or one "
                "not in the dropdown, type the path into the `folder_override` "
                "string input (takes precedence)."
            ),
            inputs=[
                io.Combo.Input(
                    "folder",
                    options=_list_bank_folders_under_input(),
                    tooltip="Memorybank folder under ComfyUI/input/. Auto-"
                            "populated from any subdirs containing "
                            "cameras.json + frames/. Restart ComfyUI to "
                            "refresh after copying new banks into input/."),
                io.String.Input(
                    "folder_override", default="", multiline=False,
                    optional=True,
                    tooltip="Optional absolute path (or path relative to "
                            "ComfyUI/input/) overriding the dropdown. Use "
                            "this for banks outside input/ or before the "
                            "next ComfyUI restart picks them up."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="[N, H, W, 3] float in [0,1]. Wire into "
                            "HYWM2Reconstruct.images."),
                io.Custom("EXTRINSICS").Output(
                    display_name="extrinsics",
                    tooltip="[N, 4, 4] w2c (CameraPack convention). Wire into "
                            "HYWM2Reconstruct.prior_extrinsics."),
                io.Custom("INTRINSICS").Output(
                    display_name="intrinsics",
                    tooltip="[N, 3, 3] pinhole K (pixel units). Wire into "
                            "HYWM2Reconstruct.prior_intrinsics."),
                io.Int.Output(
                    display_name="num_entries",
                    tooltip="N. Number of frames + cameras loaded."),
            ],
        )

    @classmethod
    def execute(cls, folder: str, folder_override: str = ""):
        import folder_paths
        input_dir = Path(folder_paths.get_input_directory())

        # Resolve folder path. Override wins; otherwise dropdown value.
        raw = (folder_override or "").strip() or folder
        if raw == "<none>" or not raw:
            raise ValueError(
                "HYWM2LoadMemoryBank: no folder selected. Place a memorybank "
                "folder under ComfyUI/input/<folder>/ (with cameras.json + "
                "frames/) and restart ComfyUI, or pass `folder_override`."
            )
        target = Path(raw)
        if not target.is_absolute():
            target = input_dir / target
        if not _looks_like_bank(target):
            raise FileNotFoundError(
                f"HYWM2LoadMemoryBank: {target} doesn't look like a "
                f"memorybank (missing cameras.json or frames/)"
            )

        # --- cameras.json -> tensors ----------------------------------
        cameras = json.loads((target / "cameras.json").read_text())
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
                f"cameras.json count mismatch: {ext_np.shape[0]} extrinsics "
                f"vs {K_np.shape[0]} intrinsics"
            )

        # --- frames/*.png -> IMAGE tensor -----------------------------
        # Skip hidden / system dirs like .ipynb_checkpoints inside frames/.
        frames_dir = target / "frames"
        frame_paths = sorted(p for p in frames_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                             and not p.name.startswith("."))
        if len(frame_paths) != N:
            raise ValueError(
                f"frame count mismatch: {len(frame_paths)} PNGs in "
                f"{frames_dir} vs {N} cameras in cameras.json"
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

        _p(f"loaded {target}: N={N}, image_size=({W},{H})")
        return io.NodeOutput(images_t, extrinsics_t, intrinsics_t, N)
