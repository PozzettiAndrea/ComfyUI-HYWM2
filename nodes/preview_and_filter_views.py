"""HYWM2PreviewAndFilterViews -- thumbnail grid with per-view toggles.

Takes an (images, extrinsics, intrinsics) triplet and emits a filtered
subset of them based on per-view boolean toggles set in the UI.

UX:
  - First execution: all N views pass through (default mask = all True).
  - Frontend renders a clickable thumbnail grid showing each view + its
    per-view camera (ext 4x4 + K 3x3 on hover).
  - User clicks thumbnails to toggle each view on/off.
  - Toggles persist via the `enabled_mask` String widget (JSON-encoded
    list of N bools). Re-queue the workflow to send the filtered subset
    downstream.

Companion: web/js/preview_views.js — registers the DOM widget that
replaces the textbox with the thumbnail grid and round-trips the click
state into `enabled_mask`.
"""

from __future__ import annotations

import json
import random
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io


def _p(msg: str) -> None:
    print(f"[HYWM2PreviewAndFilterViews] {msg}", file=sys.stderr, flush=True)


def _rand_id(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


class HYWM2PreviewAndFilterViews(io.ComfyNode):
    """Preview a triplet of (images, extrinsics, intrinsics) with per-view toggles."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2PreviewAndFilterViews",
            display_name="HYWM2 Preview + Filter Views",
            category="HYWM2",
            description=(
                "Interactive thumbnail grid of a multi-view triplet "
                "(images + extrinsics + intrinsics). Click thumbnails to "
                "toggle individual views ON/OFF; the filtered subset flows "
                "downstream.\n\n"
                "First run: all views ON (so you can see what you have). "
                "Subsequent runs use whatever toggle state you set. Re-queue "
                "after editing toggles to push the new subset onwards.\n\n"
                "Output node — can be queued on its own to preview a view "
                "set without any downstream consumers wired."
            ),
            is_output_node=True,
            inputs=[
                io.Image.Input("images",
                    tooltip="[N, H, W, 3] multi-view image batch."),
                io.Custom("EXTRINSICS").Input("extrinsics",
                    tooltip="[N, 4, 4] w2c per view."),
                io.Custom("INTRINSICS").Input("intrinsics",
                    tooltip="[N, 3, 3] K per view."),
                io.String.Input(
                    "enabled_mask", default="", multiline=False,
                    tooltip="JSON list of N booleans (e.g. '[true,true,false,...]'). "
                            "Empty = all enabled (default). The JS widget "
                            "writes here when you click thumbnails. You can "
                            "also edit by hand."),
            ],
            outputs=[
                io.Image.Output(display_name="images",
                    tooltip="Filtered [M, H, W, 3] (M = number of enabled views)."),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics",
                    tooltip="Filtered [M, 4, 4] w2c."),
                io.Custom("INTRINSICS").Output(display_name="intrinsics",
                    tooltip="Filtered [M, 3, 3] K."),
                io.Int.Output(display_name="num_enabled",
                    tooltip="M -- number of views currently enabled."),
            ],
        )

    @classmethod
    def execute(cls, images, extrinsics, intrinsics, enabled_mask: str = ""):
        # ---- Normalize inputs ----------------------------------------
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if isinstance(extrinsics, np.ndarray):
            extrinsics = torch.from_numpy(extrinsics)
        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics)
        images = images.float()
        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()

        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"images shape {tuple(images.shape)} not [N, H, W, 3]")
        N, H, W, _ = images.shape

        if extrinsics.dim() == 3 and extrinsics.shape[0] == 1 and N > 1:
            extrinsics = extrinsics.expand(N, -1, -1)
        if intrinsics.dim() == 3 and intrinsics.shape[0] == 1 and N > 1:
            intrinsics = intrinsics.expand(N, -1, -1)
        if extrinsics.shape != (N, 4, 4):
            raise ValueError(
                f"extrinsics shape {tuple(extrinsics.shape)} not [N={N}, 4, 4]")
        if intrinsics.shape != (N, 3, 3):
            raise ValueError(
                f"intrinsics shape {tuple(intrinsics.shape)} not [N={N}, 3, 3]")

        # ---- Parse enabled_mask --------------------------------------
        mask: list[bool]
        s = (enabled_mask or "").strip()
        if not s:
            mask = [True] * N
        else:
            try:
                parsed = json.loads(s)
                if not isinstance(parsed, list):
                    raise TypeError("not a list")
                mask = [bool(v) for v in parsed]
                if len(mask) != N:
                    _p(f"enabled_mask length {len(mask)} != N={N}; "
                       f"resetting to all-True")
                    mask = [True] * N
            except Exception as e:
                _p(f"enabled_mask parse failed ({type(e).__name__}: {e}); "
                   f"using all-True")
                mask = [True] * N

        # ---- Dump ALL thumbnails to temp/ (so user can see every view) --
        # The JS widget reads /view?filename=...&subfolder=...&type=temp
        # to display.
        try:
            import folder_paths
            temp_root = Path(folder_paths.get_temp_directory())
        except Exception:
            temp_root = Path("/tmp")
        prefix = f"hywm2_views_{_rand_id()}_{int(time.time())}"
        out_dir = temp_root / prefix
        out_dir.mkdir(parents=True, exist_ok=True)

        images_np = (images.detach().cpu().numpy().clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
        ext_list = extrinsics.detach().cpu().numpy().astype(np.float32).tolist()
        K_list = intrinsics.detach().cpu().numpy().astype(np.float32).tolist()

        views_payload = []
        for i in range(N):
            fn = f"{i:04d}.png"
            Image.fromarray(images_np[i]).save(out_dir / fn, compress_level=1)
            views_payload.append({
                "index": int(i),
                "filename": fn,
                "subfolder": prefix,
                "type": "temp",
                "ext": ext_list[i],
                "K": K_list[i],
                "enabled": bool(mask[i]),
            })
        # Diagnostic: confirm exactly what the frontend will request.
        # The browser will hit /view?filename=...&subfolder=...&type=temp
        # which ComfyUI resolves to <temp_root>/<subfolder>/<filename>.
        _p(f"wrote {N} thumbnails -> {out_dir}  (subfolder='{prefix}', type='temp')")
        if N > 0:
            sample = out_dir / views_payload[0]["filename"]
            _p(f"sample on disk: {sample}  exists={sample.is_file()}  "
               f"size={sample.stat().st_size if sample.is_file() else 0}B")

        # ---- Filter the outputs --------------------------------------
        idx = [i for i, on in enumerate(mask) if on]
        if not idx:
            # Avoid empty-tensor IPC issues: keep at least one row but log it.
            _p("WARNING: zero views enabled; emitting first view to keep "
               "tensors non-empty downstream. Re-enable some toggles.")
            idx = [0]
        idx_t = torch.tensor(idx, dtype=torch.long)
        out_imgs = images.index_select(0, idx_t)
        out_ext = extrinsics.index_select(0, idx_t).contiguous()
        out_K = intrinsics.index_select(0, idx_t).contiguous()
        M = len(idx)

        _p(f"N={N}, enabled={M}, mask={mask}")

        ui_payload = {
            "count": N,
            "enabled_mask": mask,
            "image_size": [int(W), int(H)],
            "views": views_payload,
        }

        return io.NodeOutput(
            out_imgs, out_ext, out_K, M,
            ui={"preview_views": [json.dumps(ui_payload)]},
        )
