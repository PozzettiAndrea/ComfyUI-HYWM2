"""HYWM2GaussianTrain — wrap HY-World 2.0's world_gs_trainer for in-ComfyUI 3DGS training.

Takes HYWM2Reconstruct's gaussian dict + the same set of training views, materializes
the gs_data/ layout the upstream trainer expects, and runs the trainer in-process via
`main(0, 0, 1, cfg)`. The full HYWM2 splat state (means/quats/scales/opacities/sh0)
seeds the optimizer via `cfg.preload_gs_path`; the canonical points.ply also gets
written so the Parser stays happy at init_type="sfm".

This is the in-process equivalent of the shell wrapper at /home/work/training.sh.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io


# SH band-0 conversion factor. rgbs = (sh0 · C0 + 0.5).clamp(0, 1)  (gs/utils.rgb_to_sh).
_C0 = 1.0 / (2.0 * math.sqrt(math.pi))  # ≈ 0.28209479


def _p(msg: str) -> None:
    print(f"[HYWM2GaussianTrain] {msg}", file=sys.stderr, flush=True)


def _request_vram_eviction(needed_bytes: int) -> None:
    """Ask comfy-env's parent ComfyUI process to evict cross-worker models so
    this transient-tensor node has VRAM headroom. Mirrors the helper added to
    PanoramaDepthMerge — see depth_merge.py:_request_vram_eviction for the
    full rationale. Cleanly no-ops outside a comfy-env worker subprocess.
    """
    try:
        import comfy_worker  # noqa: F401 - injected at worker startup
        try:
            comfy_worker.call_parent(
                "request_vram_budget", total_size=int(needed_bytes)
            )
            _p(f"  -> requested {needed_bytes / 1e9:.2f} GB eviction via comfy_worker.call_parent")
        except Exception as e:
            _p(f"  -> comfy_worker.call_parent failed: {e}")
    except ImportError:
        _p("  -> comfy_worker module unavailable; local free_memory only")

    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        mm.free_memory(int(needed_bytes), device)
    except Exception as e:
        _p(f"  -> local mm.free_memory failed: {e}")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _mem_snapshot() -> str:
    """Compact 'RAM / swap / VRAM' snapshot string — same telemetry shape
    WorldStereoGenerate prints around offload_everything (inference.py:430-454).
    """
    parts: list[str] = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        parts.append(f"{vm.used / 1024**3:.1f}/{vm.total / 1024**3:.1f}GB ram")
        parts.append(f"{sm.used / 1024**3:.1f}/{sm.total / 1024**3:.1f}GB swap")
    except Exception as _e:
        parts.append(f"ram=? ({type(_e).__name__})")
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024**3
            parts.append(f"{used:.1f}/{total / 1024**3:.1f}GB vram")
        except Exception as _e:
            parts.append(f"vram=? ({type(_e).__name__})")
    return ", ".join(parts)


def _offload_everything() -> None:
    """Aggressive eviction modeled on WorldStereoGenerate.offload_everything
    (inference.py:421-470). Two-step:

      1. Ask the parent ComfyUI process to evict EVERY model patcher across
         sibling worker subprocesses (cross-worker dump via the IPC bridge).
         Total budget = full GPU memory so the parent's _handle_vram_budget
         treats it as 'evict whatever you can find'.
      2. In-worker: unload every patcher this worker's `current_loaded_models`
         knows about (HYWM2Reconstruct lives in the SAME hywm2-nodes worker as
         us, so its DiT patcher is here too). Then soft_empty_cache + gc +
         empty_cache for full release.

    HYWM2's DiT will auto-reload on its next Reconstruct call. That's the
    intended trade-off vs OOMing the trainer.
    """
    _p(f"offload_everything: before --> {_mem_snapshot()}")

    # Cross-worker eviction (best-effort; no-op outside a comfy-env worker).
    try:
        import comfy_worker  # noqa: F401 - injected by comfy-env at worker startup
        try:
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory
            else:
                total_vram = 0
            comfy_worker.call_parent(
                "request_vram_budget", total_size=int(total_vram),
            )
            _p(f"  -> cross-worker eviction requested ({total_vram / 1e9:.1f} GB)")
        except Exception as e:
            _p(f"  -> comfy_worker.call_parent failed: {type(e).__name__}: {e}")
    except ImportError:
        _p("  -> comfy_worker module unavailable; in-worker only")

    # In-worker: unload everything this worker manages.
    try:
        import comfy.model_management as mm
        try:
            mm.unload_all_models()
            _p("  -> mm.unload_all_models() done")
        except Exception as e:
            _p(f"  -> mm.unload_all_models() raised {type(e).__name__}: {e}")
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    except Exception as e:
        _p(f"  -> comfy.model_management unavailable: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _p(f"offload_everything: after  --> {_mem_snapshot()}")


def _install_tqdm_throttle(interval_seconds: float = 5.0) -> callable:
    """Monkey-patch `tqdm.tqdm.update` to emit a newline-terminated progress
    line every `interval_seconds`. Worker stderr is line-buffered, so tqdm's
    default `\\r`-based redraw stays invisible during a 5000-step training
    run that takes 10-30 min. Periodic newline-terminated lines surface real
    progress in the worker log.

    Returns the restore function to call in a `finally` block.
    """
    import time
    try:
        import tqdm as _tqdm_mod
    except ImportError:
        _p("tqdm not importable; progress prints disabled")
        return lambda: None

    original_update = _tqdm_mod.tqdm.update
    # Per-instance last-print timestamps (id(self) -> monotonic time).
    last_print: dict[int, float] = {}

    def _emit(self) -> None:
        try:
            n = int(self.n or 0)
            total = int(self.total or 0)
            pct = (100.0 * n / total) if total else 0.0
            # tqdm.format_dict carries elapsed seconds + smoothed rate.
            d = self.format_dict if hasattr(self, "format_dict") else {}
            elapsed = float(d.get("elapsed", 0.0) or 0.0)
            rate = (n / elapsed) if elapsed > 0 else 0.0
            postfix = self.postfix if hasattr(self, "postfix") and self.postfix else ""
            line = (
                f"[HYWM2GaussianTrain]   step {n}/{total} ({pct:.1f}%) "
                f"— {rate:.1f} it/s — elapsed {elapsed:.1f}s"
            )
            if postfix:
                line += f" — {postfix}"
            print(line, file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[HYWM2GaussianTrain] tqdm throttle emit failed: {e}",
                  file=sys.stderr, flush=True)

    def _patched_update(self, n: int = 1) -> object:
        ret = original_update(self, n)
        try:
            now = time.monotonic()
            key = id(self)
            last = last_print.get(key, 0.0)
            if now - last >= interval_seconds:
                last_print[key] = now
                _emit(self)
        except Exception:
            pass
        return ret

    _tqdm_mod.tqdm.update = _patched_update
    _p(f"tqdm throttle installed (every {interval_seconds:.0f}s)")

    def _restore() -> None:
        try:
            _tqdm_mod.tqdm.update = original_update
            _p("tqdm throttle restored")
        except Exception:
            pass

    return _restore


def _rescale_extrinsics_for_scene_scale(
    extrinsics: torch.Tensor, gaussian_means: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Defend against trainer's `Scene scale: 0.0` collapse.

    HYWM2Reconstruct's `predicted_extrinsics` are normalized — the first
    view is at identity (camera_0 at origin) and the rest sit close to it
    in HYWM2's internal frame. The trainer's Parser computes
    `scene_scale = max(camera-camera-distances)` (gs/opencv.py:987). When
    all cameras are ~coincident that max is ~0, and downstream
    `means_lr * scene_scale = 0` means positions never update during
    training. Same factor multiplies the means-bound box for densification,
    so we get no growth either.

    Fix: rescale BOTH extrinsics translations AND gaussian means by
    1 / median_camera_distance so cameras span ~unit scale. The two have
    to scale together — moving cameras without moving gaussians would
    change the relative geometry the splat encodes.

    Returns (extrinsics_scaled, gaussian_means_scaled, scale_factor).
    scale_factor is what we multiplied translations + means by; 1.0 if
    no rescale was needed.
    """
    ext = extrinsics.detach().cpu().float().clone()
    if ext.dim() == 4 and ext.shape[0] == 1:
        ext = ext[0]
    # Camera centers in world = -R^T @ t where w2c = [R | t].
    R = ext[:, :3, :3]                              # [N, 3, 3]
    t = ext[:, :3, 3:4]                             # [N, 3, 1]
    centers = -torch.bmm(R.transpose(1, 2), t).squeeze(-1)  # [N, 3]
    N = centers.shape[0]
    if N < 2:
        return extrinsics, gaussian_means, 1.0
    # Pairwise distances (just the upper triangle is enough; this is N=12-ish).
    cross = centers.unsqueeze(0) - centers.unsqueeze(1)     # [N, N, 3]
    dists = cross.norm(dim=-1)                              # [N, N]
    triu_idx = torch.triu_indices(N, N, offset=1)
    pair_dists = dists[triu_idx[0], triu_idx[1]]            # [N*(N-1)/2]
    median_dist = float(pair_dists.median().item())
    max_dist = float(pair_dists.max().item())

    print(
        f"[HYWM2GaussianTrain] camera-camera distance stats (raw): "
        f"median={median_dist:.6f} max={max_dist:.6f}",
        file=sys.stderr, flush=True,
    )

    # If cameras span O(1) already, leave alone (avoid double-scaling on
    # a real WorldStereo scene). Threshold at 0.1 = cameras within 10cm,
    # which is the HYWM2-normalized regime.
    if max_dist >= 0.1:
        return extrinsics, gaussian_means, 1.0

    # Target: median camera-camera distance = 1.0 in world units. Compute
    # the scale factor as 1 / median (or 1 / max if median is degenerate).
    pivot = max(median_dist, max_dist, 1e-9)
    scale = 1.0 / pivot

    # Scale translations of every w2c matrix. The trick: w2c moves world ->
    # camera. If we scale world coords by `s`, the camera-frame coords stay
    # the same iff we scale the translation by `s` too. So we replace
    # t_new = t * s and leave R alone. Camera-camera distances become
    # `s * dist`, matching what the means transform does.
    ext_scaled = ext.clone()
    ext_scaled[:, :3, 3] = ext[:, :3, 3] * scale
    means_scaled = gaussian_means.detach().cpu().float() * scale

    # Restore the input shape (preserve the singleton batch dim if the
    # caller passed [1, N, 4, 4]).
    if extrinsics.dim() == 4 and extrinsics.shape[0] == 1:
        ext_scaled = ext_scaled.unsqueeze(0)

    print(
        f"[HYWM2GaussianTrain] rescaling extrinsics + means by {scale:.4f} "
        f"(so median camera distance -> 1.0). Output PLY is in scaled "
        f"world units; multiply by {1/scale:.4f} to recover original "
        f"HYWM2 metric scale.",
        file=sys.stderr, flush=True,
    )
    return ext_scaled, means_scaled, scale


def _normalize_K_to_pixel(intr: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Sniff PanoPack-style normalized K (fx<2) and rescale to pixel-K for the
    given image (W, H). Same sniff used in Sharp's predict nodes
    (predict_metric_depth.py:178-204). Pure pass-through if already pixel-K.
    """
    intr = intr.detach().float().clone()
    sample_fx = float(intr[0, 0, 0] if intr.dim() == 3 else intr[0, 0])
    if sample_fx < 2.0:
        if intr.dim() == 3:
            intr[:, 0, :] *= float(W)
            intr[:, 1, :] *= float(H)
        else:
            intr[0, :] *= float(W)
            intr[1, :] *= float(H)
        sample_fx_after = float(intr[0, 0, 0] if intr.dim() == 3 else intr[0, 0])
        _p(f"detected normalized intrinsics (fx<2); rescaled to pixel-K for "
           f"{W}×{H}: fx={sample_fx_after:.1f}")
    return intr


def _dump_image_batch(images: torch.Tensor, out_dir: Path, keys: list[str]) -> None:
    """Save [N, H, W, 3] float[0,1] IMAGE batch to PNGs named <key>.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np_imgs = images.detach().cpu().clamp(0, 1).numpy()
    for i, frame in enumerate(np_imgs):
        arr = (frame * 255.0 + 0.5).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        Image.fromarray(arr).save(out_dir / f"{keys[i]}.png")


def _dump_depths(depths: torch.Tensor, out_dir: Path, keys: list[str]) -> float:
    """Save per-frame metric depth as 16-bit grayscale PNGs. Returns the scale
    factor used (so the trainer-side decode can match: depth_meters = png / scale).

    depths: [N, H, W] or [N, H, W, 1|3] float meters.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    d = depths.detach().cpu().float()
    if d.dim() == 4 and d.shape[-1] in (1, 3, 4):
        d = d[..., 0]  # collapse channel — depth scalar replicated to RGB
    # Use a per-batch scale so values map nicely into uint16's [0, 65535] range.
    max_d = float(d.max().item())
    if max_d <= 1e-6:
        max_d = 1.0
    scale = 65535.0 / max_d
    arr = (d.numpy() * scale).clip(0, 65535).astype(np.uint16)
    for i, frame in enumerate(arr):
        Image.fromarray(frame, mode="I;16").save(out_dir / f"{keys[i]}.png")
    return scale


def _dump_normals(normals: torch.Tensor, out_dir: Path, keys: list[str]) -> None:
    """Save per-frame normals as 8-bit RGB PNGs. Input shape [N, H, W, 3]
    expected in [-1, 1] (so (n+1)/2 = [0, 1] = RGB). HYWM2Reconstruct.normals
    already emits in viz form (n+1)/2 ∈ [0, 1] per `decode_normals_image` —
    so we just need to cast.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_np = normals.detach().cpu().clamp(0, 1).float().numpy()
    arr = (n_np * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    for i, frame in enumerate(arr):
        Image.fromarray(frame).save(out_dir / f"{keys[i]}.png")


def _write_cameras_json(
    out_path: Path,
    extrinsics: torch.Tensor,
    intrinsics_pixel: torch.Tensor,
    keys: list[str],
) -> None:
    """Write cameras.json in the trainer's dict form:
        {"<key>.png": {"extrinsic": [4x4 w2c], "intrinsic": [3x3 pixel-K]}}
    """
    ext = extrinsics.detach().cpu().float().numpy()
    intr = intrinsics_pixel.detach().cpu().float().numpy()
    if ext.ndim == 4 and ext.shape[0] == 1:
        ext = ext[0]
    if intr.ndim == 4 and intr.shape[0] == 1:
        intr = intr[0]
    # Allow 3-row extrinsics (pad bottom row).
    if ext.shape[-2:] == (3, 4):
        pad = np.tile(np.array([0, 0, 0, 1], dtype=ext.dtype), (ext.shape[0], 1, 1))
        ext = np.concatenate([ext, pad], axis=1)
    out: dict = {}
    for i, key in enumerate(keys):
        # Parser keys cameras.json by the image's filename WITHOUT extension
        # in some paths but WITH extension in others (line 588 of opencv.py
        # reads `cam_info[camera_id]` where camera_id is the bare image name).
        # Write both forms to be safe — the dict load picks whichever it
        # finds first.
        entry = {
            "extrinsic": ext[i].tolist(),
            "intrinsic": intr[i].tolist(),
        }
        out[key] = entry
        out[f"{key}.png"] = entry
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))


def _write_points_ply(
    out_path: Path, means: torch.Tensor, rgbs: torch.Tensor,
) -> None:
    """Colored point cloud for the trainer's points.ply seed.

    means: [N, 3] float meters in world space.
    rgbs:  [N, 3] float in [0, 1] linear.

    The trainer reads via trimesh.load(...) at gs/opencv.py:717 — trimesh
    happily takes a `vertex` element with `x/y/z` floats and `red/green/blue`
    uint8 channels.
    """
    from plyfile import PlyData, PlyElement

    m = means.detach().cpu().float().numpy()
    r = (rgbs.detach().cpu().clamp(0, 1).float().numpy() * 255.0 + 0.5).astype(np.uint8)
    N = m.shape[0]
    if r.shape[0] != N:
        raise ValueError(f"_write_points_ply: means N={N} != rgbs N={r.shape[0]}")
    arr = np.empty(N, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr["x"] = m[:, 0]
    arr["y"] = m[:, 1]
    arr["z"] = m[:, 2]
    arr["red"] = r[:, 0]
    arr["green"] = r[:, 1]
    arr["blue"] = r[:, 2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out_path))


def _write_preload_pt(
    out_path: Path,
    gaussians: dict,
    max_gaussians: int = 0,
    world_scale: float = 1.0,
) -> None:
    """Save HYWM2's full splat state as the trainer's preload_gs_path .pt.

    Schema the trainer expects (world_gs_trainer.py:442-451):
        {"splats": {"means3d", "quats", "scales", "opacities", "sh0", "shN"}}
    where sh0 is [N, 1, 3] and shN is [N, K, 3] (K depends on sh_degree).

    Conversion from HYWM2_GAUSSIANS:
        rgbs ∈ [0, 1] -> sh0 = (rgbs - 0.5) / C0      (inverse of rgb_to_sh)
        shN -> zeros at shape [N, 0, 3]               (sh_degree=0 default)

    `max_gaussians > 0` caps the preload via random subsample. Defends
    against OOM in gsplat's isect_tiles when HYWM2's prediction has
    millions of gaussians (its tile-intersection workspace scales as
    O(N_gaussians × max_overlapping_tiles), which blows up on a 24 GB
    card at N > ~1M).

    `world_scale != 1.0` rescales both means (linear) and scales (additive
    in log space: log(s · world_scale) = log(s) + log(world_scale)). The
    trainer's scales are stored in log-space (`world_gs_trainer.py:408`,
    `scales = torch.log(...)`); we have to match that representation.
    """
    means = gaussians["means"].detach().float().cpu()
    quats = gaussians["quats"].detach().float().cpu()
    scales = gaussians["scales"].detach().float().cpu()
    opacities = gaussians["opacities"].detach().float().cpu()
    rgbs = gaussians["rgbs"].detach().float().cpu().clamp(0, 1)

    # World rescale (mirrors _rescale_extrinsics_for_scene_scale on the
    # gaussian side). means scale linearly; log-space scales pick up a
    # log(world_scale) offset.
    if world_scale != 1.0:
        means = means * float(world_scale)
        scales = scales + math.log(float(world_scale))

    N = means.shape[0]

    # Optional random subsample. Deterministic per-N via a seeded generator
    # so reruns are reproducible.
    if 0 < max_gaussians < N:
        g = torch.Generator(device="cpu").manual_seed(N)
        keep_idx = torch.randperm(N, generator=g)[:max_gaussians]
        means = means[keep_idx]
        quats = quats[keep_idx]
        scales = scales[keep_idx]
        opacities = opacities[keep_idx]
        rgbs = rgbs[keep_idx]
        N = max_gaussians
        # Log via print directly since module-level _p isn't in scope here.
        print(
            f"[HYWM2GaussianTrain] preload subsampled {gaussians['means'].shape[0]} "
            f"-> {N} gaussians (cap)",
            file=sys.stderr, flush=True,
        )

    sh0 = ((rgbs - 0.5) / _C0).unsqueeze(1)  # [N, 1, 3]
    shN = torch.zeros((N, 0, 3), dtype=torch.float32)

    payload = {
        "step": 0,
        "splats": {
            # Position key is "means3d" not "means" — the trainer renames
            # it at world_gs_trainer.py:444. Other keys match HYWM2_GAUSSIANS.
            "means3d": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": sh0,
            "shN": shN,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))


class HYWM2GaussianTrain(io.ComfyNode):
    """Wrap HY-World 2.0's world_gs_trainer for in-pipeline 3DGS training."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2GaussianTrain",
            display_name="HYWM2 Train Gaussians",
            category="HYWM2",
            description=(
                "Run HY-World 2.0's world_gs_trainer (the same trainer driven "
                "by training.sh) directly from a ComfyUI workflow. Materializes "
                "the gs_data/ layout from the wired gaussians + images + "
                "extrinsics + intrinsics, seeds the optimizer with HYWM2's "
                "full splat state via --preload-gs-path, and returns the path "
                "to the final trained PLY."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("HYWM2_GAUSSIANS").Input(
                    "gaussians",
                    tooltip="HYWM2_GAUSSIANS dict (means/quats/scales/opacities/"
                            "rgbs). Wire from HYWM2Reconstruct.gaussians. The "
                            "full splat state is saved as preload_gs.pt so the "
                            "trainer hot-starts from it; means+rgbs are also "
                            "written as points.ply to satisfy the Parser."),
                io.Image.Input(
                    "images",
                    tooltip="Training views [N, H, W, 3] float[0,1]. Should be "
                            "the SAME N views you ran HYWM2Reconstruct on."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics",
                    tooltip="Per-view world-to-camera [N, 4, 4]. Use "
                            "HYWM2Reconstruct.predicted_extrinsics for "
                            "consistency with the gaussians."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics",
                    tooltip="Per-view K [N, 3, 3]. Accepts pixel-K (Sharp/HYWM2 "
                            "convention) or normalized-K (PanoPack convention) — "
                            "auto-rescaled to pixel-K for the image (W, H) at "
                            "execute time."),
                io.Int.Input(
                    "max_steps", default=5000, min=100, max=50000,
                    tooltip="Number of training iterations. Default 5000 "
                            "matches training.sh's default and gives "
                            "publication-quality splats in ~10-30 min on a "
                            "3090. For quick smoke tests try 200-500."),
                io.Combo.Input(
                    "preset",
                    options=["default", "mcmc", "prune_only"],
                    default="default",
                    tooltip="Which of the trainer's three preset configs to "
                            "clone (matches the 'default'/'mcmc'/'prune_only' "
                            "args of world_gs_trainer.py). 'default' is the "
                            "standard densification heuristic; 'mcmc' uses MCMC "
                            "from the gsplat-as-MCMC paper; 'prune_only' "
                            "disables growth and only prunes."),
                io.Boolean.Input(
                    "save_ply", default=True,
                    tooltip="Write .ply checkpoints at the trainer's ply_steps "
                            "milestones + at max_steps - 1. Required True if "
                            "you want a final PLY out."),
                io.Int.Input(
                    "preload_max_gaussians", default=0,
                    min=0, max=20_000_000,
                    tooltip="OPTIONAL safety cap on the preload gaussian count "
                            "(random subsample). 0 = disabled (default; pass "
                            "every input gaussian through as the training "
                            "warm-start). Set a positive value (e.g. 500_000) "
                            "if a HYWM2Reconstruct run produced more gaussians "
                            "than fit in VRAM and you want a quick "
                            "node-side downsample without re-running the "
                            "predictor. Primary control over count is "
                            "HYWM2Reconstruct's own filters "
                            "(gaussians_downsample / gaussians_voxel_size / "
                            "gaussians_conf_percentile) — shape upstream "
                            "where possible."),
                io.String.Input(
                    "output_prefix", default="hywm2_train", optional=True,
                    tooltip="Subdir prefix under ComfyUI's output dir. Final "
                            "results land at <output>/<prefix>_<ts>/gs_data/"
                            "gs_result/ply/point_cloud_<step>.ply."),
                io.String.Input(
                    "hyworld2_repo_path",
                    default="/home/work/HY-World-2.0", optional=True,
                    tooltip="Root of the HY-World 2.0 source tree. Prepended "
                            "to sys.path so `import hyworld2.worldgen."
                            "world_gs_trainer` resolves. Override if you moved "
                            "the repo. Leave blank to use the default."),
                io.Boolean.Input(
                    "offload_everything", default=True, optional=True,
                    tooltip=(
                        "Aggressive VRAM eviction before training (mirrors "
                        "WorldStereoGenerate). When True, calls "
                        "comfy_worker.call_parent('request_vram_budget', "
                        "total_size=<full GPU>) to evict sibling-worker "
                        "patchers, then mm.unload_all_models() to dump every "
                        "patcher this worker registered (e.g. "
                        "HYWM2Reconstruct's DiT — which lives in the SAME "
                        "hywm2-nodes worker as us). HYWM2's DiT will "
                        "auto-reload on its next Reconstruct call (~6-10s "
                        "tax). When False, falls back to bounded "
                        "_request_vram_eviction(~18 GB) only. Recommended "
                        "True; only flip off if you know nothing else is "
                        "resident and want to skip the reload."
                    )),
                io.Image.Input(
                    "depth_raw", optional=True,
                    tooltip="Optional per-view metric depth [N, H, W, 3] from "
                            "HYWM2Reconstruct.depth_raw. If wired, written to "
                            "depths/ AND cfg.depth_loss=True so the trainer's "
                            "depth supervision loss activates. Improves "
                            "geometry on under-textured regions."),
                io.Image.Input(
                    "normals", optional=True,
                    tooltip="Optional per-view surface normals [N, H, W, 3] in "
                            "viz form ((n+1)/2 ∈ [0,1]) from "
                            "HYWM2Reconstruct.normals. If wired, written to "
                            "normals/ AND cfg.normal_loss=True so the trainer's "
                            "normal-consistency loss activates."),
            ],
            outputs=[
                io.String.Output(
                    display_name="ply_path",
                    tooltip="Absolute path to the highest-step "
                            "point_cloud_<step>.ply in the trainer's "
                            "result_dir/ply/. Drop-in for "
                            "MergeGaussians / GeomPackPreviewGaussian / "
                            "external viewers."),
                io.String.Output(
                    display_name="result_dir",
                    tooltip="Absolute path to gs_result/ (contains ply/, "
                            "ckpts/, renders/, tb/, videos/)."),
            ],
        )

    @classmethod
    def execute(
        cls,
        gaussians: dict,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        max_steps: int = 5000,
        preset: str = "default",
        save_ply: bool = True,
        preload_max_gaussians: int = 0,
        output_prefix: str = "hywm2_train",
        hyworld2_repo_path: str = "/home/work/HY-World-2.0",
        offload_everything: bool = True,
        depth_raw: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
    ):
        # ----- Validate inputs -----
        if not isinstance(gaussians, dict) or "means" not in gaussians:
            raise ValueError("HYWM2GaussianTrain: gaussians must be an HYWM2_GAUSSIANS dict with a 'means' key.")
        means_t = gaussians["means"]
        if means_t.dim() != 2 or means_t.shape[0] < 1 or means_t.shape[1] != 3:
            raise ValueError(
                f"HYWM2GaussianTrain: gaussians['means'] must be [N>=1, 3]; got {tuple(means_t.shape)}"
            )

        if images.dim() == 3:
            images = images.unsqueeze(0)
        N_img = int(images.shape[0])
        if N_img == 0:
            raise ValueError("HYWM2GaussianTrain: empty images batch")
        H_img = int(images.shape[1])
        W_img = int(images.shape[2])

        ext = extrinsics
        if ext.dim() == 4 and ext.shape[0] == 1:
            ext = ext[0]
        if ext.dim() != 3 or ext.shape[0] != N_img or ext.shape[-2:] not in ((4, 4), (3, 4)):
            raise ValueError(
                f"HYWM2GaussianTrain: extrinsics shape {tuple(extrinsics.shape)} "
                f"doesn't match N={N_img} images. Expected [N, 4, 4]."
            )

        intr = intrinsics
        if intr.dim() == 4 and intr.shape[0] == 1:
            intr = intr[0]
        if intr.dim() == 2:
            intr = intr.unsqueeze(0).expand(N_img, 3, 3).contiguous()
        if intr.dim() != 3 or intr.shape[0] != N_img or intr.shape[-2:] != (3, 3):
            raise ValueError(
                f"HYWM2GaussianTrain: intrinsics shape {tuple(intrinsics.shape)} "
                f"doesn't match N={N_img} images. Expected [N, 3, 3]."
            )

        # Convert normalized K -> pixel K once, at the image's native (W, H).
        intr_pixel = _normalize_K_to_pixel(intr, W_img, H_img)

        N_gauss = int(means_t.shape[0])
        _p(f"inputs OK: N_views={N_img} @ {W_img}×{H_img}, N_gauss={N_gauss}, "
           f"max_steps={max_steps}, preset={preset}, depth_loss={depth_raw is not None}, "
           f"normal_loss={normals is not None}")

        # ----- Allocate working dirs under ComfyUI's output dir -----
        try:
            import folder_paths
            output_root = Path(folder_paths.get_output_directory())
        except Exception:
            output_root = Path(tempfile.gettempdir()) / "hywm2_train_output"
        ts_ms = int(time.time() * 1000)
        work_dir = output_root / f"{output_prefix}_{ts_ms}"
        gs_data = work_dir / "gs_data"
        gs_result = gs_data / "gs_result"
        gs_data.mkdir(parents=True, exist_ok=True)
        gs_result.mkdir(parents=True, exist_ok=True)
        _p(f"gs_data: {gs_data}")
        _p(f"gs_result: {gs_result}")

        # ----- Materialize the trainer's gs_data layout -----
        keys = [f"frame_{i:04d}" for i in range(N_img)]

        _p(f"writing {N_img} images to images/...")
        _dump_image_batch(images, gs_data / "images", keys)

        # Defend against the trainer's Scene scale = 0 collapse. HYWM2's
        # predicted_extrinsics are normalized (first view at identity, rest
        # nearby) so the trainer's max-camera-distance-based scene_scale
        # would be ~0, which zeros out the means learning rate. Rescale
        # extrinsics translations AND gaussian means together so cameras
        # span ~unit world distance. No-op for already-spread cameras
        # (e.g. real WorldStereo exports).
        ext, gaussian_means_scaled, world_scale = _rescale_extrinsics_for_scene_scale(
            ext, means_t,
        )

        _p("writing cameras.json...")
        _write_cameras_json(gs_data / "cameras.json", ext, intr_pixel, keys)

        # points.ply is now a 1-vertex placeholder — the trainer's Parser
        # requires the file to exist (gs/opencv.py:712-722, trimesh.load on
        # missing file would raise), but with cfg.init_type='random' and
        # init_num_pts=1 below, the sfm/random branches both produce just
        # ONE garbage gaussian that gets pruned out within densification's
        # first cycle. The REAL starting state is the HYWM2 preload.
        _p("writing points.ply (1-vertex placeholder, init_type=random)")
        _write_points_ply(
            gs_data / "points.ply",
            gaussian_means_scaled[:1].detach().cpu(),
            (gaussians.get("rgbs") if gaussians.get("rgbs") is not None
             else torch.full((N_gauss, 3), 0.5))[:1].detach().cpu(),
        )

        _p("writing preload_gs.pt (HYWM2 splat hot-start)...")
        preload_path = gs_data / "preload_gs.pt"
        # Pass world_scale through so _write_preload_pt scales BOTH means
        # AND log-scales: log(s * scale) = log(s) + log(scale). gaussian
        # scales encode physical extent in world units, so a world rescale
        # has to track both.
        _write_preload_pt(
            preload_path, gaussians,
            max_gaussians=int(preload_max_gaussians),
            world_scale=float(world_scale),
        )

        depth_loss_on = depth_raw is not None
        normal_loss_on = normals is not None
        if depth_loss_on:
            _p(f"writing {N_img} depths to depths/ (16-bit grayscale)...")
            _dump_depths(depth_raw, gs_data / "depths", keys)
        if normal_loss_on:
            _p(f"writing {N_img} normals to normals/ (8-bit RGB)...")
            _dump_normals(normals, gs_data / "normals", keys)

        # ----- VRAM eviction -----
        # Branch on offload_everything: aggressive (default) vs bounded.
        if offload_everything:
            _p("offload_everything=True -> aggressive eviction")
            _offload_everything()
        else:
            # Bounded path: only ~18 GB requested via free_memory(N, device).
            # Use when caller knows nothing else is resident and wants to
            # skip the DiT reload tax. Training peak scales with point count
            # + image res — 18 GB is a sane conservative estimate on a 24 GB
            # card.
            peak_estimate = int(18 * 1024**3)
            _p(f"offload_everything=False -> bounded eviction "
               f"(estimate: {peak_estimate / 1e9:.1f} GB)")
            _request_vram_eviction(peak_estimate)

        # ----- Import the trainer in-process -----
        # Prepend the user-provided repo path to sys.path. Default to
        # /home/work/HY-World-2.0 per the plan.
        repo_path = (hyworld2_repo_path or "/home/work/HY-World-2.0").strip()
        if not repo_path:
            repo_path = "/home/work/HY-World-2.0"
        if not Path(repo_path).is_dir():
            raise FileNotFoundError(
                f"HYWM2GaussianTrain: hyworld2_repo_path doesn't exist: {repo_path!r}. "
                f"Set the input to the root of your HY-World-2.0 checkout."
            )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # The trainer also imports `gs.*` and `nerfview` / `viser` etc.
        # `cd` into worldgen dir so its relative imports resolve consistently.
        worldgen_dir = Path(repo_path) / "hyworld2" / "worldgen"
        if not worldgen_dir.is_dir():
            raise FileNotFoundError(
                f"HYWM2GaussianTrain: expected {worldgen_dir} to exist under {repo_path}."
            )
        if str(worldgen_dir) not in sys.path:
            sys.path.insert(0, str(worldgen_dir))

        try:
            from hyworld2.worldgen.world_gs_trainer import Config, main
            from gsplat.strategy import DefaultStrategy, MCMCStrategy
        except ImportError as e:
            raise ImportError(
                f"HYWM2GaussianTrain: failed to import the trainer from "
                f"{repo_path}: {e}. Check hyworld2_repo_path."
            ) from e

        # ----- Build the trainer's Config -----
        if preset == "mcmc":
            cfg = Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            )
        elif preset == "prune_only":
            cfg = Config(
                strategy=DefaultStrategy(
                    verbose=True,
                    prune_opa=0.005,
                    grow_grad2d=9999,
                    grow_scale3d=9999,
                    grow_scale2d=9999,
                    prune_scale3d=0.1,
                    prune_scale2d=0.15,
                ),
            )
        else:
            cfg = Config(strategy=DefaultStrategy(verbose=True))

        cfg.data_dir = str(gs_data)
        cfg.result_dir = str(gs_result)
        cfg.max_steps = int(max_steps)
        cfg.save_ply = bool(save_ply)
        cfg.disable_viewer = True
        cfg.preload_gs_path = str(preload_path)
        cfg.depth_loss = bool(depth_loss_on)
        cfg.normal_loss = bool(normal_loss_on)

        # Architecture: HYWM2_GAUSSIANS is treated as a half-trained CHECKPOINT
        # the trainer resumes from. The trainer always builds an initial
        # params block (sfm or random) BEFORE concatenating preload_gs_path
        # onto it (world_gs_trainer.py:386-451). There's no "skip init" path
        # — we have to pick some init_num_pts. We minimize to the smallest
        # number that satisfies the trainer's `knn(points, k=4)` at line 406
        # which derives initial scales from the 4 nearest neighbors of each
        # point. sklearn needs >= k+1 = 5 points; bump to 100 for a comfy
        # margin. Densification prunes these 100 random gaussians within
        # the first prune cycle (~step 100), so by step 200 the effective
        # training state is HYWM2's full preload.
        cfg.init_type = "random"
        cfg.init_num_pts = 100

        # Step list rescale (matches __main__ behavior at world_gs_trainer.py:2601).
        try:
            cfg.adjust_steps(cfg.steps_scaler)
        except Exception as e:
            _p(f"cfg.adjust_steps warning: {e}")

        _p(f"starting trainer: data_dir={cfg.data_dir}, "
           f"result_dir={cfg.result_dir}, max_steps={cfg.max_steps}, "
           f"save_ply={cfg.save_ply}, depth_loss={cfg.depth_loss}, "
           f"normal_loss={cfg.normal_loss}, init_type=random+init_num_pts=1, "
           f"preload_gs_path={cfg.preload_gs_path}")

        # ----- Run the trainer -----
        # Install tqdm throttle so we get newline-terminated progress lines
        # every 5s in the worker log (tqdm's default \\r updates are invisible
        # through line-buffered worker stderr).
        _restore_tqdm = _install_tqdm_throttle(interval_seconds=5.0)
        t0 = time.time()
        try:
            # ComfyUI's executor wraps the whole node-execution chain in
            # `torch.inference_mode()` (execution.py:736). That's STRICTER
            # than torch.no_grad -- tensors created inside can never have
            # requires_grad=True, no matter what nn.Parameter does. The
            # gaussian trainer needs gradient tracking on its splats
            # (gsplat/strategy/default.py:150 calls .retain_grad() on
            # info["means2d"], which crashes if the tensor was made in
            # inference_mode). Escape via the documented opt-out:
            # `torch.inference_mode(False)` toggles the per-thread flag,
            # and `torch.enable_grad()` makes sure no_grad isn't set
            # either. Single-GPU: world_size=1, local_rank=0, world_rank=0.
            with torch.inference_mode(False), torch.enable_grad():
                main(0, 0, 1, cfg)
        finally:
            _restore_tqdm()
        elapsed = time.time() - t0
        _p(f"training done in {elapsed:.1f}s")

        # Best-effort cleanup of the trainer's working tensors before we
        # return; the worker subprocess otherwise holds them until the next
        # node runs.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- Resolve the final PLY -----
        ply_dir = gs_result / "ply"
        if not ply_dir.is_dir():
            raise RuntimeError(
                f"HYWM2GaussianTrain: training finished but no ply dir at "
                f"{ply_dir}. Did the trainer crash silently? Check the worker "
                f"log for tqdm output / errors."
            )
        candidates = sorted(
            ply_dir.glob("point_cloud_*.ply"),
            key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
        )
        if not candidates:
            raise RuntimeError(
                f"HYWM2GaussianTrain: no point_cloud_*.ply in {ply_dir}. "
                f"Either save_ply was False or the trainer didn't reach a save "
                f"step. Re-run with save_ply=True and max_steps >= 4000."
            )
        ply_path = candidates[-1]
        _p(f"final PLY: {ply_path} ({ply_path.stat().st_size / 1e6:.1f} MB)")

        return io.NodeOutput(str(ply_path), str(gs_result))


NODE_CLASS_MAPPINGS = {"HYWM2GaussianTrain": HYWM2GaussianTrain}
NODE_DISPLAY_NAME_MAPPINGS = {"HYWM2GaussianTrain": "HYWM2 Train Gaussians"}
