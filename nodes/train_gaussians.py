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


def _install_preload_only_csw_patch(wgt_mod, debug_5gs: bool = False) -> callable:
    """Monkey-patch `wgt_mod.create_splats_with_optimizers` so when
    `preload_gs_path` is set, splats are built **only** from the preload —
    skipping upstream's init/knn scaffold (random N points + knn-derived
    scales, then `torch.cat` with preload). The scaffold gaussians have
    random positions, init_opa=0.1, and tiny scales — under normal
    training they're pruned within the first densification cycle, but
    with `freeze_count=True` they'd stick around as background noise.
    This patch makes the starting state exactly the preload, nothing else.

    Returns a `restore()` callable. Caller MUST invoke it in a finally
    block so subsequent runs in the same worker see the unpatched
    function.

    If `debug_5gs=True`, the patched function also installs a hook on
    `optimizers["means"].step` that prints 5 fixed-random gaussian
    indices' state after every training iteration (see
    `_install_5gs_step_hook`).
    """
    _orig = wgt_mod.create_splats_with_optimizers

    def _build_from_preload(parser, init_type, init_num_pts, init_extent,
                            init_opacity, init_scale, preload_gs_path,
                            means_lr, scales_lr, opacities_lr, quats_lr,
                            sh0_lr, shN_lr, scene_scale, sh_degree,
                            sparse_grad, visible_adam, batch_size,
                            feature_dim, device, world_rank, world_size,
                            use_mask_gaussian, mask_lr, mask_init_value):
        # ---- Load preload + SH pad ----
        print(f"[HYWM2GaussianTrain] preload-only init: loading splats from "
              f"{preload_gs_path}", file=sys.stderr, flush=True)
        preload_gs = torch.load(preload_gs_path, weights_only=False)
        s = preload_gs["splats"]

        # Pad shN to match the trainer's sh_degree if preload has fewer bands.
        expected_shN_K = (sh_degree + 1) ** 2 - 1
        if "shN" in s and s["shN"].shape[1] < expected_shN_K:
            pad_K = expected_shN_K - s["shN"].shape[1]
            pad = torch.zeros((s["shN"].shape[0], pad_K, 3), dtype=s["shN"].dtype)
            s["shN"] = torch.cat([s["shN"], pad], dim=1)

        def _slice(t):
            return t[world_rank::world_size]

        # ---- Build params straight from preload tensors ----
        params = [
            ["means",     torch.nn.Parameter(_slice(s["means3d"])),   means_lr * scene_scale],
            ["scales",    torch.nn.Parameter(_slice(s["scales"])),    scales_lr],
            ["quats",     torch.nn.Parameter(_slice(s["quats"])),     quats_lr],
            ["opacities", torch.nn.Parameter(_slice(s["opacities"])), opacities_lr],
        ]
        if feature_dim is None:
            params.append(["sh0", torch.nn.Parameter(_slice(s["sh0"])), sh0_lr])
            if "shN" in s:
                params.append(["shN", torch.nn.Parameter(_slice(s["shN"])), shN_lr])
        else:
            if "features" in s:
                params.append(["features", torch.nn.Parameter(_slice(s["features"])), sh0_lr])
            if "colors" in s:
                params.append(["colors", torch.nn.Parameter(_slice(s["colors"])), sh0_lr])

        if use_mask_gaussian:
            N_final = params[0][1].shape[0]
            mask_scores = torch.zeros((N_final, 2))
            mask_scores[:, 0] = mask_init_value
            mask_scores[:, 1] = 1.0
            params.append(["mask_score", torch.nn.Parameter(mask_scores), mask_lr])

        N_total = params[0][1].shape[0]
        print(f"[HYWM2GaussianTrain] preload-only init: {N_total} gaussians "
              f"(no scaffold randoms)", file=sys.stderr, flush=True)

        # ---- Build optimizers (same construction as upstream lines 463-484) ----
        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
        BS = batch_size * world_size
        if sparse_grad:
            optimizer_class = torch.optim.SparseAdam
        elif visible_adam:
            optimizer_class = getattr(wgt_mod, "SelectiveAdam", torch.optim.Adam)
        else:
            optimizer_class = torch.optim.Adam
        optimizers = {
            name: optimizer_class(
                [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, _, lr in params
        }
        return splats, optimizers

    def _patched(parser, init_type="sfm", init_num_pts=100_000, init_extent=3.0,
                 init_opacity=0.1, init_scale=1.0, preload_gs_path=None,
                 means_lr=1.6e-4, scales_lr=5e-3, opacities_lr=5e-2, quats_lr=1e-3,
                 sh0_lr=2.5e-3, shN_lr=2.5e-3, scene_scale=1.0, sh_degree=3,
                 sparse_grad=False, visible_adam=False, batch_size=1,
                 feature_dim=None, device="cuda", world_rank=0, world_size=1,
                 use_mask_gaussian=False, mask_lr=0.01, mask_init_value=10.0):
        if preload_gs_path is None:
            # No preload — preserve upstream behavior exactly.
            return _orig(parser=parser, init_type=init_type,
                         init_num_pts=init_num_pts, init_extent=init_extent,
                         init_opacity=init_opacity, init_scale=init_scale,
                         preload_gs_path=preload_gs_path, means_lr=means_lr,
                         scales_lr=scales_lr, opacities_lr=opacities_lr,
                         quats_lr=quats_lr, sh0_lr=sh0_lr, shN_lr=shN_lr,
                         scene_scale=scene_scale, sh_degree=sh_degree,
                         sparse_grad=sparse_grad, visible_adam=visible_adam,
                         batch_size=batch_size, feature_dim=feature_dim,
                         device=device, world_rank=world_rank, world_size=world_size,
                         use_mask_gaussian=use_mask_gaussian, mask_lr=mask_lr,
                         mask_init_value=mask_init_value)

        splats, optimizers = _build_from_preload(
            parser, init_type, init_num_pts, init_extent, init_opacity,
            init_scale, preload_gs_path, means_lr, scales_lr, opacities_lr,
            quats_lr, sh0_lr, shN_lr, scene_scale, sh_degree, sparse_grad,
            visible_adam, batch_size, feature_dim, device, world_rank,
            world_size, use_mask_gaussian, mask_lr, mask_init_value,
        )
        if debug_5gs:
            _install_5gs_step_hook(splats, optimizers)
        return splats, optimizers

    wgt_mod.create_splats_with_optimizers = _patched

    def _restore():
        wgt_mod.create_splats_with_optimizers = _orig

    return _restore


def _install_5gs_step_hook(splats, optimizers, n_track: int = 5) -> None:
    """Wrap `optimizers["means"].step` so after every optimizer step it
    prints `n_track` fixed gaussian indices' current state — position,
    scale (linear, from exp(log_scale)), quat, opacity (from sigmoid),
    and base color (band-0 SH back to RGB). The indices are picked once
    on the first call so we follow the same gaussians through the run.

    Output format (one line per gaussian):
      [5GS step N] #idx μ=(x,y,z) Δμ=L  σ=(sx,sy,sz)  q=(w,x,y,z)  α=opa  rgb=(r,g,b)
    where Δμ is the L2 norm of the position change since the previous step.
    """
    if "means" not in optimizers:
        print("[HYWM2GaussianTrain] debug_5gs: no 'means' optimizer; skipping hook.",
              file=sys.stderr, flush=True)
        return

    state = {"step": 0, "indices": None, "prev_means": None}

    def _pick_indices():
        N = splats["means"].shape[0]
        if N == 0:
            return torch.tensor([], dtype=torch.long)
        g = torch.Generator()
        g.manual_seed(42)
        k = min(n_track, N)
        return torch.randperm(N, generator=g)[:k]

    def _fmt_vec(v, fmt="{:+.4f}"):
        return "(" + ",".join(fmt.format(float(x)) for x in v) + ")"

    def _print_5gs():
        state["step"] += 1
        if state["indices"] is None:
            state["indices"] = _pick_indices()
            idx_str = ",".join(str(int(i)) for i in state["indices"])
            print(f"[HYWM2GaussianTrain] debug_5gs: tracking indices "
                  f"[{idx_str}] (seed=42)", file=sys.stderr, flush=True)
        idxs = state["indices"]
        if len(idxs) == 0:
            return
        with torch.no_grad():
            means = splats["means"][idxs].detach().cpu()           # [k, 3]
            scales = torch.exp(splats["scales"][idxs]).detach().cpu()  # [k, 3]
            quats = splats["quats"][idxs].detach().cpu()           # [k, 4]
            opa = torch.sigmoid(splats["opacities"][idxs]).detach().cpu()  # [k]
            sh0 = splats["sh0"][idxs, 0, :].detach().cpu()         # [k, 3]
            rgb = (sh0 * _C0 + 0.5).clamp(0.0, 1.0)
        if state["prev_means"] is None:
            dmeans = torch.zeros(len(idxs))
        else:
            dmeans = (means - state["prev_means"]).norm(dim=-1)
        state["prev_means"] = means.clone()

        step_n = state["step"]
        for j, idx in enumerate(idxs.tolist()):
            print(
                f"[5GS step {step_n}] #{idx} "
                f"μ={_fmt_vec(means[j])} "
                f"Δμ={float(dmeans[j]):.2e}  "
                f"σ={_fmt_vec(scales[j], '{:.2e}')}  "
                f"q={_fmt_vec(quats[j], '{:+.3f}')}  "
                f"α={float(opa[j]):.3f}  "
                f"rgb={_fmt_vec(rgb[j], '{:.3f}')}",
                file=sys.stderr, flush=True,
            )

    # Use PyTorch's official post-step hook API. Earlier we tried
    # replacing `optimizer.step` directly, but that turns the bound method
    # into a plain function — and `ExponentialLR.__init__` calls
    # `patch_track_step_called(optimizer)` which does `step_fn.__func__`,
    # which only exists on bound methods. Using the hook API leaves
    # `optimizer.step` untouched.
    def _step_post_hook(_opt, _args, _kwargs):
        try:
            _print_5gs()
        except Exception as e:
            print(f"[HYWM2GaussianTrain] debug_5gs hook error: {e}",
                  file=sys.stderr, flush=True)

    optimizers["means"].register_step_post_hook(_step_post_hook)


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
    in HYWM2's internal frame. The trainer's Parser computes its
    `scene_scale` from the MAX DISTANCE FROM THE CAMERA CENTROID, not
    pairwise distances (`gs/opencv.py:984-987`):

        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    We have to use the SAME metric — pairwise medians can differ by
    orders of magnitude for asymmetric layouts (which HYWM2's normalized
    extrinsics always are: first view at identity, the rest clustered).
    A previous version of this function used pairwise-median + an
    `if max_dist >= 0.1: skip` early-return; both were wrong.

    Strategy: ALWAYS rescale so Parser's max-centroid-distance lands at
    exactly TARGET_SCENE_SCALE = 1.0. Idempotent — a real WorldStereo
    scene already at ~1.0 gets scale_factor=1.0 (no-op).

    Returns (extrinsics_scaled, gaussian_means_scaled, scale_factor).
    """
    TARGET_SCENE_SCALE = 1.0

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

    # Match Parser's exact metric: max distance from camera-centroid.
    scene_center = centers.mean(dim=0)                         # [3]
    dists_from_centroid = (centers - scene_center).norm(dim=1)  # [N]
    max_centroid_dist = float(dists_from_centroid.max().item())

    # Sanity stats for the log — also surface pairwise max for sanity-check
    # against the old metric (helps debug if the two ever disagree wildly).
    cross = centers.unsqueeze(0) - centers.unsqueeze(1)
    dists_pw = cross.norm(dim=-1)
    triu_idx = torch.triu_indices(N, N, offset=1)
    pair_dists = dists_pw[triu_idx[0], triu_idx[1]]
    print(
        f"[HYWM2GaussianTrain] camera-distance stats: "
        f"max_from_centroid={max_centroid_dist:.6f} "
        f"(pairwise median={float(pair_dists.median().item()):.6f}, "
        f"pairwise max={float(pair_dists.max().item()):.6f})",
        file=sys.stderr, flush=True,
    )

    # Genuinely-coincident cameras: nothing we can do, fall through.
    if max_centroid_dist < 1e-9:
        print(
            f"[HYWM2GaussianTrain] cameras are coincident "
            f"(max_centroid_dist={max_centroid_dist:.2e}); skipping rescale. "
            f"Scene scale will be ~0 — training will not optimize means.",
            file=sys.stderr, flush=True,
        )
        return extrinsics, gaussian_means, 1.0

    # Scale so Parser's scene_scale lands at TARGET_SCENE_SCALE.
    scale = TARGET_SCENE_SCALE / max_centroid_dist
    if abs(scale - 1.0) < 0.01:
        print(
            f"[HYWM2GaussianTrain] scene already O(1) "
            f"(max_centroid_dist={max_centroid_dist:.4f} ~ target {TARGET_SCENE_SCALE}); "
            f"no rescale needed (scale={scale:.4f})",
            file=sys.stderr, flush=True,
        )
        # Still return as-is to avoid no-op float drift.
        return extrinsics, gaussian_means, 1.0

    # Scale w2c.t so that c2w.t = -R^T @ (s*t) = s * original_c2w.t — i.e.
    # camera centers (and hence Parser's scene_scale) scale by `s`.
    ext_scaled = ext.clone()
    ext_scaled[:, :3, 3] = ext[:, :3, 3] * scale
    means_scaled = gaussian_means.detach().cpu().float() * scale

    # Restore the input shape (preserve the singleton batch dim if the
    # caller passed [1, N, 4, 4]).
    if extrinsics.dim() == 4 and extrinsics.shape[0] == 1:
        ext_scaled = ext_scaled.unsqueeze(0)

    print(
        f"[HYWM2GaussianTrain] scene scale fix: max_centroid_dist="
        f"{max_centroid_dist:.6f} -> scale_factor={scale:.4f}; "
        f"Parser's scene_scale should land at ~{TARGET_SCENE_SCALE}. "
        f"Output PLY is in scaled world units; multiply by "
        f"{1/scale:.4f} to recover original HYWM2 metric scale.",
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


def _dump_image_batch(
    images: torch.Tensor,
    out_dir: Path,
    keys: list[str],
    target_h: int | None = None,
    target_w: int | None = None,
) -> None:
    """Save [N, H, W, 3] float[0,1] IMAGE batch to PNGs named <key>.png.

    If target_h/target_w are provided and differ from the input's H/W,
    each frame is bilinear-downsampled before saving. Used to materialize
    the trainer's `images_<factor>/` mirror at reduced resolution.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = images.detach().cpu().clamp(0, 1)
    src_h, src_w = int(imgs.shape[1]), int(imgs.shape[2])
    if (target_h is not None and target_w is not None
            and (target_h != src_h or target_w != src_w)):
        # F.interpolate needs [N, C, H, W]; our layout is [N, H, W, C].
        x = imgs.permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(
            x, size=(int(target_h), int(target_w)),
            mode="bilinear", align_corners=False, antialias=True,
        )
        imgs = x.permute(0, 2, 3, 1).clamp(0, 1)
    np_imgs = imgs.numpy()
    for i, frame in enumerate(np_imgs):
        arr = (frame * 255.0 + 0.5).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        Image.fromarray(arr).save(out_dir / f"{keys[i]}.png")


def _dump_depths(
    depths: torch.Tensor,
    out_dir: Path,
    keys: list[str],
    target_h: int | None = None,
    target_w: int | None = None,
) -> float:
    """Save per-frame metric depth as 16-bit grayscale PNGs. Returns the scale
    factor used (so the trainer-side decode can match: depth_meters = png / scale).

    depths: [N, H, W] or [N, H, W, 1|3] float meters.

    target_h / target_w: if provided and != depth dims, resize each frame to
    (target_h, target_w) via bilinear. The trainer reads depths/<key>.png at
    the IMAGE resolution; if they were predicted at a different size (e.g.
    HYWM2Reconstruct ran at adaptive 756² but images are 768²), the
    depth_loss tensor-shape multiply at world_gs_trainer.py:1218 crashes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    d = depths.detach().cpu().float()
    if d.dim() == 4 and d.shape[-1] in (1, 3, 4):
        d = d[..., 0]  # collapse channel — depth scalar replicated to RGB
    # Resize if requested.
    if target_h and target_w and (d.shape[-2] != target_h or d.shape[-1] != target_w):
        d_resized = torch.nn.functional.interpolate(
            d.unsqueeze(1),  # (N, 1, H, W)
            size=(int(target_h), int(target_w)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        d = d_resized
    # Use a per-batch scale so values map nicely into uint16's [0, 65535] range.
    max_d = float(d.max().item())
    if max_d <= 1e-6:
        max_d = 1.0
    scale = 65535.0 / max_d
    arr = (d.numpy() * scale).clip(0, 65535).astype(np.uint16)
    for i, frame in enumerate(arr):
        Image.fromarray(frame, mode="I;16").save(out_dir / f"{keys[i]}.png")
    return scale


def _dump_normals(
    normals: torch.Tensor,
    out_dir: Path,
    keys: list[str],
    target_h: int | None = None,
    target_w: int | None = None,
) -> None:
    """Save per-frame normals as 8-bit RGB PNGs. Input shape [N, H, W, 3]
    expected in [-1, 1] (so (n+1)/2 = [0, 1] = RGB). HYWM2Reconstruct.normals
    already emits in viz form (n+1)/2 ∈ [0, 1] per `decode_normals_image` —
    so we just need to cast.

    target_h / target_w: if provided and != normal dims, resize each frame
    to (target_h, target_w) via bilinear. The trainer reads
    normals/<key>.png at the IMAGE resolution; if they were predicted at a
    different size (e.g. HYWM2Reconstruct's adaptive 756² for 768² images),
    the normal_loss tensor multiply at world_gs_trainer.py:1218 crashes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = normals.detach().cpu().clamp(0, 1).float()
    # Resize if requested. Bilinear is OK for normal-map viz space
    # (subsequent (n+1)/2 inverse leaves vectors approximately normalized;
    # trainer re-normalizes anyway).
    if target_h and target_w and (n.shape[-3] != target_h or n.shape[-2] != target_w):
        # (N, H, W, 3) -> (N, 3, H, W) for interpolate, back to (N, H, W, 3).
        n_resized = torch.nn.functional.interpolate(
            n.permute(0, 3, 1, 2),
            size=(int(target_h), int(target_w)),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()
        n = n_resized.clamp(0, 1)
    n_np = n.numpy()
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


def _read_3dgs_ply(path: str) -> dict:
    """Parse a standard 3DGS PLY into the tensor dict the trainer needs.

    Returns dict with:
      means      [N, 3]   world-space positions (float32)
      quats      [N, 4]   wxyz quaternions (UNNORMALIZED; renderer normalizes)
      scales     [N, 3]   per-axis LOG scales (real_scale = exp(scales))
      opacities  [N]      sigmoid-space opacities (real = sigmoid(opacities))
      sh0        [N, 1, 3] band-0 SH coefficients
      shN        [N, K, 3] higher SH coefficients; K = (sh_degree+1)² - 1.
                          Empty (K=0) for sh_degree=0 PLYs.

    Field layout follows `graphdeco-inria/gaussian-splatting`: `x/y/z`,
    `rot_0..3` (wxyz), `scale_0..2` (log), `opacity` (logit), `f_dc_0..2`
    (sh0), `f_rest_0..N` (higher SH bands, interleaved R-then-G-then-B).
    Same parser as `ComfyUI-GaussianPack/preview_gaussian_camera.py:111`,
    copied inline to avoid a cross-pack import.
    """
    from plyfile import PlyData

    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"{path}: PLY has no 'vertex' element (not a 3DGS PLY)")
    v = ply["vertex"].data
    N = len(v)
    names = set(v.dtype.names)

    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError(f"{path}: PLY vertex missing x/y/z (not a point cloud)")
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    if all(k in names for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
        quats = np.stack(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1
        ).astype(np.float32)
    else:
        quats = np.zeros((N, 4), dtype=np.float32)
        quats[:, 0] = 1.0

    if all(k in names for k in ("scale_0", "scale_1", "scale_2")):
        scales = np.stack(
            [v["scale_0"], v["scale_1"], v["scale_2"]], axis=1
        ).astype(np.float32)
    else:
        scales = np.full((N, 3), math.log(1e-3), dtype=np.float32)

    if "opacity" in names:
        opacities = np.asarray(v["opacity"]).astype(np.float32)
    else:
        opacities = np.zeros(N, dtype=np.float32)

    if all(k in names for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh0 = np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1
        ).astype(np.float32)[:, None, :]
    else:
        sh0 = np.full((N, 1, 3), 0.5 / _C0, dtype=np.float32)

    rest_keys = sorted(
        (n for n in names if n.startswith("f_rest_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    K_total = len(rest_keys)
    if K_total > 0 and K_total % 3 == 0:
        K_AC = K_total // 3
        stacked = np.stack(
            [np.asarray(v[k]) for k in rest_keys], axis=1
        ).astype(np.float32)
        shN = stacked.reshape(N, 3, K_AC).transpose(0, 2, 1)
    else:
        shN = np.zeros((N, 0, 3), dtype=np.float32)

    sh_degree = (
        int(round(math.sqrt(1 + shN.shape[1]))) - 1 if shN.shape[1] > 0 else 0
    )
    print(
        f"[HYWM2GaussianTrain] read PLY: {N} gaussians, sh_degree={sh_degree} "
        f"(shN K={shN.shape[1]}) from {path}",
        file=sys.stderr, flush=True,
    )

    return {
        "means": torch.from_numpy(means),
        "quats": torch.from_numpy(quats),
        "scales": torch.from_numpy(scales),
        "opacities": torch.from_numpy(opacities),
        "sh0": torch.from_numpy(sh0),
        "shN": torch.from_numpy(shN),
    }


def _matrix_to_quaternion_wxyz(R: torch.Tensor) -> torch.Tensor:
    """3×3 rotation matrix -> wxyz unit quaternion (Shepperd's method).

    Trace-branching for numerical stability. Returns shape [4] on CPU.
    Convention matches 3DGS: q = (w, x, y, z) such that
    rotation_apply(q, v) = q ⊗ (0, v) ⊗ q*.
    """
    R = R.float().cpu()
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = float((1.0 + trace).sqrt() * 2.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]).item() / s
        y = (R[0, 2] - R[2, 0]).item() / s
        z = (R[1, 0] - R[0, 1]).item() / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = float((1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2.0)
        w = (R[2, 1] - R[1, 2]).item() / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]).item() / s
        z = (R[0, 2] + R[2, 0]).item() / s
    elif R[1, 1] >= R[2, 2]:
        s = float((1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2.0)
        w = (R[0, 2] - R[2, 0]).item() / s
        x = (R[0, 1] + R[1, 0]).item() / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]).item() / s
    else:
        s = float((1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2.0)
        w = (R[1, 0] - R[0, 1]).item() / s
        x = (R[0, 2] + R[2, 0]).item() / s
        y = (R[1, 2] + R[2, 1]).item() / s
        z = 0.25 * s
    q = torch.tensor([w, x, y, z], dtype=torch.float32)
    return q / q.norm()


def _quat_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2 in wxyz convention. Broadcasts on leading dims.

    Input shapes: q1 [..., 4], q2 [..., 4]. Output: [..., 4].
    Used to compose rotations: combined = q_R ⊗ q_orig where q_R is a
    pre-rotation applied to the gaussian's orientation.
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _transform_gaussians_to_frame(
    gaussians: dict,
    from_extrinsics: torch.Tensor,
    to_extrinsics: torch.Tensor,
) -> dict:
    """Rigid-transform gaussian means + quats from one camera-pose frame
    to another.

    Both `from_extrinsics` and `to_extrinsics` describe the SAME physical
    cameras (same view 0, same view 1, ...) in two different world-frame
    conventions. The transform T_from2to = inv(W_from_0) @ W_to_0 sends a
    point in `from`'s world to the equivalent point in `to`'s world.

    Use case: HYWM2Reconstruct returns `gaussians` in its rebased world
    where view-0 extrinsic ≈ identity. PanoramaSplit's original
    extrinsics describe the same 12 cameras in a different frame (view-0
    is a rotation, not identity). To train with PanoramaSplit's
    extrinsics, we transform the HYWM2 gaussians into PanoramaSplit's
    frame at the input boundary of the trainer.

    Transformed fields: means (homogeneous), quats (pre-multiply by q_R).
    Untransformed: scales, opacities, sh0 (rotation-invariant).
    shN (higher SH bands): NOT rotated. Full SH rotation requires
    Wigner D-matrices — warn and skip if non-empty.

    No-op fast path: if T is within 1e-4 of identity (the two frames
    already match), return the original dict unchanged.
    """
    W_from = from_extrinsics
    W_to = to_extrinsics
    if W_from.dim() == 4 and W_from.shape[0] == 1:
        W_from = W_from[0]
    if W_to.dim() == 4 and W_to.shape[0] == 1:
        W_to = W_to[0]
    if W_from.dim() == 3:
        W_from = W_from[0]
    if W_to.dim() == 3:
        W_to = W_to[0]
    W_from = W_from.float().cpu()
    W_to = W_to.float().cpu()
    if W_from.shape == (3, 4):
        W_from = torch.cat(
            [W_from, torch.tensor([[0., 0., 0., 1.]])], dim=0,
        )
    if W_to.shape == (3, 4):
        W_to = torch.cat(
            [W_to, torch.tensor([[0., 0., 0., 1.]])], dim=0,
        )

    T = torch.linalg.inv(W_from) @ W_to       # [4, 4], from-world -> to-world
    if torch.allclose(T, torch.eye(4), atol=1e-4):
        _p("transform_gaussians: from/to frames identical within 1e-4 (no-op)")
        return gaussians

    R = T[:3, :3]
    t = T[:3, 3]
    _p(
        f"transform_gaussians: rotating PLY gaussians into target frame "
        f"(|R - I|_F = {(R - torch.eye(3)).norm().item():.4f}, "
        f"|t| = {t.norm().item():.4f})"
    )

    means_in = gaussians["means"].float().cpu()                   # [N, 3]
    means_new = (R @ means_in.T).T + t                            # [N, 3]
    q_R = _matrix_to_quaternion_wxyz(R)                           # [4]
    q_R_b = q_R.unsqueeze(0).expand(gaussians["quats"].shape[0], 4)
    quats_new = _quat_multiply_wxyz(q_R_b, gaussians["quats"].float().cpu())

    shN = gaussians.get("shN")
    if shN is not None and shN.shape[1] > 0:
        _p(
            f"WARNING: shN has K={shN.shape[1]} higher SH bands which are "
            f"NOT being rotated. Full SH rotation requires Wigner D-matrices. "
            f"Appearance may have view-dependent artifacts; if visible, train "
            f"with sh_degree=0 or contribute Wigner D rotation."
        )

    out = dict(gaussians)
    out["means"] = means_new
    out["quats"] = quats_new
    return out


def _render_first_frames(
    gaussians: dict,
    extrinsics_w2c: torch.Tensor,
    intrinsics_pixel: torch.Tensor,
    input_images: torch.Tensor,
    height: int,
    width: int,
    near: float = 0.01,
) -> torch.Tensor:
    """Render the input gaussians from each provided extrinsic at full
    resolution, then concatenate side-by-side with the matching input
    image. Returns batch [N, H, 2W, 3] in [0, 1].

    Left half of each frame = render. Right half = ground-truth image.

    Used as a diagnostic right before training starts — if the two halves
    don't match, the gaussians and extrinsics are in different frames
    (or wrong intrinsics, axis flips, etc.). Saves the user from sitting
    through a long training run only to find the alignment was wrong.

    gsplat's `viewmats` arg expects w2c (the trainer at
    world_gs_trainer.py:943 explicitly inverts c2w to w2c before
    passing), so we pass `extrinsics_w2c` directly.

    Rendered per-view (V loop) to keep VRAM bounded at large N_gauss.
    """
    from gsplat.rendering import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means = gaussians["means"].to(device, torch.float32)
    quats = gaussians["quats"].to(device, torch.float32)
    scales_lin = gaussians["scales"].to(device, torch.float32).exp()
    opacities_lin = gaussians["opacities"].to(device, torch.float32).sigmoid()
    sh0 = gaussians["sh0"].to(device, torch.float32)             # [N, 1, 3]
    shN = gaussians.get("shN")
    if shN is not None and shN.shape[1] > 0:
        colors = torch.cat([sh0, shN.to(device, torch.float32)], dim=1)
    else:
        colors = sh0
    sh_degree = int(round(math.sqrt(colors.shape[1]))) - 1

    viewmats = extrinsics_w2c.to(device, torch.float32)
    if viewmats.dim() == 4 and viewmats.shape[0] == 1:
        viewmats = viewmats[0]
    Ks = intrinsics_pixel.to(device, torch.float32)
    if Ks.dim() == 4 and Ks.shape[0] == 1:
        Ks = Ks[0]
    bg = torch.zeros(3, dtype=torch.float32, device=device)

    V = int(viewmats.shape[0])
    _p(f"first_renders: rasterizing {V} views @ {width}×{height} "
       f"(N_gauss={means.shape[0]}, sh_degree={sh_degree})")

    renders = []
    for v in range(V):
        rc, _alphas, _info = rasterization(
            means=means,
            quats=quats,
            scales=scales_lin,
            opacities=opacities_lin,
            colors=colors,
            viewmats=viewmats[v:v + 1],
            Ks=Ks[v:v + 1],
            width=int(width),
            height=int(height),
            near_plane=float(near),
            sh_degree=sh_degree,
            backgrounds=bg,
            packed=True,
        )
        renders.append(rc[0].clamp(0.0, 1.0).cpu())              # [H, W, 3]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    renders_t = torch.stack(renders, dim=0)                      # [N, H, W, 3]

    # Match the input images batch to (V, H, W, 3) for side-by-side
    # concat. Inputs come in at native resolution which may equal H/W
    # already; if not, bilinear-resize so the two halves line up.
    inputs = input_images.detach().cpu().float().clamp(0, 1)
    if inputs.dim() == 3:
        inputs = inputs.unsqueeze(0)
    if int(inputs.shape[1]) != height or int(inputs.shape[2]) != width:
        x = inputs.permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(
            x, size=(height, width), mode="bilinear", align_corners=False,
            antialias=True,
        )
        inputs = x.permute(0, 2, 3, 1).clamp(0, 1)

    # Crop to matching V if either side has extras.
    V_pair = min(renders_t.shape[0], inputs.shape[0])
    side_by_side = torch.cat(
        [renders_t[:V_pair], inputs[:V_pair]], dim=2,             # cat width
    )                                                             # [V, H, 2W, 3]
    _p(f"first_renders: emitting [{V_pair}, {height}, {2 * width}, 3] "
       f"side-by-side (left=render, right=input)")
    return side_by_side


def _write_preload_pt(
    out_path: Path,
    gaussians: dict,
    max_gaussians: int = 0,
    world_scale: float = 1.0,
) -> None:
    """Save splats as the trainer's preload_gs_path .pt.

    Schema the trainer expects (world_gs_trainer.py:442-451):
        {"splats": {"means3d", "quats", "scales", "opacities", "sh0", "shN"}}
    where sh0 is [N, 1, 3] and shN is [N, K, 3] (K depends on sh_degree).

    Input dict (from `_read_3dgs_ply`) carries sh0 and shN directly —
    higher SH bands are preserved straight through, no rgb→sh0 conversion.

    `max_gaussians > 0` caps the preload via random subsample. Defends
    against OOM in gsplat's isect_tiles when the source PLY has millions
    of gaussians (its tile-intersection workspace scales as
    O(N_gaussians × max_overlapping_tiles)).

    `world_scale != 1.0` rescales both means (linear) and scales (additive
    in log space: log(s · world_scale) = log(s) + log(world_scale)). The
    trainer's scales are stored in log-space (`world_gs_trainer.py:408`).
    """
    means = gaussians["means"].detach().float().cpu()
    quats = gaussians["quats"].detach().float().cpu()
    scales = gaussians["scales"].detach().float().cpu()
    opacities = gaussians["opacities"].detach().float().cpu()
    sh0 = gaussians["sh0"].detach().float().cpu()
    shN = gaussians["shN"].detach().float().cpu()

    # World rescale: means scale linearly; log-space scales pick up a
    # log(world_scale) offset.
    if world_scale != 1.0:
        means = means * float(world_scale)
        scales = scales + math.log(float(world_scale))

    N = means.shape[0]

    if 0 < max_gaussians < N:
        g = torch.Generator(device="cpu").manual_seed(N)
        keep_idx = torch.randperm(N, generator=g)[:max_gaussians]
        means = means[keep_idx]
        quats = quats[keep_idx]
        scales = scales[keep_idx]
        opacities = opacities[keep_idx]
        sh0 = sh0[keep_idx]
        if shN.shape[1] > 0:
            shN = shN[keep_idx]
        else:
            shN = torch.zeros((max_gaussians, 0, 3), dtype=torch.float32)
        N_in = gaussians["means"].shape[0]
        N = max_gaussians
        print(
            f"[HYWM2GaussianTrain] preload subsampled {N_in} -> {N} gaussians (cap)",
            file=sys.stderr, flush=True,
        )

    payload = {
        "step": 0,
        "splats": {
            # Position key is "means3d" not "means" — the trainer renames
            # it at world_gs_trainer.py:444.
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
                "by training.sh) directly from a ComfyUI workflow. Takes a 3DGS "
                "PLY path as the starting splat, materializes the gs_data/ "
                "layout from images + extrinsics + intrinsics, and returns the "
                "path to the final trained PLY. To train freshly-generated "
                "HYWM2 gaussians: wire HYWM2Reconstruct -> "
                "HYWM2ExportGaussiansPLY -> here."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "ply_path",
                    tooltip="Absolute path to a 3DGS PLY file (standard "
                            "graphdeco-inria layout: x/y/z, rot_0..3, "
                            "scale_0..2, opacity, f_dc_0..2, f_rest_*). The "
                            "full splat state is loaded and saved as "
                            "preload_gs.pt so the trainer hot-starts from it; "
                            "a 1000-point subsample is also written as "
                            "points.ply to satisfy the Parser. Use "
                            "HYWM2ExportGaussiansPLY for fresh HYWM2 output, "
                            "or wire from any node that outputs a 3DGS PLY."),
                io.Image.Input(
                    "images",
                    tooltip="Training views [N, H, W, 3] float[0,1]. Should be "
                            "the SAME N views you ran HYWM2Reconstruct on."),
                io.Custom("EXTRINSICS").Input(
                    "extrinsics",
                    tooltip="Per-view world-to-camera [N, 4, 4] — the "
                            "TARGET frame in which the trained PLY will "
                            "live. Pass PanoramaSplit's original "
                            "extrinsics if you want the output PLY in "
                            "the original panorama world frame. The "
                            "trainer will transform the input gaussians "
                            "into this frame if `gaussians_extrinsics` "
                            "is also wired (e.g. HYWM2 reconstruction)."),
                io.Custom("INTRINSICS").Input(
                    "intrinsics",
                    tooltip="Per-view K [N, 3, 3]. Accepts pixel-K (Sharp/HYWM2 "
                            "convention) or normalized-K (PanoPack convention) — "
                            "auto-rescaled to pixel-K for the image (W, H) at "
                            "execute time."),
                io.Custom("EXTRINSICS").Input(
                    "gaussians_extrinsics", optional=True,
                    tooltip=(
                        "OPTIONAL. The extrinsics in whose world-frame "
                        "the input PLY's gaussians live. Wire "
                        "HYWM2Reconstruct.predicted_extrinsics here — "
                        "HYWM2 rebases the world so view 0 becomes "
                        "identity, but PanoramaSplit's extrinsics keep "
                        "the original frame. When provided, the trainer "
                        "applies a rigid transform to the PLY gaussians "
                        "(means + quats) at load time so they align with "
                        "the `extrinsics` frame. Leave unwired when the "
                        "PLY is already in `extrinsics`'s frame (e.g. "
                        "training from a previously-trained PLY)."
                    )),
                io.Int.Input(
                    "max_steps", default=5000, min=0, max=50000,
                    tooltip="Number of training iterations. Default 5000 "
                            "matches training.sh's default and gives "
                            "publication-quality splats in ~10-30 min on a "
                            "3090. For quick smoke tests try 200-500. "
                            "Set to 0 to skip training entirely — the "
                            "node will only compute the `first_renders` "
                            "diagnostic and return the input PLY as the "
                            "`ply_path` output. Useful for validating "
                            "alignment without paying for any training "
                            "time."),
                io.Int.Input(
                    "data_factor", default=2, min=1, max=8, optional=True,
                    tooltip=(
                        "Training resolution downscale. 1 = full image "
                        "resolution; 2 = half (4× memory reduction, default); "
                        "4 = quarter (16× memory reduction). The trainer's "
                        "Parser scales images AND intrinsics together so "
                        "geometry is preserved — only the rasterizer's per-"
                        "pixel cost changes. Output PLY is in original world "
                        "units regardless. Bump to 2-4 when OOM hits on big "
                        "splats; drop to 1 for max quality on small splats."
                    )),
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
                    "resize_inputs", default=True, optional=True,
                    tooltip=(
                        "When True (default), the optional depth_raw and "
                        "normals inputs are bilinear-resized to match the "
                        "images batch's (H, W) before they're written to "
                        "depths/ and normals/. HYWM2Reconstruct's adaptive "
                        "target_size often differs from the IMAGE input's "
                        "native size (e.g. images are 768×768 but HYWM2 ran "
                        "at 756×756), and the trainer's normal/depth loss "
                        "tensor-multiply at world_gs_trainer.py:1218 "
                        "requires matching dims. When False, the node "
                        "errors out with a clear shape-mismatch message "
                        "instead of silently rescaling — use this if you've "
                        "carefully prepared depth/normal maps at the image "
                        "resolution and want a hard guard against unintended "
                        "rescaling."
                    )),
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
                io.Boolean.Input(
                    "freeze_positions", default=False, optional=True,
                    tooltip=(
                        "Lock gaussian POSITIONS (means) at their HYWM2 "
                        "preload values. Only colors / scales / opacities "
                        "/ rotations train. Use when HYWM2's geometry is "
                        "already good (typical for panorama inputs) and "
                        "you only want to refine appearance. Internally "
                        "sets cfg.means_lr = 0.0."
                    )),
                io.Boolean.Input(
                    "freeze_count", default=False, optional=True,
                    tooltip=(
                        "Disable densification + pruning. Gaussian count "
                        "stays exactly as preloaded for the full training "
                        "run — no splits, duplicates, or prunes. Pairs "
                        "well with freeze_positions for pure appearance "
                        "refinement. Internally sets "
                        "cfg.strategy.refine_stop_iter = 0."
                    )),
                io.Boolean.Input(
                    "start_from_preload", default=True, optional=True,
                    tooltip=(
                        "When True (default), training starts with EXACTLY "
                        "the HYWM2 preload gaussians — no scaffold randoms. "
                        "Upstream's create_splats_with_optimizers always "
                        "builds an init block (sfm or random) FIRST and "
                        "concatenates the preload onto it; those scaffold "
                        "gaussians normally get pruned by step ~100, but "
                        "with freeze_count=True they stick around forever. "
                        "This toggle monkey-patches the trainer to build "
                        "params directly from the preload, skipping the "
                        "init scaffold entirely. Turn off only to reproduce "
                        "upstream behavior exactly."
                    )),
                io.Boolean.Input(
                    "debug_first_renders", default=True, optional=True,
                    tooltip=(
                        "Diagnostic: at step 0, render the input PLY's "
                        "gaussians from each provided extrinsic via "
                        "gsplat, then concatenate side-by-side with the "
                        "matching input image. Emitted as the "
                        "`first_renders` output. Default True since this "
                        "is the single most useful sanity check for "
                        "alignment bugs. Turn off to skip the extra "
                        "rasterizer pass (~1-3s for N=12 @ 768²)."
                    )),
                io.Boolean.Input(
                    "debug_print_5gs", default=False, optional=True,
                    tooltip=(
                        "Diagnostic: every training step, print 5 fixed-"
                        "random gaussians' state (position, position-delta, "
                        "linear scale, quaternion, opacity, base RGB color) "
                        "to stderr. Indices are picked once at seed=42 so "
                        "we follow the same 5 across the whole run. Useful "
                        "for sanity-checking that means_lr * scene_scale "
                        "is non-zero (positions move) and that "
                        "scales/opacity/colors are actually updating. "
                        "Noisy — 5 lines per step."
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
                io.Image.Output(
                    display_name="first_renders",
                    tooltip=(
                        "Diagnostic: per-view side-by-side image "
                        "[render | input_image] for the gaussian state at "
                        "training step 0, at FULL image resolution "
                        "(W_img × H_img). Matches the resolution of the "
                        "post-training preview (PreviewGaussianCamera). "
                        "If alignment is correct, both halves match."
                    )),
                io.Image.Output(
                    display_name="first_renders_training_res",
                    tooltip=(
                        "Diagnostic: same as `first_renders` but rendered "
                        "at the actual TRAINING resolution "
                        "(W_img/data_factor × H_img/data_factor with "
                        "scaled K). This is what the trainer's loss "
                        "function actually sees. Compare to "
                        "`first_renders` to spot training-resolution "
                        "artifacts (e.g. gaussians too small for "
                        "full-res preview but fine for training-res "
                        "rasterization). Identical to `first_renders` "
                        "when data_factor=1."
                    )),
            ],
        )

    @classmethod
    def execute(
        cls,
        ply_path: str,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        max_steps: int = 5000,
        data_factor: int = 2,
        preset: str = "default",
        save_ply: bool = True,
        preload_max_gaussians: int = 0,
        output_prefix: str = "hywm2_train",
        hyworld2_repo_path: str = "/home/work/HY-World-2.0",
        resize_inputs: bool = True,
        offload_everything: bool = True,
        freeze_positions: bool = False,
        freeze_count: bool = False,
        start_from_preload: bool = True,
        debug_first_renders: bool = True,
        debug_print_5gs: bool = False,
        gaussians_extrinsics: torch.Tensor | None = None,
        depth_raw: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
    ):
        # ----- Validate inputs -----
        if not ply_path or not Path(ply_path).is_file():
            raise ValueError(
                f"HYWM2GaussianTrain: ply_path must point to an existing 3DGS "
                f"PLY file; got {ply_path!r}."
            )
        _p(f"reading PLY: {ply_path}")
        gaussians = _read_3dgs_ply(ply_path)
        means_t = gaussians["means"]
        if means_t.dim() != 2 or means_t.shape[0] < 1 or means_t.shape[1] != 3:
            raise ValueError(
                f"HYWM2GaussianTrain: PLY produced means with shape "
                f"{tuple(means_t.shape)}; expected [N>=1, 3]."
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

        # Transform the input PLY gaussians into the `extrinsics` frame if
        # the user provided a different `gaussians_extrinsics`. This is the
        # fix for HYWM2-style "view 0 anchored to identity" rebasing — the
        # gaussians live in HYWM2's rebased world but we want to train and
        # output in PanoramaSplit's original world. See
        # `_transform_gaussians_to_frame` docstring for the math.
        if gaussians_extrinsics is not None:
            gaussians = _transform_gaussians_to_frame(
                gaussians,
                from_extrinsics=gaussians_extrinsics,
                to_extrinsics=ext,
            )
            means_t = gaussians["means"]   # refresh — transform replaced it

        N_gauss = int(means_t.shape[0])
        _p(f"inputs OK: N_views={N_img} @ {W_img}×{H_img}, N_gauss={N_gauss}, "
           f"max_steps={max_steps}, preset={preset}, depth_loss={depth_raw is not None}, "
           f"normal_loss={normals is not None}, "
           f"gaussians_extrinsics_wired={gaussians_extrinsics is not None}")

        # ----- Camera-geometry debug dump (alignment forensics) -----
        # Helps verify:
        # - Extrinsics are w2c (we ASSUME so; trainer at gs/opencv.py:604
        #   does camtoworlds = inv(w2c_mats)).
        # - Camera positions are at world origin for panorama splits, or
        #   spread for video.
        # - Look directions ("forward" axes) cover the sphere for an
        #   icosahedron-12 split (no two views nearly-identical).
        # - Pixel K's fx, fy ≈ W/2 / tan(fov/2) for the expected FOV.
        # - The PLY gaussians sit IN FRONT OF the cameras (positive Z in
        #   OpenCV cam frame).
        def _summarize_extrinsics(ext_t: torch.Tensor, intr_t: torch.Tensor,
                                  gaussians_means: torch.Tensor,
                                  label: str) -> None:
            e = ext_t.detach().cpu().float()
            if e.dim() == 4 and e.shape[0] == 1:
                e = e[0]
            R = e[:, :3, :3]                                # [N, 3, 3]
            t = e[:, :3, 3:4]                               # [N, 3, 1]
            centers = -torch.bmm(R.transpose(1, 2), t).squeeze(-1)  # world-space cam centers
            # OpenCV camera-frame convention: +X right, +Y down, +Z forward.
            # World-space forward direction of camera i:  R[i]^T @ [0,0,1].
            forward_world = R[:, 2, :]   # row 2 of R is the +Z axis in world after  R*x_world = x_cam.
            # Sanity (alternative): in OpenCV w2c, R = [r1; r2; r3]; the
            # camera's +Z axis in world is r3.
            up_world = -R[:, 1, :]       # +Y_cam points DOWN in OpenCV, so world-up ≈ -row1.
            right_world = R[:, 0, :]
            _p(f"camera geometry [{label}] N={e.shape[0]}:")
            _p(f"  center stats: "
               f"mean={tuple(round(float(x), 4) for x in centers.mean(0).tolist())} "
               f"range=[{centers.min().item():.4f}, {centers.max().item():.4f}]")
            # Per-view dump (compact, one line each).
            for i in range(e.shape[0]):
                c  = centers[i].tolist()
                fw = forward_world[i].tolist()
                up = up_world[i].tolist()
                _p(f"  view {i:>2}: center=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  "
                   f"forward=({fw[0]:+.3f},{fw[1]:+.3f},{fw[2]:+.3f})  "
                   f"up=({up[0]:+.3f},{up[1]:+.3f},{up[2]:+.3f})  "
                   f"|fwd|={torch.tensor(fw).norm().item():.3f}")
            # Per-view K compact dump.
            k = intr_t.detach().cpu().float()
            if k.dim() == 4 and k.shape[0] == 1:
                k = k[0]
            for i in range(min(2, k.shape[0])):
                K = k[i]
                _p(f"  K[{i}]: fx={K[0,0].item():.2f} fy={K[1,1].item():.2f} "
                   f"cx={K[0,2].item():.2f} cy={K[1,2].item():.2f}  "
                   f"(expected fx~fy~{W_img/2/math.tan(math.radians(90)/2):.1f} for FOV=90°)")
            # Gaussian extent vs cameras.
            g = gaussians_means.detach().cpu().float()
            g_center = g.mean(0)
            g_radius = (g - g_center).norm(dim=-1).max().item()
            _p(f"  gaussians: N={g.shape[0]} centroid="
               f"({g_center[0].item():+.3f},{g_center[1].item():+.3f},{g_center[2].item():+.3f}) "
               f"max_radius_from_centroid={g_radius:.3f}")
            # For each view: where are gaussians relative to camera frame?
            # In OpenCV cam frame, a gaussian at world p projects as p_cam = R @ p + t.
            # If p_cam.z < 0 the gaussian is BEHIND the camera (won't render).
            # Sample-test on a random 1000 gaussians.
            if g.shape[0] > 0:
                k_sample = min(1000, g.shape[0])
                samp = g[torch.linspace(0, g.shape[0]-1, k_sample).long()]
                for i in range(min(3, e.shape[0])):
                    p_cam = (R[i] @ samp.T + t[i]).T   # [k_sample, 3]
                    z_cam = p_cam[:, 2]
                    pct_front = float((z_cam > 0).sum().item()) / k_sample * 100.0
                    _p(f"  view {i:>2}: {pct_front:.1f}% of gaussians IN FRONT of camera "
                       f"(z_cam > 0 in OpenCV convention)")

        _summarize_extrinsics(ext, intr_pixel, means_t, "input (pre-rescale)")

        # ----- First-renders diagnostic -----
        # Render the gaussians as the trainer is about to see them, BEFORE
        # any training-induced color/scale drift. If left half (render)
        # doesn't match right half (input) per view, training will produce
        # garbage because the optimizer has to "correct" a geometry
        # mismatch with appearance changes.
        first_renders_out: torch.Tensor
        first_renders_train_out: torch.Tensor
        # Force-on when max_steps=0 (the whole point of that mode is to
        # see the renders without training; skipping the render too would
        # leave nothing useful).
        want_first = debug_first_renders or int(max_steps) == 0
        if want_first:
            try:
                first_renders_out = _render_first_frames(
                    gaussians, ext, intr_pixel, images,
                    height=H_img, width=W_img,
                )
            except Exception as e:
                _p(f"first_renders failed (continuing): {type(e).__name__}: {e}")
                first_renders_out = torch.zeros(
                    (int(N_img), int(H_img), int(2 * W_img), 3),
                    dtype=torch.float32,
                )
            # Second render at training resolution — exactly what the
            # trainer's loss function sees per step. Use Parser's K-scaling
            # (gs/opencv.py:581: K[:2,:] /= factor) so the projection
            # matches what `rasterize_splats` does inside the trainer.
            df = int(data_factor)
            train_h = H_img // df
            train_w = W_img // df
            try:
                K_train = intr_pixel.detach().clone()
                # Scale K's first two rows: fx, fy, cx, cy all divided
                # by data_factor. Operates on the per-view batch [N, 3, 3].
                K_train[..., :2, :] = K_train[..., :2, :] / df
                first_renders_train_out = _render_first_frames(
                    gaussians, ext, K_train, images,
                    height=train_h, width=train_w,
                )
            except Exception as e:
                _p(f"first_renders_training_res failed (continuing): "
                   f"{type(e).__name__}: {e}")
                first_renders_train_out = torch.zeros(
                    (int(N_img), int(train_h), int(2 * train_w), 3),
                    dtype=torch.float32,
                )
        else:
            first_renders_out = torch.zeros(
                (1, int(H_img), int(2 * W_img), 3), dtype=torch.float32,
            )
            first_renders_train_out = torch.zeros(
                (1, int(H_img // int(data_factor)),
                 int(2 * W_img // int(data_factor)), 3),
                dtype=torch.float32,
            )

        # ----- max_steps=0 short-circuit -----
        # User just wants alignment validation, no training. Skip all the
        # disk writes, trainer subprocess spinup, and the main() call.
        # Return the INPUT ply_path so downstream nodes still receive a
        # valid PLY (the input PLY is in the same frame the renders were
        # produced in, so consistent for further visualization).
        if int(max_steps) == 0:
            _p("max_steps=0 -> skipping training; returning input PLY + "
               "first_renders only")
            return io.NodeOutput(
                str(ply_path), "", first_renders_out, first_renders_train_out,
            )

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
        # Naming: prefix `panorama_` forces the Parser's Dataset to treat
        # these as panorama frames, which bypasses the train/val split rule
        # `idx % test_every != 0` at gs/opencv.py:1061 — `is_pano=True` puts
        # every frame in train. We don't want a val hold-out for a 12-view
        # panorama reconstruction (every face is essential supervision).
        keys = [f"panorama_{i:04d}" for i in range(N_img)]

        # The Parser requires `images/` to ALWAYS exist (it reads each
        # PNG's W×H from there at gs/opencv.py:588, even when factor > 1).
        # When data_factor > 1 it then loads training pixels from
        # `images_<factor>/`, which must be the downsampled mirror.
        _p(f"writing {N_img} full-res images to images/...")
        _dump_image_batch(images, gs_data / "images", keys)
        train_h = H_img // int(data_factor)
        train_w = W_img // int(data_factor)
        if int(data_factor) > 1:
            ds_dir = gs_data / f"images_{int(data_factor)}"
            _p(f"writing {N_img} downsampled images ({train_w}×{train_h}) "
               f"to images_{int(data_factor)}/...")
            _dump_image_batch(images, ds_dir, keys,
                              target_h=train_h, target_w=train_w)

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
        _summarize_extrinsics(ext, intr_pixel, gaussian_means_scaled,
                              f"post-rescale (world_scale={world_scale:.4f})")

        _p("writing cameras.json...")
        _write_cameras_json(gs_data / "cameras.json", ext, intr_pixel, keys)

        # points.ply needs >= a few hundred vertices so the trainer's Parser
        # can run `align_principal_axes` (which bails at <3 points and
        # returns identity), and so `parser.scene_center` / `parser.points`
        # are meaningful for downstream rendering bounds. With
        # cfg.init_type='random' the Parser's points are NOT used to seed
        # gaussians (random branch ignores parser.points) — they're only
        # used for the principal-axes / bounds metadata. The REAL starting
        # state is still the HYWM2 preload via cfg.preload_gs_path.
        N_seed = int(min(1000, gaussian_means_scaled.shape[0]))
        if N_seed >= 2:
            seed_idx = torch.linspace(
                0, gaussian_means_scaled.shape[0] - 1, N_seed
            ).long()
        else:
            seed_idx = torch.tensor([0], dtype=torch.long)
        # Recover RGBs from band-0 SH: rgb = sh0 * C0 + 0.5 (inverse of
        # gs/utils.rgb_to_sh). points.ply needs uint8 RGB, which the writer
        # quantizes from [0,1] linear.
        sh0_t = gaussians["sh0"]  # [N, 1, 3]
        if sh0_t.dim() == 3 and sh0_t.shape[1] >= 1:
            rgbs_for_ply = (sh0_t[:, 0, :] * _C0 + 0.5).clamp(0.0, 1.0)
        else:
            rgbs_for_ply = torch.full((N_gauss, 3), 0.5)
        _p(f"writing points.ply ({N_seed}-vertex subsample, init_type=random)")
        _write_points_ply(
            gs_data / "points.ply",
            gaussian_means_scaled[seed_idx].detach().cpu(),
            rgbs_for_ply[seed_idx].detach().cpu(),
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

        # Resolution-mismatch handling. The trainer reads depths/normals
        # at the image resolution and multiplies them against rendered
        # tensors — if they're not the same shape it crashes. Bilinear-
        # resize when resize_inputs=True; otherwise error out loudly.
        def _check_or_resize(tensor: torch.Tensor, name: str) -> tuple[int | None, int | None]:
            """Return (target_h, target_w) for the dump helper. Target is
            the training resolution = full / data_factor — the trainer
            renders at this size and multiplies the loaded depth/normal
            against the rendered tensor (world_gs_trainer.py:1218), so
            shapes must match `train_h × train_w`, NOT the full image.

            Raises if shapes mismatch and resize_inputs is False.
            """
            if tensor is None:
                return None, None
            t = tensor
            if t.dim() == 3:
                t_h, t_w = int(t.shape[-2]), int(t.shape[-1])
            elif t.dim() == 4:
                # [N, H, W, C] convention from HYWM2Reconstruct.
                if t.shape[-1] in (1, 3, 4):
                    t_h, t_w = int(t.shape[1]), int(t.shape[2])
                else:
                    # legacy [N, C, H, W]
                    t_h, t_w = int(t.shape[-2]), int(t.shape[-1])
            else:
                raise ValueError(f"{name}: unexpected shape {tuple(t.shape)}")
            if t_h == train_h and t_w == train_w:
                return None, None  # already correct shape
            if not resize_inputs:
                raise ValueError(
                    f"HYWM2GaussianTrain: {name} is {t_h}×{t_w} but the "
                    f"trainer renders at {train_w}×{train_h} (= "
                    f"images {W_img}×{H_img} / data_factor={data_factor}). "
                    f"The depth/normal loss multiplies these against the "
                    f"rendered tensor (world_gs_trainer.py:1218), so they "
                    f"must match exactly. Either set resize_inputs=True "
                    f"(default), or provide {name} pre-resized to "
                    f"{train_w}×{train_h}."
                )
            _p(f"  {name}: resize {t_h}×{t_w} -> {train_w}×{train_h} (resize_inputs=True)")
            return train_h, train_w

        if depth_loss_on:
            _p(f"writing {N_img} depths to depths/ (16-bit grayscale)...")
            d_h, d_w = _check_or_resize(depth_raw, "depth_raw")
            _dump_depths(depth_raw, gs_data / "depths", keys,
                         target_h=d_h, target_w=d_w)
        if normal_loss_on:
            _p(f"writing {N_img} normals to normals/ (8-bit RGB)...")
            n_h, n_w = _check_or_resize(normals, "normals")
            _dump_normals(normals, gs_data / "normals", keys,
                          target_h=n_h, target_w=n_w)

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
            # Module handle for the Parser monkey-patch below. CRITICAL:
            # the trainer imports Parser via `from gs.opencv import Parser`
            # (world_gs_trainer.py:35). With `worldgen_dir` on sys.path,
            # that resolves to a top-level `gs.opencv` module. Python
            # caches `gs.opencv` and `hyworld2.worldgen.gs.opencv` as
            # SEPARATE sys.modules entries (even though they're the same
            # file) — each with its own Parser class object. We MUST
            # patch the same one the trainer uses, so import via `gs.opencv`.
            import gs.opencv as _opencv_mod
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

        # Memory wins. cfg.packed enables gsplat's sparse rasterization
        # (world_gs_trainer.py:947 passes it straight to rasterization).
        # Big win for panorama inputs where each camera sees ~1/12 of the
        # sphere — 11/12 of gaussians are always out-of-frustum.
        cfg.packed = True
        # data_factor downsamples training images + scales intrinsics
        # consistently (passed at world_gs_trainer.py:649 to the Parser).
        # 1=full res, 2=half (4x memory), 4=quarter (16x memory).
        cfg.data_factor = int(data_factor)
        # CRITICAL: disable Parser's `normalize` step. With normalize=True
        # (the upstream default), Parser silently APPLIES a rigid transform
        # to `camtoworlds` via `similarity_from_cameras + align_principal_
        # _axes` (gs/opencv.py:843-880). The transform rotates+rescales the
        # camera frame to canonical axes BUT does not touch our preload
        # gaussians (which load directly via cfg.preload_gs_path). The
        # result is the cameras and gaussians end up in two different
        # world frames, causing the rendered views to "look like ghosts of
        # other views" after training (the optimizer tries to fit a
        # frame-mismatched supervision signal by smearing colors). Set
        # `no_normalize=True` to keep camtoworlds == what we wrote in
        # cameras.json — same frame as the preload gaussians.
        cfg.no_normalize = True
        _p(f"cfg.packed=True (sparse rasterization), "
           f"cfg.data_factor={cfg.data_factor} "
           f"(training at {W_img // cfg.data_factor}×{H_img // cfg.data_factor}), "
           f"cfg.no_normalize=True (Parser camtoworlds = as-written)")

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

        # cfg.test_every defaults to 32, which with HYWM2's typical N_img=12
        # would give us a 1-frame val set and noisy eval triggers. Override
        # so we get ~4 val frames across the dataset.
        try:
            test_every_override = max(2, int(N_img) // 4)
            cfg.test_every = test_every_override
            _p(f"cfg.test_every override: {test_every_override} "
               f"(N_img={N_img}, default was 32)")
        except Exception as e:
            _p(f"cfg.test_every override warning: {e}")

        # The panorama_ key rename above sends ALL frames to train (the
        # dataset's `is_pano` branch overrides the test_every rule —
        # gs/opencv.py:1061). Result: val set is empty, and the trainer's
        # eval pass crashes at line 1951 with `ellipse_time /= len(valloader)`
        # → ZeroDivisionError. Disable eval entirely. Loss is still tracked
        # via tqdm postfix during training; PSNR/SSIM metrics just aren't
        # computed (they were noisy with 0-1 val frames anyway).
        cfg.eval_steps = []
        _p("cfg.eval_steps=[] (no val frames; eval would crash)")

        # Optional appearance-only refinement toggles. These are pure cfg
        # flips — the trainer already respects both attributes natively
        # (no monkey-patching).
        if freeze_positions:
            # The means param group's lr is `cfg.means_lr * scene_scale`
            # (world_gs_trainer.py:421). Setting means_lr=0 zeroes the
            # Adam step for positions for the entire run; the
            # ExponentialLR scheduler scales from this base so it stays
            # at zero.
            cfg.means_lr = 0.0
            _p("freeze_positions=True -> cfg.means_lr = 0.0 (positions locked)")
        if freeze_count:
            # DefaultStrategy.step_post_backward early-returns at
            # `step >= self.refine_stop_iter` (gsplat/strategy/default.py
            # :162-163) and the trainer-side guard at
            # world_gs_trainer.py:1801-1805 also short-circuits. Setting
            # to 0 means densification/pruning never runs.
            try:
                cfg.strategy.refine_stop_iter = 0
                _p("freeze_count=True -> cfg.strategy.refine_stop_iter = 0 "
                   "(no densification/pruning)")
            except AttributeError:
                _p(f"freeze_count=True but cfg.strategy has no "
                   f"refine_stop_iter attribute "
                   f"(strategy={type(cfg.strategy).__name__}); toggle ignored.")

        # Step list rescale (matches __main__ behavior at world_gs_trainer.py:2601).
        try:
            cfg.adjust_steps(cfg.steps_scaler)
        except Exception as e:
            _p(f"cfg.adjust_steps warning: {e}")

        _p(f"starting trainer: data_dir={cfg.data_dir}, "
           f"result_dir={cfg.result_dir}, max_steps={cfg.max_steps}, "
           f"save_ply={cfg.save_ply}, depth_loss={cfg.depth_loss}, "
           f"normal_loss={cfg.normal_loss}, init_type=random+init_num_pts=100, "
           f"test_every={cfg.test_every}, "
           f"packed={cfg.packed}, data_factor={cfg.data_factor}, "
           f"freeze_positions={freeze_positions}, "
           f"freeze_count={freeze_count}, "
           f"start_from_preload={start_from_preload}, "
           f"debug_print_5gs={debug_print_5gs}, "
           f"preload_gs_path={cfg.preload_gs_path}")

        # ----- Run the trainer -----
        # Install tqdm throttle so we get newline-terminated progress lines
        # every 5s in the worker log (tqdm's default \\r updates are invisible
        # through line-buffered worker stderr).
        _restore_tqdm = _install_tqdm_throttle(interval_seconds=5.0)

        # Parser.scene_scale monkey-patch for panorama/coincident-camera inputs.
        # Parser computes scene_scale = max(distance_from_camera_centroid)
        # (gs/opencv.py:984-987). For PanoramaSplit inputs all N cameras
        # sit at world origin (only orientations differ) → scene_scale=0,
        # which freezes `means_lr * scene_scale = 0` and zeroes the
        # densification thresholds (world_gs_trainer.py:419-432, 717).
        # Result: optimizer can't move gaussians, compensates by growing
        # a few to engulf the scene → "all one flat color" output.
        #
        # Fix: when Parser computes scene_scale ~ 0, substitute the
        # point-cloud spread (max distance from the points-centroid).
        # That IS the natural scene scale for a panorama (radius of the
        # gaussian cloud the cameras see). No-op for video/sfm inputs
        # where cameras already have spread.
        _orig_parser_init = _opencv_mod.Parser.__init__

        def _patched_parser_init(self, *args, **kwargs):
            _orig_parser_init(self, *args, **kwargs)
            if self.scene_scale < 1e-6:
                pts_attr = getattr(self, "points", None)
                if pts_attr is not None and len(pts_attr) >= 3:
                    pts = np.asarray(pts_attr, dtype=np.float64)
                    center = pts.mean(axis=0)
                    point_scale = float(
                        np.max(np.linalg.norm(pts - center, axis=1))
                    )
                    if point_scale > 1e-6:
                        self.scene_scale = point_scale
                        print(
                            f"[HYWM2GaussianTrain] Parser.scene_scale was 0 "
                            f"(coincident cameras / panorama layout); "
                            f"overrode with point-cloud max-from-centroid "
                            f"= {point_scale:.4f}",
                            file=sys.stderr, flush=True,
                        )
                        return
                self.scene_scale = 1.0
                print(
                    "[HYWM2GaussianTrain] Parser.scene_scale was 0 and "
                    "point cloud unavailable/degenerate; defaulting to "
                    "scene_scale=1.0",
                    file=sys.stderr, flush=True,
                )

        _opencv_mod.Parser.__init__ = _patched_parser_init

        # Optional: replace upstream's "init scaffold + cat preload" with
        # "preload-only" splat construction. Also installs per-step 5gs
        # debug hook when debug_print_5gs=True. See the helper docstring
        # for the full rationale.
        import hyworld2.worldgen.world_gs_trainer as _wgt_mod
        _restore_csw_patch = None
        if start_from_preload:
            _restore_csw_patch = _install_preload_only_csw_patch(
                _wgt_mod, debug_5gs=debug_print_5gs,
            )
            _p("start_from_preload=True -> create_splats_with_optimizers "
               "patched: training starts from preload only (no scaffold)")
        elif debug_print_5gs:
            _p("debug_print_5gs=True but start_from_preload=False; the 5gs "
               "hook is only installed when the patched csw runs. Set "
               "start_from_preload=True to enable.")

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
            _opencv_mod.Parser.__init__ = _orig_parser_init
            if _restore_csw_patch is not None:
                _restore_csw_patch()
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

        return io.NodeOutput(
            str(ply_path), str(gs_result),
            first_renders_out, first_renders_train_out,
        )


NODE_CLASS_MAPPINGS = {"HYWM2GaussianTrain": HYWM2GaussianTrain}
NODE_DISPLAY_NAME_MAPPINGS = {"HYWM2GaussianTrain": "HYWM2 Train Gaussians"}
