"""LoadHYWM2Model node - downloads WorldMirror 2.0 checkpoint and passes config paths."""

import logging
from pathlib import Path

import torch
from comfy_api.latest import io
# NOTE: `comfy.model_management` is imported lazily inside `execute` because
# importing it triggers torch.cuda.current_device() — which crashes on
# CPU-only CI hosts where torch is built without CUDA.

log = logging.getLogger("hywm2")

try:
    from .comfy_utils import get_hywm2_models_path
except ImportError:
    from comfy_utils import get_hywm2_models_path


# HuggingFace repo for WorldMirror 2.0 checkpoints.
#
# Default mirror is the bf16 rehost (~3.27 GB, fp32-critical layers
# preserved per upstream's _collect_fp32_critical_modules contract).
# To pin against the official fp32 weights, set:
#   REPO_ID  = "tencent/HY-World-2.0"
#   SUBFOLDER = "HY-WorldMirror-2.0"
# and update EXPECTED_SIZES below to 5_053_553_272.
REPO_ID = "apozz/hy-worldmirror-2-bf16"
SUBFOLDER = ""

# Required files inside the (optional) subfolder
_PREFIX = f"{SUBFOLDER}/" if SUBFOLDER else ""
REQUIRED_FILES = [
    f"{_PREFIX}model.safetensors",
    f"{_PREFIX}config.json",
]

# Expected file sizes for verification (within 10% tolerance)
EXPECTED_SIZES = {
    f"{_PREFIX}model.safetensors": 3_273_940_974,
    f"{_PREFIX}config.json": 842,
}


class LoadHYWM2Model(io.ComfyNode):
    """
    Load HY-World 2.0 / WorldMirror 2.0 model configuration.

    Downloads checkpoints if needed and passes config paths to downstream nodes.
    Actual model loading happens lazily in inference subprocesses.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadHYWM2Model",
            display_name="(Down)Load HYWM2 Model",
            category="HYWM2",
            description="Load HY-World 2.0 / WorldMirror 2.0 configuration. Downloads checkpoints if needed.",
            inputs=[
                io.Combo.Input(
                    "precision",
                    options=["auto", "bf16", "fp32"],
                    default="auto",
                    tooltip="Inference precision. auto: bf16 on Ampere+, fp32 otherwise. WorldMirror keeps a few critical layers in fp32 even when bf16 is selected.",
                ),
                io.Int.Input(
                    "target_size",
                    default=952,
                    min=224,
                    max=2048,
                    step=14,
                    tooltip="Maximum inference resolution (longest edge). Snapped to multiples of 14 internally.",
                ),
                io.Boolean.Input(
                    "predict_camera",
                    default=True,
                    tooltip="Predict camera extrinsics + intrinsics. Off frees ~10 M params.",
                ),
                io.Boolean.Input(
                    "predict_depth",
                    default=True,
                    tooltip="Predict per-pixel z + confidence + valid-mask. Off frees ~25 M params.",
                ),
                io.Boolean.Input(
                    "predict_normals",
                    default=True,
                    tooltip="Predict per-pixel surface normals + confidence. Off frees ~25 M params.",
                ),
                io.Boolean.Input(
                    "predict_points",
                    default=True,
                    tooltip="Predict per-pixel 3D world coords + confidence. Off frees ~25 M params.",
                ),
                io.Boolean.Input(
                    "predict_gaussians",
                    default=True,
                    tooltip="Predict 3DGS attributes + run the splat renderer. Off frees ~70 M params.",
                ),
            ],
            outputs=[
                io.Custom("HYWM2_MODEL").Output(
                    display_name="model",
                    tooltip="WorldMirror 2.0 model handle (config dict). Pass to HYWM2Reconstruct.",
                ),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        precision: str,
        target_size: int,
        predict_camera: bool = True,
        predict_depth: bool = True,
        predict_normals: bool = True,
        predict_points: bool = True,
        predict_gaussians: bool = True,
    ):
        log.info("Loading HYWM2 / WorldMirror 2.0 model handle...")

        # Resolve precision "auto" using GPU capabilities. Lazy import:
        # comfy.model_management probes CUDA at module load and crashes on
        # CPU-only CI hosts (torch built without CUDA).
        if precision == "auto":
            import comfy.model_management as mm
            device = mm.get_torch_device()
            precision = "bf16" if mm.should_use_bf16(device) else "fp32"
        log.info("Precision: %s", precision)

        # The 3DGS renderer rasterizes against predictions["camera_poses"],
        # so predict_gaussians strictly requires predict_camera. Force it on
        # rather than failing later inside _gen_all_preds.
        if predict_gaussians and not predict_camera:
            log.warning(
                "predict_gaussians=True forces predict_camera=True "
                "(the 3DGS renderer reads camera_poses)."
            )
            predict_camera = True

        # Toggle → upstream "disable_heads" list (upstream uses 'normal'/'gs')
        head_flags = {
            "camera": predict_camera,
            "depth": predict_depth,
            "normal": predict_normals,
            "points": predict_points,
            "gs": predict_gaussians,
        }
        disable_heads = [name for name, on in head_flags.items() if not on]
        if disable_heads:
            log.info("Disabling prediction heads: %s", disable_heads)

        # Ensure checkpoint files are present
        model_dir = cls._get_or_download_checkpoint()

        log.info("Model handle ready (model_dir=%s)", model_dir)

        model = {
            "model_dir": str(model_dir),
            "subfolder": SUBFOLDER,
            "repo_id": REPO_ID,
            "precision": precision,
            "enable_bf16": precision == "bf16",
            "target_size": int(target_size),
            "disable_heads": disable_heads,
        }
        return io.NodeOutput(model)

    @staticmethod
    def _get_or_download_checkpoint() -> Path:
        """Ensure the WorldMirror 2.0 subfolder is present on disk; download missing files."""
        models_dir = get_hywm2_models_path()

        missing = [
            fname
            for fname in REQUIRED_FILES
            if not LoadHYWM2Model._verify_checkpoint(models_dir / fname, fname)
        ]

        if missing:
            log.info("Need to download %d file(s) from %s...", len(missing), REPO_ID)
            LoadHYWM2Model._download_files(models_dir, missing)
        else:
            log.info("All required WorldMirror 2.0 files present")

        return models_dir / SUBFOLDER

    @staticmethod
    def _verify_checkpoint(filepath: Path, filename: str) -> bool:
        """Verify a checkpoint file exists and has approximately the expected size."""
        if not filepath.exists():
            return False

        if filename.endswith(".json"):
            return filepath.stat().st_size > 0

        expected = EXPECTED_SIZES.get(filename)
        if expected:
            actual = filepath.stat().st_size
            if abs(actual - expected) > expected * 0.1:
                return False

        return True

    @staticmethod
    def _download_files(target_dir: Path, files: list):
        """Download specific files from HuggingFace, preserving subfolder layout.

        Bridges hf_hub_download's tqdm progress into ComfyUI's ProgressBar so
        the queue UI shows real-time byte-level progress for the 4.7 GB
        WorldMirror checkpoint.
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("huggingface_hub required: pip install huggingface-hub") from e

        import comfy.utils
        import tqdm as _tqdm_mod

        log.info("Downloading from HuggingFace: %s", REPO_ID)

        total_bytes = sum(EXPECTED_SIZES.get(f, 0) for f in files) or len(files)
        pbar = comfy.utils.ProgressBar(total_bytes)
        cumulative_done = 0

        class _ComfyTqdm(_tqdm_mod.tqdm):
            # Captures per-file byte progress and forwards it to the
            # outer ComfyUI bar. file_base = bytes already finished
            # before this tqdm instance started.
            file_base = 0

            def update(self, n=1):
                ret = super().update(n)
                if n:
                    pbar.update_absolute(
                        min(_ComfyTqdm.file_base + self.n, total_bytes),
                        total_bytes,
                    )
                return ret

        for filename in files:
            log.info("Downloading %s...", filename)
            _ComfyTqdm.file_base = cumulative_done
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(target_dir),
                tqdm_class=_ComfyTqdm,
            )
            cumulative_done = min(
                cumulative_done + EXPECTED_SIZES.get(filename, 0),
                total_bytes,
            )
            pbar.update_absolute(cumulative_done, total_bytes)

        log.info("Download complete")
