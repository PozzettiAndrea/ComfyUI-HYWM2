"""LoadHYWM2Model node - downloads WorldMirror 2.0 checkpoint and passes config paths."""

import logging
from pathlib import Path

import torch
import comfy.model_management as mm
from comfy_api.latest import io

log = logging.getLogger("hywm2")

try:
    from .comfy_utils import get_hywm2_models_path
except ImportError:
    from comfy_utils import get_hywm2_models_path


# HuggingFace repo for WorldMirror 2.0 checkpoints
REPO_ID = "tencent/HY-World-2.0"
SUBFOLDER = "HY-WorldMirror-2.0"

# Required files inside the subfolder
REQUIRED_FILES = [
    f"{SUBFOLDER}/model.safetensors",
    f"{SUBFOLDER}/config.json",
]

# Expected file sizes for verification (within 10% tolerance)
EXPECTED_SIZES = {
    f"{SUBFOLDER}/model.safetensors": 5_053_553_272,
    f"{SUBFOLDER}/config.json": 842,
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
                io.String.Input(
                    "disable_heads",
                    default="",
                    multiline=False,
                    tooltip="Comma-separated heads to disable to save memory. Options: camera, depth, normal, points, gs.",
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
        disable_heads: str,
    ):
        log.info("Loading HYWM2 / WorldMirror 2.0 model handle...")

        # Resolve precision "auto" using GPU capabilities
        if precision == "auto":
            device = mm.get_torch_device()
            precision = "bf16" if mm.should_use_bf16(device) else "fp32"
        log.info("Precision: %s", precision)

        # Parse disable_heads
        heads = [h.strip() for h in disable_heads.split(",") if h.strip()]
        valid_heads = {"camera", "depth", "normal", "points", "gs"}
        invalid = [h for h in heads if h not in valid_heads]
        if invalid:
            raise ValueError(
                f"Invalid head names in disable_heads: {invalid}. "
                f"Valid options: {sorted(valid_heads)}"
            )

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
            "disable_heads": heads,
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
        """Download specific files from HuggingFace, preserving subfolder layout."""
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("huggingface_hub required: pip install huggingface-hub") from e

        log.info("Downloading from HuggingFace: %s", REPO_ID)

        for filename in files:
            log.info("Downloading %s...", filename)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

        log.info("Download complete")
