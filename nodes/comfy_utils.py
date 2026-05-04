"""Utility functions for HYWM2 nodes."""

import os
from pathlib import Path
import folder_paths

# Register model folder with ComfyUI's folder_paths system
_hywm2_models_dir = os.path.join(folder_paths.models_dir, "hywm2")
os.makedirs(_hywm2_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("hywm2", _hywm2_models_dir)


def get_hywm2_models_path() -> Path:
    """Get the path to HYWM2 models directory within ComfyUI models folder."""
    models_dir = Path(folder_paths.models_dir) / "hywm2"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
