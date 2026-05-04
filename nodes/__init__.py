from .load_model import LoadHYWM2Model
from .reconstruct import HYWM2Reconstruct
from .decode_export import (
    HYWM2DecodeDepth,
    HYWM2DecodeNormals,
    HYWM2DecodePoints,
    HYWM2DecodeGaussians,
    HYWM2ExportPointsPLY,
    HYWM2ExportGaussiansPLY,
    HYWM2PreviewPointCloud,
)

NODE_CLASS_MAPPINGS = {
    "LoadHYWM2Model": LoadHYWM2Model,
    "HYWM2Reconstruct": HYWM2Reconstruct,
    "HYWM2DecodeDepth": HYWM2DecodeDepth,
    "HYWM2DecodeNormals": HYWM2DecodeNormals,
    "HYWM2DecodePoints": HYWM2DecodePoints,
    "HYWM2DecodeGaussians": HYWM2DecodeGaussians,
    "HYWM2ExportPointsPLY": HYWM2ExportPointsPLY,
    "HYWM2ExportGaussiansPLY": HYWM2ExportGaussiansPLY,
    "HYWM2PreviewPointCloud": HYWM2PreviewPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHYWM2Model": "(Down)Load HYWM2 Model",
    "HYWM2Reconstruct": "HYWM2 Reconstruct",
    "HYWM2DecodeDepth": "HYWM2 Decode Depth",
    "HYWM2DecodeNormals": "HYWM2 Decode Normals",
    "HYWM2DecodePoints": "HYWM2 Decode Points",
    "HYWM2DecodeGaussians": "HYWM2 Decode Gaussians",
    "HYWM2ExportPointsPLY": "HYWM2 Export Points PLY",
    "HYWM2ExportGaussiansPLY": "HYWM2 Export Gaussians PLY",
    "HYWM2PreviewPointCloud": "HYWM2 Preview Point Cloud",
}
