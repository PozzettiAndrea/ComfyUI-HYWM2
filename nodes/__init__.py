from .load_model import LoadHYWM2Model
from .load_memory_bank import HYWM2LoadMemoryBank
from .preview_and_filter_views import HYWM2PreviewAndFilterViews
from .reconstruct import HYWM2Reconstruct
from .sample_panorama import HYWM2SamplePanorama
from .decode_export import (
    HYWM2ExportPointsPLY,
    HYWM2ExportGaussiansPLY,
    HYWM2ExportGaussiansSplat,
    HYWM2PreviewPointCloud,
)
from .ply_viewer import HYWM2PLYAdvancedGaussianViewer
from .splat_viewer import HYWM2SplatAdvancedViewer

NODE_CLASS_MAPPINGS = {
    "LoadHYWM2Model": LoadHYWM2Model,
    "HYWM2LoadMemoryBank": HYWM2LoadMemoryBank,
    "HYWM2PreviewAndFilterViews": HYWM2PreviewAndFilterViews,
    "HYWM2Reconstruct": HYWM2Reconstruct,
    "HYWM2SamplePanorama": HYWM2SamplePanorama,
    "HYWM2ExportPointsPLY": HYWM2ExportPointsPLY,
    "HYWM2ExportGaussiansPLY": HYWM2ExportGaussiansPLY,
    "HYWM2ExportGaussiansSplat": HYWM2ExportGaussiansSplat,
    "HYWM2PreviewPointCloud": HYWM2PreviewPointCloud,
    "HYWM2PLYAdvancedGaussianViewer": HYWM2PLYAdvancedGaussianViewer,
    "HYWM2SplatAdvancedViewer": HYWM2SplatAdvancedViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHYWM2Model": "(Down)Load HYWM2 Model",
    "HYWM2LoadMemoryBank": "HYWM2 Load Memory Bank",
    "HYWM2PreviewAndFilterViews": "HYWM2 Preview + Filter Views",
    "HYWM2Reconstruct": "HYWM2 Reconstruct",
    "HYWM2SamplePanorama": "HYWM2 Sample Panorama (Equirect -> Perspective)",
    "HYWM2ExportPointsPLY": "HYWM2 Export Points PLY",
    "HYWM2ExportGaussiansPLY": "HYWM2 Export Gaussians PLY",
    "HYWM2ExportGaussiansSplat": "HYWM2 Export Gaussians .splat",
    "HYWM2PreviewPointCloud": "HYWM2 Preview Point Cloud",
    "HYWM2PLYAdvancedGaussianViewer": "PLY Advanced Gaussian Viewer",
    "HYWM2SplatAdvancedViewer": "Splat Viewer",
}
