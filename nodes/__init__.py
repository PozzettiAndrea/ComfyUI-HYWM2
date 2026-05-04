from .load_model import LoadHYWM2Model
from .reconstruct import HYWM2Reconstruct

NODE_CLASS_MAPPINGS = {
    "LoadHYWM2Model": LoadHYWM2Model,
    "HYWM2Reconstruct": HYWM2Reconstruct,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHYWM2Model": "(Down)Load HYWM2 Model",
    "HYWM2Reconstruct": "HYWM2 Reconstruct",
}
