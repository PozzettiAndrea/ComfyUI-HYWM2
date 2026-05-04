"""HYWM2Reconstruct node - runs WorldMirror 2.0 multi-view reconstruction.

Stub: schema is wired up so the node graph type-checks, but execute() raises
NotImplementedError until the inference path lands. This mirrors the SAM3D
pattern where the loader returns a config-dict handle and downstream nodes
load weights lazily inside their own subprocess.
"""

import logging
from typing import Any

import torch
from comfy_api.latest import io

log = logging.getLogger("hywm2")


class HYWM2Reconstruct(io.ComfyNode):
    """
    Reconstruct depth, normals, point clouds, and 3DGS from a multi-view image batch.

    Inference logic is not yet implemented — this node currently raises
    NotImplementedError but exists to validate node-graph wiring (loader handle
    flows into reconstruct).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2Reconstruct",
            display_name="HYWM2 Reconstruct",
            category="HYWM2",
            description=(
                "Run WorldMirror 2.0 forward pass on a multi-view image batch. "
                "Provide >= 2 views in the IMAGE batch."
            ),
            inputs=[
                io.Custom("HYWM2_MODEL").Input(
                    "model",
                    tooltip="Model handle from LoadHYWM2Model",
                ),
                io.Image.Input(
                    "images",
                    tooltip="Multi-view image batch (S >= 2 views).",
                ),
                io.String.Input(
                    "prior_camera_json",
                    default="",
                    multiline=False,
                    tooltip="Optional path to a camera_params.json for prior injection.",
                ),
            ],
            outputs=[
                io.Custom("HYWM2_PREDICTIONS").Output(
                    display_name="predictions",
                    tooltip="WorldMirror predictions: depth, normals, pts3d, camera_poses, camera_intrs, splats.",
                ),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        model: Any,
        images: torch.Tensor,
        prior_camera_json: str,
    ):
        log.info(
            "HYWM2Reconstruct called: model_dir=%s, images shape=%s, prior_camera_json=%r",
            model.get("model_dir") if isinstance(model, dict) else model,
            tuple(images.shape) if hasattr(images, "shape") else None,
            prior_camera_json,
        )
        raise NotImplementedError(
            "HYWM2Reconstruct: inference not yet wired up. "
            f"Received model handle: {model}"
        )
