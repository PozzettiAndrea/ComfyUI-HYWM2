"""HYWM2SamplePanorama — sample perspective cutouts from an equirectangular
panorama for downstream WorldMirror reconstruction.

Ported from ComfyUI-Sharp's SamplePanorama node. Default output size is 952
to line up with WorldMirror 2.0's native target_size; results get snapped to
multiples of 14 (ViT patch size) downstream by the Reconstruct node.
"""

import logging
import math

import torch
import torch.nn.functional as F

from comfy_api.latest import io

log = logging.getLogger("hywm2")


def _create_rotation_matrix(yaw: float, pitch: float) -> torch.Tensor:
    """3x3 camera-to-world rotation from yaw (Y) + pitch (X)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    R_yaw = torch.tensor([
        [cy,  0, sy],
        [0,   1,  0],
        [-sy, 0, cy],
    ], dtype=torch.float32)
    cp, sp = math.cos(pitch), math.sin(pitch)
    R_pitch = torch.tensor([
        [1,  0,   0],
        [0,  cp, -sp],
        [0,  sp,  cp],
    ], dtype=torch.float32)
    return R_yaw @ R_pitch


def _sample_perspective_from_equirectangular(
    panorama: torch.Tensor,
    yaw: float,
    pitch: float,
    fov_radians: float,
    output_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a square perspective view from an equirectangular panorama.

    Returns (image[H,W,3], extrinsics_w2c[4,4], intrinsics[4,4]).
    """
    if panorama.dim() == 4:
        panorama = panorama[0]
    H, W, _ = panorama.shape
    device = panorama.device

    f_px = (output_size / 2) / math.tan(fov_radians / 2)
    cx = (output_size - 1) / 2
    cy = (output_size - 1) / 2

    u = torch.arange(output_size, dtype=torch.float32, device=device)
    v = torch.arange(output_size, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    dx = (uu - cx) / f_px
    dy = (vv - cy) / f_px
    dz = torch.ones_like(dx)
    rays_cam = F.normalize(torch.stack([dx, dy, dz], dim=-1), dim=-1)

    R = _create_rotation_matrix(yaw, pitch).to(device)
    rays_world = torch.einsum("ij,hwj->hwi", R, rays_cam)

    rx, ry, rz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    ray_yaw = torch.atan2(rx, rz)
    ray_pitch = torch.asin(torch.clamp(ry, -1, 1))

    eq_x = (ray_yaw / math.pi + 1) * (W - 1) / 2
    eq_y = (0.5 - ray_pitch / math.pi) * (H - 1)
    grid = torch.stack([eq_x / (W - 1) * 2 - 1, eq_y / (H - 1) * 2 - 1], dim=-1).unsqueeze(0)

    pano_nchw = panorama.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(pano_nchw, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    perspective = sampled[0].permute(1, 2, 0)

    # Pinhole intrinsics in pixel coords (3x3, CameraPack convention).
    intrinsics = torch.tensor([
        [f_px, 0,    cx],
        [0,    f_px, cy],
        [0,    0,    1 ],
    ], dtype=torch.float32, device=device)

    # World-to-camera extrinsics (CameraPack convention).
    extrinsics = torch.eye(4, dtype=torch.float32, device=device)
    extrinsics[:3, :3] = R.T
    return perspective, extrinsics, intrinsics


class HYWM2SamplePanorama(io.ComfyNode):
    """Sample perspective cutouts from an equirectangular panorama."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYWM2SamplePanorama",
            display_name="HYWM2 Sample Panorama (Equirect -> Perspective)",
            category="HYWM2",
            description=(
                "Cover a 360°×180° equirectangular panorama with overlapping "
                "perspective crops. Default output_size=952 lines up with "
                "WorldMirror 2.0's target_size (snapped to multiples of 14 "
                "downstream)."
            ),
            inputs=[
                io.Image.Input("panorama"),
                io.Float.Input("fov_degrees", default=65.0, min=30.0, max=120.0, step=1.0,
                               tooltip="Field of view in degrees per perspective cutout."),
                io.Float.Input("overlap_percent", default=10.0, min=0.0, max=50.0, step=1.0,
                               tooltip="Overlap between adjacent samples as %% of FOV."),
                io.Int.Input("output_size", default=952, min=224, max=2048, step=14,
                             tooltip="Square output resolution. 952 matches WorldMirror's default."),
                io.Boolean.Input("skip_poles", default=True, optional=True,
                                 tooltip="Skip samples pointing near zenith/nadir (often low quality)."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.Int.Output(display_name="num_horizontal"),
                io.Int.Output(display_name="num_vertical"),
            ],
        )

    @classmethod
    def execute(
        cls,
        panorama: torch.Tensor,
        fov_degrees: float = 65.0,
        overlap_percent: float = 10.0,
        output_size: int = 952,
        skip_poles: bool = True,
    ):
        fov_radians = math.radians(fov_degrees)
        step_degrees = fov_degrees * (1 - overlap_percent / 100)

        num_horizontal = math.ceil(360 / step_degrees)
        if skip_poles:
            vertical_range = 150
            vertical_start = -75
        else:
            vertical_range = 180
            vertical_start = -90
        num_vertical = max(1, math.ceil(vertical_range / step_degrees))

        log.info("HYWM2SamplePanorama: FOV=%.1f° overlap=%.1f%% step=%.1f°",
                 fov_degrees, overlap_percent, step_degrees)
        log.info("HYWM2SamplePanorama: %d horizontal × %d vertical = %d crops",
                 num_horizontal, num_vertical, num_horizontal * num_vertical)

        if panorama.dim() == 3:
            panorama = panorama.unsqueeze(0)
        pano = panorama[0]

        all_images, all_extrinsics = [], []
        intrinsics = None

        for v_idx in range(num_vertical):
            pitch = math.radians(vertical_start + (v_idx + 0.5) * step_degrees)
            for h_idx in range(num_horizontal):
                yaw = math.radians(-180 + (h_idx + 0.5) * step_degrees)
                img, extr, intr = _sample_perspective_from_equirectangular(
                    pano, yaw, pitch, fov_radians, output_size,
                )
                all_images.append(img)
                all_extrinsics.append(extr)
                if intrinsics is None:
                    intrinsics = intr

        images_batch = torch.stack(all_images, dim=0)
        extrinsics_batch = torch.stack(all_extrinsics, dim=0)

        log.info("HYWM2SamplePanorama: output shape %s", tuple(images_batch.shape))

        return io.NodeOutput(images_batch, extrinsics_batch, intrinsics,
                             num_horizontal, num_vertical)
