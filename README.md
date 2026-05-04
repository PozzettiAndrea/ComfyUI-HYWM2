# ComfyUI-HYWM2

ComfyUI custom nodes wrapping [HY-World 2.0 / WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) (Tencent Hunyuan) for 3D reconstruction from multi-view images and video — depth, surface normals, camera parameters, dense point clouds, and 3D Gaussian Splatting in a single feed-forward pass.

## Status

Scaffolding. Two stub nodes are wired up (`LoadHYWM2Model`, `HYWM2Reconstruct`) so the pack registers cleanly; inference logic lands in a follow-up.

## Installation

Please always install from the ComfyUI Manager.

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-HYWM2/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## License

The wrapper code in this repository is GPL-3.0 (see `LICENSE`).

The HY-World 2.0 model weights and upstream code are released by Tencent Hunyuan under the **Tencent Hunyuan Community License**. Downloaded weights remain governed by that license — see [License.txt](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt).

## Acknowledgments

- [HY-World 2.0 / WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) by Tencent Hunyuan
- ComfyUI community
