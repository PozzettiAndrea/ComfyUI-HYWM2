> [!WARNING]
> Warning, uses experimental package `comfy-env` to attempt a one click isolated install. Will download and use pixi package manager.



https://github.com/user-attachments/assets/9c402c05-2a99-4cb6-b131-c1035fb6eab1

# ComfyUI-HYWM2

ComfyUI custom nodes wrapping [HY-World 2.0 / WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) (Tencent Hunyuan) for 3D reconstruction from multi-view images and video — depth, surface normals, camera parameters, dense point clouds, and 3D Gaussian Splatting in a single feed-forward pass.

<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-HYWM2/">
<img src="https://pozzettiandrea.github.io/ComfyUI-HYWM2/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-HYWM2/">View Live Test Gallery →</a></b>
</div>

## Installation

Three options, in order of speed → reliability:

1. **ComfyUI Manager (recommended)** — search for `HYWM2` in the Manager and click Install from the highest version displayed. If that doesn't work, try nightly.
2. **Manager via Git URL** — in ComfyUI Manager: "Install via Git URL" with `https://github.com/PozzettiAndrea/ComfyUI-HYWM2.git`.
3. **Manual (most reliable)**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PozzettiAndrea/ComfyUI-HYWM2.git
   cd ComfyUI-HYWM2
   pip install -r requirements.txt --upgrade
   python install.py
   ```

> **Please report any problems** you hit during installation or use of my nodes — open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-HYWM2/discussions) or [Issue](https://github.com/PozzettiAndrea/ComfyUI-HYWM2/issues). Very grateful for your help! 🙏

---

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-HYWM2/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## License

The wrapper code in this repository is GPL-3.0 (see `LICENSE`).

The HY-World 2.0 model weights and upstream code are released by Tencent Hunyuan under the **Tencent HY-WORLD 2.0 Community License**. Downloaded weights remain governed by that license — see [License.txt](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt). The license restricts distribution to the Territory (worldwide excluding EU, UK, South Korea).

## Acknowledgments

- [HY-World 2.0 / WorldMirror 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) by Tencent Hunyuan
- [`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D) for the bundled 3DGS viewer
- ComfyUI community
- All contributors

## Attribution

This package vendors the inference code from **HY-World 2.0** under `nodes/hyworld2/`:

- **Original repository**: <https://github.com/Tencent-Hunyuan/HY-World-2.0>
- **License**: Tencent HY-WORLD 2.0 Community License (see `LICENSE`)
- **Authors**: Tencent Hunyuan team
- **Paper**: [HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds](https://3d-models.hunyuan.tencent.com/world/world2_0/HY_World_2_0.pdf)

The vendored code is redistributed under the terms of the Tencent HY-WORLD 2.0 Community License, which permits use, reproduction, distribution, modification, and creation of derivative works within the Territory. The bf16 weight mirror at [`apozz/hy-worldmirror-2-bf16`](https://huggingface.co/apozz/hy-worldmirror-2-bf16) is a Model Derivative under §3.b of that license.

We gratefully acknowledge Tencent Hunyuan for making HY-World 2.0 available to the research community.
