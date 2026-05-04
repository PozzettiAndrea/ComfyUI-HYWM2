# Bundled Example Inputs

A curated subset of the upstream WorldMirror 2.0 example scenes, copied from
`Tencent-Hunyuan/HY-World-2.0/examples/worldrecon`. These are copied to the
ComfyUI `input/` folder at startup by `prestartup_script.py` so the bundled
workflows can reference them by relative path.

| Path | Type | Notes |
|------|------|-------|
| `worldrecon/realistic/Flower/` | single image | Smallest realistic example. |
| `worldrecon/realistic/Workspace/` | multi-view | Indoor desk scene. |
| `worldrecon/realistic/Archway_Tunnel/` | multi-view | Outdoor / architecture. |
| `worldrecon/stylistic/A_Stylized_Kitchen/` | multi-view | 4 stylized indoor frames. |
| `worldrecon/stylistic/Cottage_Autumn/` | multi-view | Stylized exterior. |
| `worldrecon/stylistic/Palace/` | multi-view | Stylized architecture. |
| `worldrecon/stylistic/Cat_Girl/` | single image | Stylized character. |

Larger upstream examples (`Statue_Face`, `Park_Stone`, `Building`,
`Tree_Building`, `Landmark`) are omitted to keep the repo small; pull them
directly from the upstream repository if needed.
