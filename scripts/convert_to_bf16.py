"""One-shot script: cast tencent/HY-World-2.0 -> HY-WorldMirror-2.0/model.safetensors
to bf16 for HF rehosting. Halves on-disk size from ~4.7 GB → ~2.4 GB.

Usage:
    python scripts/convert_to_bf16.py \
        --src /path/to/HY-WorldMirror-2.0/model.safetensors \
        --dst /path/to/output/model.safetensors

Then upload `dst` + `config.json` + a copy of upstream `License.txt` + a
README noting the modification (per Tencent HY-WORLD 2.0 Community License
§3.a-c) to a HuggingFace mirror, e.g. `apozz/hy-worldmirror-2-bf16`.

Layers kept in fp32 (numerically critical, see upstream
`pipeline._collect_fp32_critical_modules`):
  * MlpFP32.fc2  (matched by name suffix `.fc2.weight` / `.fc2.bias`
                  inside any `mlp_fp32` block — fall back to keep ANY
                  fc2 belonging to a module named *MlpFP32* in fp32)
  * scratch.output_conv2  (matched by `.scratch.output_conv2.*`)

Anything not in those name patterns is cast bf16 if it's floating point,
left untouched if integral.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# Keys to keep in fp32. Match by *substring* on the flat key name.
FP32_KEY_PATTERNS = [
    r"\.fc2\.(weight|bias)$",            # MlpFP32 final projection (~rare)
    r"\.scratch\.output_conv2\.",        # DPT head's final dense conv
]


def _is_fp32_key(key: str) -> bool:
    return any(re.search(p, key) for p in FP32_KEY_PATTERNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="Source model.safetensors (fp32)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Destination model.safetensors (bf16 + selective fp32)")
    parser.add_argument("--keep-fp32-extra", nargs="*", default=[],
                        help="Additional substring patterns to keep fp32.")
    args = parser.parse_args()

    if not args.src.is_file():
        raise SystemExit(f"src not found: {args.src}")
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    patterns = list(FP32_KEY_PATTERNS) + list(args.keep_fp32_extra or [])
    print(f"[convert] loading {args.src} ({args.src.stat().st_size / 1e9:.2f} GB)")
    state = load_file(str(args.src))

    cast = 0
    kept = 0
    skipped_int = 0
    new_state: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not v.is_floating_point():
            new_state[k] = v
            skipped_int += 1
            continue
        if any(re.search(p, k) for p in patterns):
            new_state[k] = v.to(torch.float32) if v.dtype != torch.float32 else v
            kept += 1
        else:
            new_state[k] = v.to(torch.bfloat16)
            cast += 1

    print(f"[convert] cast {cast} tensors to bf16, kept {kept} in fp32, "
          f"left {skipped_int} integral tensors untouched")

    print(f"[convert] writing {args.dst}")
    save_file(new_state, str(args.dst))
    print(f"[convert] done. dst size: {args.dst.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
