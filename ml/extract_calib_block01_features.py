#!/usr/bin/env python3
"""Extract DINO block-1 patch features for calibration images under data/calib/img*."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from ml.lib.constants import DEFAULT_DINO_REPO, DEFAULT_DINO_WEIGHTS, FALLBACK_DINO_WEIGHTS
from ml.lib.dino.inference import extract_multi_block_features
from ml.lib.dino.model import resolve_dino_weights


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def extract_block01_for_dir(
    img_dir: Path,
    *,
    device: torch.device,
    repo_dir: Path,
    weights: str,
    num_blocks: int,
    overwrite: bool,
) -> dict:
    norm_path = img_dir / "normalized.png"
    out_path = img_dir / "block01_features.npy"
    if not norm_path.exists():
        return {"dir": img_dir.name, "skipped": True, "reason": "missing normalized.png"}
    if out_path.exists() and not overwrite:
        return {"dir": img_dir.name, "skipped": True, "reason": "block01_features.npy exists"}

    rgb = _load_rgb(norm_path)
    result = extract_multi_block_features(
        rgb,
        device=device,
        block_indices=[1],
        num_blocks=num_blocks,
        repo_dir=repo_dir,
        weights=weights,
    )
    feats = result.features(1).numpy().astype(np.float32)
    np.save(out_path, feats)
    return {
        "dir": img_dir.name,
        "saved": str(out_path),
        "shape": list(feats.shape),
        "native_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract block-1 DINO features for data/calib/img* calibration folders"
    )
    parser.add_argument("--calib-root", type=Path, default=_ROOT / "data/calib")
    parser.add_argument("--repo", type=Path, default=DEFAULT_DINO_REPO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    weights_path = resolve_dino_weights(str(args.weights))
    if not weights_path and FALLBACK_DINO_WEIGHTS.exists():
        weights_path = str(FALLBACK_DINO_WEIGHTS)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    image_dirs = sorted(
        p for p in args.calib_root.iterdir() if p.is_dir() and p.name.startswith("img")
    )
    if not image_dirs:
        raise FileNotFoundError(f"No img* folders under {args.calib_root}")

    report = []
    for img_dir in image_dirs:
        report.append(
            extract_block01_for_dir(
                img_dir,
                device=device,
                repo_dir=args.repo,
                weights=weights_path,
                num_blocks=args.num_blocks,
                overwrite=args.overwrite,
            )
        )

    print(json.dumps({"device": str(device), "results": report}, indent=2))


if __name__ == "__main__":
    main()
