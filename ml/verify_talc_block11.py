#!/usr/bin/env python3
"""Verify block-11 talc embedding margin on labeled artifact sample regions."""

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

from ml.lib.calibration.colors import class_masks_from_colored_png
from ml.lib.calibration.store import CalibrationStore
from ml.lib.calibration.talc_threshold import (
    resolve_min_region_mean_margin,
    resolve_talc_intensity_max,
    resolve_talc_matrix_mean_embedding,
    resolve_talc_mean_embedding,
    resolve_talc_min_cosine_margin,
)
from ml.lib.constants import CLS_MATRIX, CLS_TALC, DEFAULT_CALIBRATION_DIR, TALC_EMBEDDING_BLOCK
from ml.lib.dino.inference import extract_multi_block_features
from ml.lib.dino.model import resolve_dino_weights
from ml.lib.segmentation.talc_intensity import segment_talc_hybrid
from ml.lib.types import SegmentConfig


def _ensure_block11(feat_path: Path, rgb: np.ndarray, *, device: torch.device) -> np.ndarray:
    if feat_path.exists():
        return np.load(feat_path).astype(np.float32)
    weights = resolve_dino_weights(str(_ROOT / "data/models/checkpoints/dinov2_vits14_reg4_pretrain.pth"))
    result = extract_multi_block_features(
        rgb,
        device=device,
        block_indices=[TALC_EMBEDDING_BLOCK],
        num_blocks=12,
        repo_dir=_ROOT / "data/models/dinov2",
        weights=weights,
    )
    feats = result.features(TALC_EMBEDDING_BLOCK).numpy().astype(np.float32)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, feats)
    return feats


def verify_artifacts(
    artifacts_root: Path,
    calibration_dir: Path,
    *,
    device: torch.device,
) -> dict:
    calib = CalibrationStore(calibration_dir).get()
    talc_mean = resolve_talc_mean_embedding(calib)
    matrix_mean = resolve_talc_matrix_mean_embedding(calib)
    if talc_mean is None or matrix_mean is None:
        raise RuntimeError("compiled calibration missing talc/matrix mean embeddings")

    cfg = SegmentConfig()
    min_margin = resolve_talc_min_cosine_margin(calib, cfg.talc_min_cosine_margin)
    intensity_max = resolve_talc_intensity_max(calib, cfg.talc_black_max)
    min_region = resolve_min_region_mean_margin(calib, min_margin)

    per_image: dict[str, dict] = {}
    for artifact_dir in sorted(p for p in artifacts_root.iterdir() if p.is_dir()):
        manual = artifact_dir / "manual" / "user_drawn_colored.png"
        norm = artifact_dir / "images" / "normalized.png"
        feat = artifact_dir / "dino" / "block11_features.npy"
        if not manual.exists() or not norm.exists():
            continue
        rgb = np.asarray(Image.open(norm).convert("RGB"))
        masks = class_masks_from_colored_png(np.asarray(Image.open(manual).convert("RGB")))
        if CLS_TALC not in masks or CLS_MATRIX not in masks:
            continue
        block11 = _ensure_block11(feat, rgb, device=device)
        fg = np.zeros(rgb.shape[:2], dtype=bool)
        talc_mask, meta = segment_talc_hybrid(
            rgb,
            fg,
            torch.from_numpy(block11),
            talc_mean,
            matrix_mean,
            fg_dilate_radius=0,
            talc_black_max=intensity_max,
            min_cosine_margin=min_margin,
            min_region_mean_margin=min_region,
        )
        pred = talc_mask.astype(bool)
        talc_sample = masks[CLS_TALC]
        bg_sample = masks[CLS_MATRIX]
        textured_talc = talc_sample  # sample-level eval only
        inter = int((pred & textured_talc).sum())
        bg_fp = int((pred & bg_sample).sum())
        per_image[artifact_dir.name] = {
            "pred_pixels": int(pred.sum()),
            "sample_talc_pixels": int(talc_sample.sum()),
            "sample_bg_pixels": int(bg_sample.sum()),
            "sample_recall": inter / max(int(talc_sample.sum()), 1),
            "sample_bg_fp_rate": bg_fp / max(int(pred.sum()), 1),
            "min_cosine_margin": min_margin,
            "talc_meta": meta,
        }

    return {
        "embedding_block": TALC_EMBEDDING_BLOCK,
        "calibration_dir": str(calibration_dir),
        "artifacts_root": str(artifacts_root),
        "per_image": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify block-11 talc margin on artifact labels")
    parser.add_argument("--artifacts-root", type=Path, default=_ROOT / "data/artifacts")
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--device", type=str, default="cpu", help="DINO device for block-11 extraction")
    args = parser.parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    report = verify_artifacts(args.artifacts_root, args.calibration_dir, device=device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
