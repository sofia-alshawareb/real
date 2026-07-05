#!/usr/bin/env python3
"""Calibrate talc embedding cosine thresholds from labeled sample regions in data/artifacts/."""

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
    calibrate_talc_from_labeled_images,
    resolve_talc_matrix_mean_embedding,
    resolve_talc_mean_embedding,
)
from ml.lib.constants import CLS_MATRIX, CLS_TALC, DEFAULT_CALIBRATION_DIR, DEFAULT_TALC_MIN_COSINE, TALC_EMBEDDING_BLOCK
from ml.lib.dino.inference import extract_multi_block_features
from ml.lib.dino.model import resolve_dino_weights


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


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def discover_artifact_samples(
    artifacts_root: Path,
    *,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, dict, str]]:
    """Load artifact folders with manual labels + block-11 features."""
    samples: list[tuple[np.ndarray, np.ndarray, dict, str]] = []
    for artifact_dir in sorted(p for p in artifacts_root.iterdir() if p.is_dir()):
        norm_path = artifact_dir / "images" / "normalized.png"
        manual_path = artifact_dir / "manual" / "user_drawn_colored.png"
        feat_path = artifact_dir / "dino" / "block11_features.npy"
        if not norm_path.exists() or not manual_path.exists():
            continue
        rgb = _load_rgb(norm_path)
        class_masks = class_masks_from_colored_png(_load_rgb(manual_path))
        if CLS_TALC not in class_masks or CLS_MATRIX not in class_masks:
            continue
        features = _ensure_block11(feat_path, rgb, device=device)
        samples.append((rgb, features, class_masks, artifact_dir.name))
    return samples


def calibrate_from_artifacts(
    artifacts_root: Path,
    calibration_dir: Path,
    *,
    merge: bool = True,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = discover_artifact_samples(artifacts_root, device=device)
    if not samples:
        raise FileNotFoundError(
            f"No labeled artifacts under {artifacts_root} "
            "(need manual/user_drawn_colored.png + dino/block11_features.npy + blue/grey samples)"
        )

    store = CalibrationStore(calibration_dir)
    reference_mean = None
    reference_matrix = None
    if store.exists():
        calib_data = store.get()
        reference_mean = resolve_talc_mean_embedding(calib_data)
        reference_matrix = resolve_talc_matrix_mean_embedding(calib_data)

    talc_contour = calibrate_talc_from_labeled_images(
        samples,
        fallback_cosine=DEFAULT_TALC_MIN_COSINE,
        reference_mean_embedding=reference_mean,
        reference_matrix_mean_embedding=reference_matrix,
    )
    talc_contour["artifact_ids"] = [s[3] for s in samples]

    if merge and store.exists():
        store.update_talc_contour(talc_contour)
    elif store.exists():
        data = store.get()
        data.meta["talc_contour"] = talc_contour
        store.save(data)
    else:
        store.initialize_from_merged(
            {k: np.zeros((0, 3), np.float32) for k in ("coarse", "fine", "talc", "matrix")},
            {k: np.zeros((0, 384), np.float32) for k in ("coarse", "fine", "talc", "matrix")},
            [s[3] for s in samples],
            talc_contour=talc_contour,
        )

    report = {
        "artifacts_root": str(artifacts_root),
        "calibration_dir": str(calibration_dir),
        "n_samples": len(samples),
        "artifact_ids": [s[3] for s in samples],
        "talc_contour": talc_contour,
    }
    report_path = calibration_dir / "talc_calib_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate talc embedding thresholds from data/artifacts sample labels"
    )
    parser.add_argument("--artifacts-root", type=Path, default=_ROOT / "data/artifacts")
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Replace talc_contour even when compiled calibration is empty",
    )
    parser.add_argument("--device", type=str, default="cpu", help="DINO device for block-11 extraction")
    args = parser.parse_args()
    device = torch.device(args.device)
    report = calibrate_from_artifacts(
        args.artifacts_root,
        args.calibration_dir,
        merge=not args.no_merge,
        device=device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
