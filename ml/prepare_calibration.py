#!/usr/bin/env python3
"""One-time script to compile calibration data from data/calib/img* folders."""

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

from ml.lib.calibration.colors import class_masks_from_colored_png
from ml.lib.calibration.extract import extract_background_embeddings, extract_samples_from_image
from ml.lib.calibration.overlays import FILTERED_TALC_OVERLAY_NAME, save_filtered_talc_overlay
from ml.lib.calibration.store import CalibrationStore
from ml.lib.constants import CALIB_BACKGROUND_KEY, CALIB_CLASS_KEYS, DEFAULT_CALIBRATION_DIR


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_block01(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def compile_calibration(
    calib_root: Path,
    output: Path,
    *,
    rgb_hist_bins: int = 32,
    random_state: int = 0,
) -> dict:
    image_dirs = sorted(p for p in calib_root.iterdir() if p.is_dir() and p.name.startswith("img"))
    if not image_dirs:
        raise FileNotFoundError(f"No img* folders under {calib_root}")

    merged_rgb: dict[str, list[np.ndarray]] = {k: [] for k in CALIB_CLASS_KEYS}
    merged_emb: dict[str, list[np.ndarray]] = {k: [] for k in CALIB_CLASS_KEYS}
    merged_bg_emb: list[np.ndarray] = []
    report: dict = {"images": {}, "source_images": []}
    overlay_dir = output / "overlays"

    for img_dir in image_dirs:
        norm_path = img_dir / "normalized.png"
        mask_path = img_dir / "user_drawn_colored.png"
        feat_path = img_dir / "block01_features.npy"
        missing = [p.name for p in (norm_path, mask_path, feat_path) if not p.exists()]
        if missing:
            report["images"][img_dir.name] = {"skipped": True, "missing": missing}
            continue

        rgb = _load_rgb(norm_path)
        mask_rgb = _load_rgb(mask_path)
        block01 = _load_block01(feat_path)
        class_masks = class_masks_from_colored_png(mask_rgb)
        overlay_meta = save_filtered_talc_overlay(
            rgb,
            class_masks,
            img_dir / FILTERED_TALC_OVERLAY_NAME,
            compiled_path=overlay_dir / f"{img_dir.name}_{FILTERED_TALC_OVERLAY_NAME}",
            random_state=random_state,
        )
        rgb_by_class, emb_by_class, img_report = extract_samples_from_image(
            rgb,
            class_masks,
            block01,
            random_state=random_state,
        )
        bg_emb = extract_background_embeddings(class_masks, block01)
        img_report["background_embedding_count"] = int(bg_emb.shape[0])
        if bg_emb.size:
            merged_bg_emb.append(bg_emb)
        if overlay_meta is not None:
            img_report["filtered_talc_overlay"] = overlay_meta
        report["images"][img_dir.name] = img_report
        report["source_images"].append(img_dir.name)
        for key in CALIB_CLASS_KEYS:
            if rgb_by_class.get(key, np.zeros((0, 3))).size:
                merged_rgb[key].append(rgb_by_class[key])
            if emb_by_class.get(key, np.zeros((0, 384))).size:
                merged_emb[key].append(emb_by_class[key])

    final_rgb = {
        k: np.vstack(v).astype(np.float32) if v else np.zeros((0, 3), np.float32)
        for k, v in merged_rgb.items()
    }
    final_emb = {
        k: np.vstack(v).astype(np.float32) if v else np.zeros((0, 384), np.float32)
        for k, v in merged_emb.items()
    }
    if merged_bg_emb:
        final_emb[CALIB_BACKGROUND_KEY] = np.vstack(merged_bg_emb).astype(np.float32)

    store = CalibrationStore(output, rgb_hist_bins=rgb_hist_bins)
    store.initialize_from_merged(final_rgb, final_emb, report["source_images"])
    report["counts"] = store.summary_counts()
    report_path = output / "prep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile calibration data from data/calib/img*")
    parser.add_argument("--calib-root", type=Path, default=_ROOT / "data/calib")
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--rgb-hist-bins", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    report = compile_calibration(
        args.calib_root,
        args.output,
        rgb_hist_bins=args.rgb_hist_bins,
        random_state=args.random_state,
    )
    print(json.dumps({"output": str(args.output), "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
