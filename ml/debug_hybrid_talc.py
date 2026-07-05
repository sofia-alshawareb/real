#!/usr/bin/env python3
"""Save hybrid talc pipeline masks step-by-step for local debugging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.morphology import dilation, disk

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from ml.lib.calibration.store import CalibrationStore
from ml.lib.calibration.filters import rgb_to_gray
from ml.lib.calibration.talc_threshold import resolve_talc_intensity_max
from ml.lib.constants import (
    CLASS_COLORS,
    CLS_COARSE,
    CLS_FINE,
    CLS_TALC,
    COARSE_FINE_DINO_BLOCK,
    DEFAULT_CALIBRATION_DIR,
    TALC_GMM_MIN_ACTIVATION,
    TALC_REFINE_MODE_DINO,
    TALC_REFINE_MODE_GRADIENT,
)
from ml.lib.dino.inference import extract_multi_block_features
from ml.lib.dino.model import resolve_dino_weights
from ml.lib.segmentation.calibrated import segment_hybrid
from ml.lib.segmentation.intensity_fg import segment_coarse_fine_intensity
from ml.lib.segmentation.gmm_histogram import save_two_gmm_histogram
from ml.lib.segmentation.talc_intensity import (
    _fill_talc_connected_components,
    _interior_gmm_fit_mask,
    _talc_band_from_two_gmm_activation,
    _talc_bands_from_gradient_gmm,
    compute_talc_refine_gradient_map,
    refine_talc_with_block01_activation,
    refine_talc_with_image_gradient,
    segment_talc_intensity_coarse,
)
from ml.lib.types import SegmentConfig
from services.ml_worker.config import load_config


def _overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    if not np.any(m):
        return np.clip(out, 0, 255).astype(np.uint8)
    color_arr = np.array(color, dtype=np.float32)
    for ch in range(3):
        out[..., ch][m] = (1.0 - alpha) * out[..., ch][m] + alpha * color_arr[ch]
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_mask_png(path: Path, mask: np.ndarray, rgb: np.ndarray, color: tuple[int, int, int]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_overlay_mask(rgb, mask, color)).save(path)
    return int(mask.astype(bool).sum())


def _save_gray_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        norm = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros(arr.shape, dtype=np.uint8)
    Image.fromarray(norm, mode="L").save(path)


def _save_labels_png(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        rgb[labels == cls_id] = color
    Image.fromarray(rgb).save(path)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _artifact_id_from_image(image_path: Path) -> str | None:
    """Return artifact id when image lives under data/artifacts/<id>/images/."""
    p = image_path.resolve()
    if p.parent.name == "images" and p.parent.parent.parent.name == "artifacts":
        return p.parent.parent.name
    return None


def _debug_run_dir(out_dir: Path, image_path: Path) -> Path:
    """Stable output folder: artifact id when available, else image stem."""
    artifact_id = _artifact_id_from_image(image_path)
    return out_dir / (artifact_id or image_path.stem)


def _default_block01_activation_path(image_path: Path) -> Path | None:
    p = image_path.resolve()
    if p.parent.name != "images":
        return None
    candidate = p.parent.parent / "dino" / "block01_activation.npy"
    return candidate if candidate.exists() else None


def _load_or_run_block01_activation(
    rgb: np.ndarray,
    activation_path: Path | None,
    *,
    device: torch.device,
) -> np.ndarray:
    if activation_path is not None and activation_path.exists():
        return np.load(activation_path).astype(np.float32)
    weights = resolve_dino_weights(str(_ROOT / "data/models/checkpoints/dinov2_vits14_reg4_pretrain.pth"))
    result = extract_multi_block_features(
        rgb,
        device=device,
        block_indices=[COARSE_FINE_DINO_BLOCK],
        num_blocks=12,
        repo_dir=_ROOT / "data/models/dinov2",
        weights=weights,
    )
    act = result.activation(COARSE_FINE_DINO_BLOCK)
    if activation_path is not None:
        activation_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(activation_path, act)
    return act


def run_debug(
    image_path: Path,
    out_dir: Path,
    *,
    calibration_dir: Path,
    config_path: Path | None,
    activation_path: Path | None,
    device: torch.device,
    talc_refine_mode: str | None = None,
) -> dict[str, Any]:
    rgb = _load_rgb(image_path)
    act = _load_or_run_block01_activation(rgb, activation_path, device=device)
    calib = CalibrationStore(calibration_dir).get()

    svc_cfg = load_config(config_path).segmentation if config_path else load_config().segmentation
    cfg = SegmentConfig(
        segmentation_mode=svc_cfg.mode,
        max_samples=svc_cfg.max_samples,
        random_state=svc_cfg.random_state,
        close_radius=svc_cfg.close_radius,
        preprocess=svc_cfg.preprocess,
        denoise=svc_cfg.denoise,
        illum_sigma=svc_cfg.illum_sigma,
        region_overlap=svc_cfg.region_overlap,
        fg_dilate_radius=svc_cfg.fg_dilate_radius,
        talc_refine_fg_dilate_radius=svc_cfg.talc_refine_fg_dilate_radius,
        talc_gmm_fg_buffer_radius=svc_cfg.talc_gmm_fg_buffer_radius,
        talc_gmm_gate_erode=svc_cfg.talc_gmm_gate_erode,
        talc_black_max=svc_cfg.talc_black_max,
        talc_contour_dilate=svc_cfg.talc_contour_dilate,
        talc_block01_overlap=svc_cfg.talc_block01_overlap,
        talc_gmm_threshold_high_bias=svc_cfg.talc_gmm_threshold_high_bias,
        talc_refine_mode=talc_refine_mode or svc_cfg.talc_refine_mode,
    )
    intensity_max = resolve_talc_intensity_max(calib, cfg.talc_black_max)

    run_dir = _debug_run_dir(out_dir, image_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(run_dir / "00_input_rgb.png")

    fg_object, partitions, fg_mask, fg_meta = segment_coarse_fine_intensity(
        rgb,
        act,
        preprocess=cfg.preprocess,
        denoise=cfg.denoise,
        illum_sigma=cfg.illum_sigma,
        max_samples=cfg.max_samples,
        random_state=cfg.random_state,
        region_overlap=cfg.region_overlap,
        close_radius=cfg.close_radius,
    )

    fg_dilated_coarse = fg_mask.astype(bool)
    if cfg.fg_dilate_radius > 0:
        fg_dilated_coarse = dilation(fg_dilated_coarse, disk(int(cfg.fg_dilate_radius)))

    fg_dilated_refine = fg_mask.astype(bool)
    if cfg.talc_refine_fg_dilate_radius > 0:
        fg_dilated_refine = dilation(
            fg_dilated_refine, disk(int(cfg.talc_refine_fg_dilate_radius))
        )

    coarse, coarse_meta = segment_talc_intensity_coarse(
        rgb,
        fg_mask.astype(bool),
        fg_dilate_radius=cfg.fg_dilate_radius,
        talc_intensity_max=intensity_max,
        close_radius=cfg.close_radius,
    )

    rough = coarse.astype(bool)
    if cfg.talc_contour_dilate > 0:
        rough = dilation(rough, disk(int(cfg.talc_contour_dilate)))

    gmm_fit_mask, gmm_fit_meta = _interior_gmm_fit_mask(
        rough,
        fg_mask.astype(bool),
        fg_buffer_radius=cfg.talc_gmm_fg_buffer_radius,
        gate_erode_radius=cfg.talc_gmm_gate_erode,
    )

    talc_band_filled: np.ndarray | None = None
    background_band: np.ndarray | None = None
    refined_pre_clip: np.ndarray
    hist_meta: dict[str, Any]
    value_map: np.ndarray
    hist_xlabel: str
    hist_title: str
    hist_low_label: str
    hist_high_label: str

    if cfg.talc_refine_mode == TALC_REFINE_MODE_DINO:
        talc_band, gmm_meta, gmm_fit_used = _talc_band_from_two_gmm_activation(
            act,
            rgb_to_gray(rgb),
            region_mask=rough,
            gmm_fit_mask=gmm_fit_mask,
            fg_mask=fg_mask.astype(bool),
            max_samples=cfg.max_samples,
            random_state=cfg.random_state,
        )
        background_band = rough & ~talc_band & (act >= TALC_GMM_MIN_ACTIVATION)
        refined_pre_clip = talc_band.copy()
        refined, refine_meta = refine_talc_with_block01_activation(
            coarse,
            act,
            rgb_to_gray(rgb),
            max_samples=cfg.max_samples,
            random_state=cfg.random_state,
            close_radius=cfg.close_radius,
            gate_dilate_radius=cfg.talc_contour_dilate,
            fg_mask=fg_mask.astype(bool),
            fg_dilate_radius=cfg.talc_refine_fg_dilate_radius,
            gmm_fg_buffer_radius=cfg.talc_gmm_fg_buffer_radius,
            gmm_gate_erode_radius=cfg.talc_gmm_gate_erode,
        )
        value_map = act
        hist_xlabel = "block-1 activation"
        hist_title = "Talc refine: 2-GMM on interior activation"
        hist_low_label = "low activation → talc"
        hist_high_label = "high activation → background"
    elif cfg.talc_refine_mode == TALC_REFINE_MODE_GRADIENT:
        gradient, _gray = compute_talc_refine_gradient_map(
            rgb,
            preprocess=cfg.preprocess,
            denoise=cfg.denoise,
            illum_sigma=cfg.illum_sigma,
        )
        talc_band, background_band, gmm_meta, gmm_fit_used = _talc_bands_from_gradient_gmm(
            gradient,
            region_mask=rough,
            gmm_fit_mask=gmm_fit_mask,
            fg_mask=fg_mask.astype(bool),
            max_samples=cfg.max_samples,
            random_state=cfg.random_state,
            threshold_high_bias=cfg.talc_gmm_threshold_high_bias,
        )
        talc_band_filled, _n_filled = _fill_talc_connected_components(talc_band)
        refined_pre_clip = talc_band_filled.copy()
        refined, refine_meta = refine_talc_with_image_gradient(
            coarse,
            rgb,
            preprocess=cfg.preprocess,
            denoise=cfg.denoise,
            illum_sigma=cfg.illum_sigma,
            overlap_threshold=cfg.talc_block01_overlap,
            max_samples=cfg.max_samples,
            random_state=cfg.random_state,
            close_radius=cfg.close_radius,
            gate_dilate_radius=cfg.talc_contour_dilate,
            fg_mask=fg_mask.astype(bool),
            fg_dilate_radius=cfg.talc_refine_fg_dilate_radius,
            gmm_fg_buffer_radius=cfg.talc_gmm_fg_buffer_radius,
            gmm_gate_erode_radius=cfg.talc_gmm_gate_erode,
            gmm_threshold_high_bias=cfg.talc_gmm_threshold_high_bias,
        )
        value_map = gradient
        hist_xlabel = "Sobel gradient magnitude"
        hist_title = "Talc refine: 2-GMM on interior gradient"
        hist_low_label = "low gradient → background"
        hist_high_label = "high gradient → talc"
    else:
        raise ValueError(
            f"Unknown talc_refine_mode {cfg.talc_refine_mode!r}; "
            f"expected {TALC_REFINE_MODE_DINO!r} or {TALC_REFINE_MODE_GRADIENT!r}"
        )

    labels, hybrid_meta = segment_hybrid(rgb, act, calib, cfg)

    counts: dict[str, int] = {}
    counts["01_fg_mask"] = _save_mask_png(
        run_dir / "01_fg_mask.png", fg_mask, rgb, CLASS_COLORS[CLS_COARSE]
    )
    counts["02_fg_dilated_coarse_exclusion"] = _save_mask_png(
        run_dir / "02_fg_dilated_coarse_exclusion.png",
        fg_dilated_coarse,
        rgb,
        (255, 128, 0),
    )
    counts["03_fg_dilated_refine_exclusion"] = _save_mask_png(
        run_dir / "03_fg_dilated_refine_exclusion.png",
        fg_dilated_refine,
        rgb,
        (255, 64, 0),
    )
    counts["04_coarse_talc_gate"] = _save_mask_png(
        run_dir / "04_coarse_talc_gate.png", coarse, rgb, CLASS_COLORS[CLS_TALC]
    )
    counts["05_coarse_gate_dilated_for_refine"] = _save_mask_png(
        run_dir / "05_coarse_gate_dilated_for_refine.png", rough, rgb, (0, 200, 255)
    )
    counts["05b_gmm_fit_interior"] = _save_mask_png(
        run_dir / "05b_gmm_fit_interior.png", gmm_fit_used, rgb, (255, 255, 0)
    )
    map_stem = "06_block01_activation" if cfg.talc_refine_mode == TALC_REFINE_MODE_DINO else "06_intensity_gradient"
    _save_gray_png(run_dir / f"{map_stem}.png", value_map)
    hist_meta: dict[str, Any] = {}
    if not gmm_meta.get("skipped") and "gmm_means" in gmm_meta:
        hist_meta = save_two_gmm_histogram(
            value_map,
            gmm_fit_used,
            means=np.asarray(gmm_meta["gmm_means"], dtype=np.float64),
            variances=np.asarray(gmm_meta["gmm_variances"], dtype=np.float64),
            weights=np.asarray(gmm_meta["gmm_weights"], dtype=np.float64),
            threshold=float(gmm_meta["gmm_threshold"]),
            dest_path=run_dir / "06b_talc_gmm_histogram.png",
            talc_side=str(gmm_meta["talc_side"]),
            unbiased_threshold=float(
                gmm_meta.get("gmm_threshold_unbiased", gmm_meta["gmm_threshold"])
            ),
            random_state=cfg.random_state,
            xlabel=hist_xlabel,
            title=hist_title,
            low_band_label=hist_low_label,
            high_band_label=hist_high_label,
        )
    counts["07_talc_band_2gmm"] = _save_mask_png(
        run_dir / "07_talc_band_2gmm.png", talc_band, rgb, CLASS_COLORS[CLS_TALC]
    )
    if talc_band_filled is not None:
        counts["07b_talc_band_filled_cc"] = _save_mask_png(
            run_dir / "07b_talc_band_filled_cc.png", talc_band_filled, rgb, (255, 100, 200)
        )
    counts["08_background_band_2gmm"] = _save_mask_png(
        run_dir / "08_background_band_2gmm.png", background_band, rgb, (160, 160, 160)
    )
    counts["09_refined_talc_before_fg_clip"] = _save_mask_png(
        run_dir / "09_refined_talc_before_fg_clip.png",
        refined_pre_clip,
        rgb,
        (180, 0, 255),
    )
    counts["10_refined_talc_final"] = _save_mask_png(
        run_dir / "10_refined_talc_final.png", refined, rgb, CLASS_COLORS[CLS_TALC]
    )
    counts["11_coarse_fg_object"] = _save_mask_png(
        run_dir / "11_coarse_fg_object.png", fg_object, rgb, CLASS_COLORS[CLS_COARSE]
    )
    counts["12_fine_partitions"] = _save_mask_png(
        run_dir / "12_fine_partitions.png", partitions, rgb, CLASS_COLORS[CLS_FINE]
    )
    _save_labels_png(run_dir / "13_final_segmentation.png", labels)
    counts["13_final_talc_in_labels"] = int((labels == CLS_TALC).sum())

    summary = {
        "image": str(image_path.resolve()),
        "output_dir": str(run_dir.resolve()),
        "config": {
            "fg_dilate_radius": cfg.fg_dilate_radius,
            "talc_refine_fg_dilate_radius": cfg.talc_refine_fg_dilate_radius,
            "talc_gmm_fg_buffer_radius": cfg.talc_gmm_fg_buffer_radius,
            "talc_gmm_gate_erode": cfg.talc_gmm_gate_erode,
            "talc_contour_dilate": cfg.talc_contour_dilate,
            "talc_gmm_threshold_high_bias": cfg.talc_gmm_threshold_high_bias,
            "talc_refine_mode": cfg.talc_refine_mode,
            "talc_intensity_max": intensity_max,
            "close_radius": cfg.close_radius,
        },
        "pixel_counts": counts,
        "coarse_meta": coarse_meta,
        "gmm_fit_meta": gmm_fit_meta,
        "gmm_histogram": hist_meta,
        "gmm_meta": gmm_meta,
        "refine_meta": refine_meta,
        "hybrid_meta": hybrid_meta,
        "fg_meta": fg_meta,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save hybrid talc pipeline masks for step-by-step debugging"
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to normalized RGB image (e.g. data/calib/img1/normalized.png)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "outputs/hybrid_talc_debug",
        help="Root output directory (default: outputs/hybrid_talc_debug)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Compiled calibration directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/ml_service.yaml",
        help="ML service config YAML",
    )
    parser.add_argument(
        "--block01-activation",
        type=Path,
        default=None,
        help="Optional precomputed block01_activation.npy (runs DINO if missing)",
    )
    parser.add_argument(
        "--talc-refine-mode",
        choices=(TALC_REFINE_MODE_DINO, TALC_REFINE_MODE_GRADIENT),
        default=None,
        help="Talc refine backend (default: segmentation.talc_refine_mode from config)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for DINO when activation must be computed",
    )
    args = parser.parse_args()

    if not args.image.is_absolute():
        args.image = _ROOT / args.image
    if not args.out_dir.is_absolute():
        args.out_dir = _ROOT / args.out_dir
    if not args.calibration_dir.is_absolute():
        args.calibration_dir = _ROOT / args.calibration_dir
    if args.config and not args.config.is_absolute():
        args.config = _ROOT / args.config
    if args.block01_activation and not args.block01_activation.is_absolute():
        args.block01_activation = _ROOT / args.block01_activation
    if args.block01_activation is None:
        args.block01_activation = _default_block01_activation_path(args.image)

    summary = run_debug(
        args.image,
        args.out_dir,
        calibration_dir=args.calibration_dir,
        config_path=args.config,
        activation_path=args.block01_activation,
        device=torch.device(args.device),
        talc_refine_mode=args.talc_refine_mode,
    )
    print(f"Saved debug masks to: {summary['output_dir']}")
    print("Files:")
    for name in sorted(Path(summary["output_dir"]).glob("*")):
        print(f"  {name.name}")
    print(f"Summary: {Path(summary['output_dir']) / 'summary.json'}")


if __name__ == "__main__":
    main()
