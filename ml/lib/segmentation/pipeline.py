"""End-to-end GMM segmentation pipeline (no fork — CUDA-safe)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ml.lib.constants import (
    CLASS_NAMES,
    REGION_MAP_DINO,
    REGION_MAP_INTENSITY_GRADIENT,
    REGION_NAMES,
)
from ml.lib.segmentation.intensity import (
    build_intensity,
    fit_region_gmm,
    intensity_gradient_map,
    thresholds_from_adjacent_intersections,
    two_gmm_threshold,
)
from ml.lib.segmentation.regions import (
    build_final_segmentation,
    class_counts,
    defect_mask_from_regions,
    morphological_close_label_map,
    region_counts,
    segment_by_thresholds,
    split_foreground_object_and_partitions,
)
from ml.lib.types import SegmentConfig, SegmentationResult


def rgb_to_gray01(rgb: np.ndarray) -> np.ndarray:
    return np.mean(rgb.astype(np.float32) / 255.0, axis=2)


def segment_image(
    rgb: np.ndarray,
    region_map_values: np.ndarray,
    config: SegmentConfig | None = None,
) -> SegmentationResult:
    """Run GMM segmentation given a precomputed region map (DINO activation or gradient)."""
    cfg = config or SegmentConfig()
    t0 = time.perf_counter()

    gray = rgb_to_gray01(rgb)
    intensity = build_intensity(
        gray,
        preprocess=cfg.preprocess,
        illum_sigma=cfg.illum_sigma,
        denoise=cfg.denoise,
    )

    # No fork pool — ThreadPoolExecutor is CUDA-safe (R1).
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_region = pool.submit(
            fit_region_gmm,
            region_map_values,
            cfg.max_samples,
            cfg.random_state,
        )
        fut_fg = pool.submit(
            two_gmm_threshold,
            intensity,
            max_samples=cfg.max_samples,
            random_state=cfg.random_state,
        )
        means, variances, weights = fut_region.result()
        fg_t, fg_means, fg_vars, fg_weights = fut_fg.result()

    thresholds = thresholds_from_adjacent_intersections(means, variances, weights)
    region_labels_raw = segment_by_thresholds(region_map_values, thresholds)
    region_labels = morphological_close_label_map(
        region_labels_raw,
        radius=cfg.close_radius,
        class_ids=list(REGION_NAMES.keys()),
    )

    fg_seed = (intensity >= fg_t).astype(np.uint8)
    from ml.lib.segmentation.regions import promote_regions_by_overlap

    fg_mask = promote_regions_by_overlap(
        region_labels, fg_seed, cfg.region_overlap
    )
    fg_object, partitions = split_foreground_object_and_partitions(fg_mask)

    non_fg = fg_mask == 0
    defect_t, def_means, def_vars, def_weights = two_gmm_threshold(
        intensity,
        max_samples=cfg.max_samples,
        random_state=cfg.random_state,
        pixel_mask=non_fg,
    )
    defect_seed = (intensity < defect_t).astype(np.uint8)
    defect_mask, defect_match_mode = defect_mask_from_regions(
        region_labels,
        defect_seed,
        region_map=cfg.region_map,
        region_overlap=cfg.region_overlap,
    )

    seg_raw = build_final_segmentation(fg_object, partitions, defect_mask)
    labels = morphological_close_label_map(
        seg_raw,
        radius=cfg.close_radius,
        class_ids=list(CLASS_NAMES.keys()),
    )

    h, w = rgb.shape[:2]
    elapsed = time.perf_counter() - t0
    metadata = {
        "region_map": cfg.region_map,
        "region_gmm4": {
            "means": means.tolist(),
            "variances": variances.tolist(),
            "weights": weights.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "fg_intensity_gmm2": {
            "threshold": float(fg_t),
            "means": fg_means.tolist(),
            "variances": fg_vars.tolist(),
            "weights": fg_weights.tolist(),
        },
        "defect_intensity_gmm2": {
            "threshold": float(defect_t),
            "means": def_means.tolist(),
            "variances": def_vars.tolist(),
            "weights": def_weights.tolist(),
            "match_mode": defect_match_mode,
        },
        "final_class_counts": class_counts(labels),
        "dino_region_counts": region_counts(region_labels),
        "elapsed_s": elapsed,
    }

    return SegmentationResult(
        labels=labels.astype(np.uint8),
        native_width=w,
        native_height=h,
        mask_width=w,
        mask_height=h,
        mask_to_native_scale=1.0,
        metadata=metadata,
    )


def build_region_map_from_intensity(
    intensity: np.ndarray,
    region_map: str,
) -> np.ndarray:
    if region_map == REGION_MAP_INTENSITY_GRADIENT:
        return intensity_gradient_map(intensity)
    if region_map == REGION_MAP_DINO:
        raise ValueError("DINO region map must be supplied externally.")
    raise ValueError(f"Unknown region_map: {region_map}")
