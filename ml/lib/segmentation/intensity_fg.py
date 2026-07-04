"""Intensity + region GMM foreground segmentation (coarse / fine)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.lib.calibration.filters import rgb_to_gray
from ml.lib.constants import REGION_NAMES
from ml.lib.segmentation.intensity import (
    build_intensity,
    fit_region_gmm,
    thresholds_from_adjacent_intersections,
    two_gmm_threshold,
)
from ml.lib.segmentation.regions import (
    morphological_close_label_map,
    promote_regions_by_overlap,
    segment_by_thresholds,
    split_foreground_object_and_partitions,
)


def segment_coarse_fine_intensity(
    rgb: np.ndarray,
    region_activation: np.ndarray,
    *,
    preprocess: bool,
    denoise: bool,
    illum_sigma: float,
    max_samples: int,
    random_state: int,
    region_overlap: float,
    close_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Old intensity pipeline: 2-GMM FG seed + region promotion → coarse / fine."""
    gray = rgb_to_gray(rgb)
    intensity = build_intensity(
        gray,
        preprocess=preprocess,
        illum_sigma=illum_sigma,
        denoise=denoise,
    )

    means, variances, weights = fit_region_gmm(
        region_activation, max_samples, random_state
    )
    fg_t, fg_means, fg_vars, fg_weights = two_gmm_threshold(
        intensity,
        max_samples=max_samples,
        random_state=random_state,
    )

    thresholds = thresholds_from_adjacent_intersections(means, variances, weights)
    region_labels_raw = segment_by_thresholds(region_activation, thresholds)
    region_labels = morphological_close_label_map(
        region_labels_raw,
        radius=close_radius,
        class_ids=list(REGION_NAMES.keys()),
    )

    fg_seed = (intensity >= fg_t).astype(np.uint8)
    fg_mask = promote_regions_by_overlap(
        region_labels,
        fg_seed,
        region_overlap,
    )
    fg_object, partitions = split_foreground_object_and_partitions(fg_mask)

    meta = {
        "method": "intensity_fg_gmm",
        "fg_threshold": float(fg_t),
        "fg_gmm_means": fg_means.tolist(),
        "region_gmm_means": means.tolist(),
        "region_thresholds": thresholds.tolist(),
    }
    return fg_object, partitions, fg_mask, meta
