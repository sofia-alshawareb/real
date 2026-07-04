"""Talc detection via perceptual black threshold outside dilated foreground."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.morphology import dilation, disk

from ml.lib.calibration.filters import rgb_to_gray
from ml.lib.constants import CLS_BACKGROUND, CLS_TALC
from ml.lib.segmentation.intensity import two_gmm_threshold
from ml.lib.segmentation.regions import morphological_close_label_map


def _is_visually_black(
    gray: np.ndarray,
    rgb: np.ndarray,
    *,
    black_max: float,
) -> np.ndarray:
    """Pixels that look black to the eye: low mean intensity and no bright RGB channel."""
    rgb_f = rgb.astype(np.float32)
    channel_max = np.max(rgb_f, axis=2)
    return (gray < black_max) & (channel_max < black_max)


def segment_talc_intensity_gmm(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    *,
    fg_dilate_radius: int,
    talc_black_max: float,
    preprocess: bool,
    denoise: bool,
    illum_sigma: float,
    max_samples: int,
    random_state: int,
    close_radius: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Outside dilated FG: black (human-eye) → talc, gray/green → background.

    Keeps 2-GMM on outside-FG grayscale values for diagnostics, but the split uses a
    fixed perceptual black cap so gray background cannot be absorbed into the dark cluster.
    """
    del preprocess, denoise, illum_sigma  # talc uses raw RGB gray, not denoised intensity
    h, w = rgb.shape[:2]
    gray = rgb_to_gray(rgb)

    fg = fg_mask.astype(bool)
    if fg_dilate_radius > 0:
        fg = dilation(fg, disk(int(fg_dilate_radius)))

    non_fg = ~fg
    if not np.any(non_fg):
        return np.zeros((h, w), dtype=np.uint8), {
            "method": "talc_intensity_gmm",
            "warning": "empty non-foreground",
        }

    values = gray[non_fg]
    if float(np.std(values)) < 1e-6:
        return np.zeros((h, w), dtype=np.uint8), {
            "method": "talc_intensity_gmm",
            "warning": "uniform intensity outside foreground",
            "fg_dilate_radius": int(fg_dilate_radius),
            "talc_black_max": float(talc_black_max),
            "non_fg_pixels": int(non_fg.sum()),
            "pixel_count": 0,
        }

    gmm_threshold, means, variances, weights = two_gmm_threshold(
        gray,
        max_samples=max_samples,
        random_state=random_state,
        pixel_mask=non_fg,
    )
    dark_id = int(np.argmin(means))
    black_max = float(talc_black_max)

    talc_mask = np.zeros((h, w), dtype=np.uint8)
    talc_mask[non_fg & _is_visually_black(gray, rgb, black_max=black_max)] = 1

    if close_radius > 0 and np.any(talc_mask):
        label_map = np.zeros((h, w), dtype=np.int32)
        label_map[talc_mask.astype(bool)] = CLS_TALC
        label_map = morphological_close_label_map(
            label_map,
            radius=close_radius,
            class_ids=[CLS_BACKGROUND, CLS_TALC],
        )
        talc_mask = (label_map == CLS_TALC).astype(np.uint8)

    meta = {
        "method": "talc_intensity_gmm",
        "fg_dilate_radius": int(fg_dilate_radius),
        "talc_black_max": black_max,
        "gmm_threshold": float(gmm_threshold),
        "intensity_threshold": black_max,
        "gmm_means": means.tolist(),
        "gmm_variances": variances.tolist(),
        "gmm_weights": weights.tolist(),
        "dark_cluster_id": dark_id,
        "non_fg_pixels": int(non_fg.sum()),
        "pixel_count": int(talc_mask.sum()),
    }
    return talc_mask, meta
