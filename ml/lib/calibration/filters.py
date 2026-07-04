"""Calibration sample filters."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from ml.lib.constants import CLS_TALC, N_GAUSSIANS_BINARY


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return np.mean(rgb.astype(np.float32), axis=2)


def filter_talc_mask(
    gray: np.ndarray,
    talc_color_mask: np.ndarray,
    *,
    min_pixels: int = 50,
    max_samples: int = 300_000,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return boolean mask of true talc pixels (dark component inside blue mask)."""
    talc_bool = talc_color_mask.astype(bool)
    n_mask = int(talc_bool.sum())
    meta: dict[str, Any] = {
        "talc_mask_pixels": n_mask,
        "talc_kept_pixels": 0,
        "talc_rejected_pixels": n_mask,
        "rejection_rate": 1.0 if n_mask else 0.0,
        "method": "none",
    }
    if n_mask == 0:
        return np.zeros_like(talc_bool, dtype=bool), meta

    kept = np.zeros_like(talc_bool, dtype=bool)
    if n_mask < min_pixels:
        vals = gray[talc_bool]
        if float(np.std(vals)) < 1e-6:
            kept = talc_bool.copy()
            meta["method"] = "uniform_fallback"
        else:
            med = float(np.median(vals))
            kept = talc_bool & (gray < med)
            meta["method"] = "median_fallback"
    else:
        try:
            values = gray[talc_bool].astype(np.float64)
            if values.size > max_samples:
                rng = np.random.default_rng(random_state)
                values = values[rng.choice(values.size, size=max_samples, replace=False)]
            if float(np.std(values)) < 1e-6:
                kept = talc_bool.copy()
                meta["method"] = "uniform_fallback"
            else:
                gmm = GaussianMixture(
                    n_components=N_GAUSSIANS_BINARY,
                    covariance_type="full",
                    random_state=random_state,
                    n_init=5,
                    max_iter=300,
                )
                gmm.fit(values.reshape(-1, 1))
                comp = gmm.predict(gray[talc_bool].astype(np.float64).reshape(-1, 1))
                means = gmm.means_.ravel()
                low_id = int(np.argmin(means))
                kept_vals = comp == low_id
                kept = np.zeros_like(talc_bool, dtype=bool)
                kept[talc_bool] = kept_vals
                meta["method"] = "2gmm_dark_component"
                meta["gmm_means"] = [float(means[0]), float(means[1])]
                meta["low_component_id"] = low_id
        except ValueError:
            vals = gray[talc_bool]
            if float(np.std(vals)) < 1e-6:
                kept = talc_bool.copy()
                meta["method"] = "uniform_fallback_gmm_failed"
            else:
                med = float(np.median(vals))
                kept = talc_bool & (gray < med)
                meta["method"] = "median_fallback_gmm_failed"

    n_kept = int(kept.sum())
    meta["talc_kept_pixels"] = n_kept
    meta["talc_rejected_pixels"] = n_mask - n_kept
    meta["rejection_rate"] = float((n_mask - n_kept) / n_mask) if n_mask else 0.0
    return kept, meta


def apply_class_filter(
    class_id: int,
    gray: np.ndarray,
    class_mask: np.ndarray,
    *,
    min_pixels: int = 50,
    max_samples: int = 300_000,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if class_id == CLS_TALC:
        return filter_talc_mask(
            gray,
            class_mask,
            min_pixels=min_pixels,
            max_samples=max_samples,
            random_state=random_state,
        )
    return class_mask.astype(bool), {"method": "none", "filtered": False}
