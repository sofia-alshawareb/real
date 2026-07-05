"""Calibration sample filters."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.lib.constants import CLS_TALC, DEFAULT_TALC_BLACK_MAX


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return np.mean(rgb.astype(np.float32), axis=2)


def is_visually_black(
    gray: np.ndarray,
    rgb: np.ndarray,
    *,
    black_max: float,
) -> np.ndarray:
    """Pixels that look black to the eye: low mean intensity and no bright RGB channel."""
    rgb_f = rgb.astype(np.float32)
    channel_max = np.max(rgb_f, axis=2)
    return (gray < black_max) & (channel_max < black_max)


def filter_talc_mask(
    gray: np.ndarray,
    talc_color_mask: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    talc_black_max: float = DEFAULT_TALC_BLACK_MAX,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep only perceptually black pixels inside the hand-drawn talc mask."""
    talc_bool = talc_color_mask.astype(bool)
    n_mask = int(talc_bool.sum())
    meta: dict[str, Any] = {
        "talc_mask_pixels": n_mask,
        "talc_kept_pixels": 0,
        "talc_rejected_pixels": n_mask,
        "rejection_rate": 1.0 if n_mask else 0.0,
        "method": "none",
        "talc_black_max": float(talc_black_max),
    }
    if n_mask == 0:
        return np.zeros_like(talc_bool, dtype=bool), meta

    black_max = float(talc_black_max)
    if rgb is not None:
        kept = talc_bool & is_visually_black(gray, rgb, black_max=black_max)
    else:
        kept = talc_bool & (gray < black_max)

    n_kept = int(kept.sum())
    meta["method"] = "black_threshold"
    meta["talc_kept_pixels"] = n_kept
    meta["talc_rejected_pixels"] = n_mask - n_kept
    meta["rejection_rate"] = float((n_mask - n_kept) / n_mask) if n_mask else 0.0
    return kept, meta


def apply_class_filter(
    class_id: int,
    gray: np.ndarray,
    class_mask: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    talc_black_max: float = DEFAULT_TALC_BLACK_MAX,
) -> tuple[np.ndarray, dict[str, Any]]:
    if class_id == CLS_TALC:
        return filter_talc_mask(
            gray,
            class_mask,
            rgb=rgb,
            talc_black_max=talc_black_max,
        )
    return class_mask.astype(bool), {"method": "none", "filtered": False}
