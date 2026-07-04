"""Map colored calibration / hint mask RGB to UI class indices."""

from __future__ import annotations

import numpy as np

from ml.lib.constants import (
    CALIB_CLASS_ID_BY_KEY,
    CLS_BACKGROUND,
    MASK_COLOR_TOLERANCE,
    UI_CLASS_COLORS,
)


def _color_matches(
    rgb: np.ndarray,
    target: tuple[int, int, int],
    tolerance: int,
) -> np.ndarray:
    tr, tg, tb = target
    if tolerance <= 0:
        return (
            (rgb[..., 0] == tr) & (rgb[..., 1] == tg) & (rgb[..., 2] == tb)
        )
    return (
        (np.abs(rgb[..., 0].astype(np.int16) - tr) <= tolerance)
        & (np.abs(rgb[..., 1].astype(np.int16) - tg) <= tolerance)
        & (np.abs(rgb[..., 2].astype(np.int16) - tb) <= tolerance)
    )


def class_masks_from_colored_png(
    mask_rgb: np.ndarray,
    *,
    tolerance: int = MASK_COLOR_TOLERANCE,
) -> dict[int, np.ndarray]:
    """Return {class_id: boolean mask} for non-background classes."""
    out: dict[int, np.ndarray] = {}
    for cls_id, color in UI_CLASS_COLORS.items():
        if cls_id == CLS_BACKGROUND:
            continue
        matched = _color_matches(mask_rgb, color, tolerance)
        if np.any(matched):
            out[cls_id] = matched
    return out


def class_key_to_id(class_key: str) -> int:
    if class_key not in CALIB_CLASS_ID_BY_KEY:
        raise ValueError(f"Unknown calibration class key: {class_key!r}")
    return CALIB_CLASS_ID_BY_KEY[class_key]
