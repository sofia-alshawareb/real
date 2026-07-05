"""Calibration prep visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ml.lib.calibration.filters import apply_class_filter, rgb_to_gray
from ml.lib.constants import CLASS_COLORS, CLS_TALC

FILTERED_TALC_OVERLAY_NAME = "filtered_talc_overlay.png"
# Amber highlight for hand-drawn talc pixels removed by the black threshold filter.
TALC_REJECTED_OVERLAY_COLOR = (255, 193, 7)


def render_filtered_talc_overlay(
    rgb: np.ndarray,
    raw_talc_mask: np.ndarray,
    filtered_talc_mask: np.ndarray,
    *,
    alpha_kept: float = 0.55,
    alpha_rejected: float = 0.45,
) -> np.ndarray:
    """Blend kept talc (blue) and rejected talc bleed (amber) over the normalized image."""
    out = rgb.astype(np.float32).copy()
    raw = raw_talc_mask.astype(bool)
    kept = filtered_talc_mask.astype(bool)
    rejected = raw & ~kept

    def _blend(mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
        if not np.any(mask):
            return
        color_arr = np.array(color, dtype=np.float32)
        for ch in range(3):
            out[..., ch][mask] = (1.0 - alpha) * out[..., ch][mask] + alpha * color_arr[ch]

    _blend(rejected, TALC_REJECTED_OVERLAY_COLOR, alpha_rejected)
    _blend(kept, CLASS_COLORS[CLS_TALC], alpha_kept)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_filtered_talc_overlay(
    rgb: np.ndarray,
    class_masks: dict[int, np.ndarray],
    dest_path: Path,
    *,
    compiled_path: Path | None = None,
    max_samples: int = 300_000,
    random_state: int = 0,
) -> dict | None:
    """Run talc black-threshold filter and save overlay PNG. Returns filter meta or None if no talc mask."""
    raw = class_masks.get(CLS_TALC)
    if raw is None or not np.any(raw):
        return None

    gray = rgb_to_gray(rgb)
    filtered, filt_meta = apply_class_filter(
        CLS_TALC,
        gray,
        raw,
        rgb=rgb,
    )
    overlay = render_filtered_talc_overlay(rgb, raw, filtered)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(dest_path)
    if compiled_path is not None:
        compiled_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(compiled_path)
    return {
        **filt_meta,
        "overlay_path": str(dest_path),
        "compiled_overlay_path": str(compiled_path) if compiled_path else None,
    }
