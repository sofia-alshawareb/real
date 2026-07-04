"""3D RGB histogram helpers for calibration backprojection."""

from __future__ import annotations

import numpy as np


def build_rgb_histogram(samples: np.ndarray, bins: int = 32) -> np.ndarray:
    """Build normalized 3D histogram from RGB samples in 0–255."""
    if samples.size == 0:
        return np.zeros((bins, bins, bins), dtype=np.float64)
    samples = samples.astype(np.float64)
    hist, _ = np.histogramdd(
        samples,
        bins=bins,
        range=((0.0, 256.0), (0.0, 256.0), (0.0, 256.0)),
    )
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def backproject_rgb_histogram(rgb: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """Per-pixel histogram lookup scores for uint8 RGB image (H,W,3)."""
    bins = hist.shape[0]
    if bins == 0 or hist.size == 0:
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    scale = 256.0 / bins
    r = np.clip((rgb[..., 0].astype(np.float32) / scale).astype(np.int32), 0, bins - 1)
    g = np.clip((rgb[..., 1].astype(np.float32) / scale).astype(np.int32), 0, bins - 1)
    b = np.clip((rgb[..., 2].astype(np.float32) / scale).astype(np.int32), 0, bins - 1)
    return hist[r, g, b].astype(np.float32)
