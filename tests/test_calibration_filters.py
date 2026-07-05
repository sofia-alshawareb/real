"""Tests for talc mask filtering."""

from __future__ import annotations

import numpy as np

from ml.lib.calibration.filters import filter_talc_mask


def test_filter_talc_keeps_dark_pixels_only():
    h, w = 40, 40
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    rgb[10:30, 10:30] = (20, 20, 20)
    gray = np.mean(rgb.astype(np.float32), axis=2)
    talc_mask = np.zeros((h, w), dtype=bool)
    talc_mask[8:32, 8:32] = True

    kept, meta = filter_talc_mask(gray, talc_mask, rgb=rgb, talc_black_max=45.0)

    assert meta["method"] == "black_threshold"
    assert meta["talc_mask_pixels"] == talc_mask.sum()
    assert kept.sum() > 0
    assert kept.sum() < talc_mask.sum()
    assert np.all(gray[kept] < 45)
