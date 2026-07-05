"""Tests for interactive calibration refinement talc filtering."""

from __future__ import annotations

import numpy as np

from ml.lib.calibration.filters import filter_talc_mask


def test_filter_talc_mask_keeps_black_only():
    h, w = 56, 56
    rgb = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)
    rgb[10:20, 10:20] = (12, 12, 12)
    rgb[30:40, 30:40] = (120, 120, 120)
    gray = np.mean(rgb.astype(np.float32), axis=2)

    hint = np.zeros((h, w), dtype=bool)
    hint[10:20, 10:20] = True
    hint[30:40, 30:40] = True

    kept, meta = filter_talc_mask(gray, hint, rgb=rgb, talc_black_max=45.0)

    assert meta["method"] == "black_threshold"
    assert kept[15, 15]
    assert not kept[35, 35]
