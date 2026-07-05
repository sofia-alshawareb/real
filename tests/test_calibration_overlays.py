"""Tests for filtered talc overlay rendering."""

from __future__ import annotations

import numpy as np

from ml.lib.calibration.overlays import render_filtered_talc_overlay


def test_render_filtered_talc_overlay_blends_kept_and_rejected():
    rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
    raw = np.zeros((32, 32), dtype=bool)
    raw[10:20, 10:20] = True
    kept = raw.copy()
    kept[10:15, 10:20] = False

    out = render_filtered_talc_overlay(rgb, raw, kept)

    assert not np.array_equal(out[18, 18], rgb[18, 18])
    assert not np.array_equal(out[12, 12], rgb[12, 12])
    assert out[18, 18].sum() != rgb[18, 18].sum()
    assert out[12, 12].sum() != rgb[12, 12].sum()
