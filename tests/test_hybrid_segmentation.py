"""Tests for hybrid segmentation (intensity coarse/fine + talc refine modes)."""

from __future__ import annotations

import numpy as np

from ml.lib.calibration.types import CalibrationData
from ml.lib.constants import CLS_COARSE, CLS_FINE, CLS_TALC, SEGMENTATION_MODE_HYBRID, TALC_REFINE_MODE_DINO, TALC_REFINE_MODE_GRADIENT
from ml.lib.segmentation.calibrated import segment_hybrid
from ml.lib.types import SegmentConfig


def _hybrid_fixture():
    h, w = 112, 112
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    rgb[30:80, 30:80] = (40, 40, 40)
    rgb[50:60, 50:60] = (180, 180, 180)
    rgb[8:36, 8:36] = (8, 8, 8)

    activation = np.full((h, w), 0.95, dtype=np.float32)
    activation[30:80, 30:80] = 0.75
    activation[50:60, 50:60] = 0.9
    activation[8:36, 8:36] = 0.55

    calib = CalibrationData(stats={})
    return rgb, activation, calib


def test_hybrid_dino_refine():
    rgb, activation, calib = _hybrid_fixture()
    cfg = SegmentConfig(
        segmentation_mode=SEGMENTATION_MODE_HYBRID,
        close_radius=0,
        fg_dilate_radius=3,
        talc_black_max=45.0,
        talc_contour_dilate=4,
        talc_refine_mode=TALC_REFINE_MODE_DINO,
    )
    labels, meta = segment_hybrid(rgb, activation, calib, cfg)
    assert meta["method"] == "hybrid"
    assert meta["talc"]["refine_mode"] == TALC_REFINE_MODE_DINO
    assert meta["talc"]["block01_refine"]["method"] == "talc_block01_activation_refine"
    assert set(np.unique(labels)).issubset({0, CLS_COARSE, CLS_FINE, CLS_TALC})
    assert (labels == CLS_TALC).any()


def test_hybrid_gradient_refine():
    rgb, activation, calib = _hybrid_fixture()
    cfg = SegmentConfig(
        segmentation_mode=SEGMENTATION_MODE_HYBRID,
        close_radius=0,
        fg_dilate_radius=3,
        talc_black_max=45.0,
        talc_contour_dilate=4,
        talc_refine_mode=TALC_REFINE_MODE_GRADIENT,
    )
    labels, meta = segment_hybrid(rgb, activation, calib, cfg)
    assert meta["talc"]["refine_mode"] == TALC_REFINE_MODE_GRADIENT
    assert meta["talc"]["gradient_refine"]["method"] == "talc_gradient_refine"
    assert (labels == CLS_TALC).any()
