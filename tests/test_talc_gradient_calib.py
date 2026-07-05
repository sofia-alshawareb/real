"""Tests for talc embedding margin calibration."""

from __future__ import annotations

import numpy as np
import pytest

from ml.lib.calibration.talc_threshold import (
    calibrate_talc_from_labeled_images,
    compute_talc_intensity_max,
    compute_talc_min_cosine_margin,
    resolve_talc_min_cosine_margin,
    resolve_talc_intensity_max,
)
from ml.lib.calibration.types import CalibrationData
from ml.lib.constants import CLS_MATRIX, CLS_TALC, DEFAULT_TALC_MIN_COSINE_MARGIN


def test_compute_min_margin_between_talc_and_background():
    talc = np.full(200, 0.12, dtype=np.float32)
    bg = np.full(200, -0.08, dtype=np.float32)
    threshold, meta = compute_talc_min_cosine_margin(talc, bg, min_patches=20)
    assert meta["method"] == "calibrated_margin_midpoint"
    assert -0.08 < threshold < 0.12
    assert meta["min_cosine_margin"] == threshold
    assert meta["min_region_mean_margin"] == pytest.approx(0.02)


def test_compute_intensity_max_from_samples():
    talc = np.full(200, 40.0, dtype=np.float32)
    bg = np.full(200, 55.0, dtype=np.float32)
    intensity_max, meta = compute_talc_intensity_max(talc, bg, min_pixels=50)
    assert meta["method"] == "calibrated_p90_or_midpoint"
    assert intensity_max == (40.0 + 55.0) / 2.0


def test_calibrate_from_labeled_images_pools_margin_samples():
    side = 140
    rgb = np.zeros((side, side, 3), dtype=np.uint8)
    rgb[0 : side // 2, 0 : side // 2] = (10, 10, 10)
    rgb[side // 2 :, side // 2 :] = (50, 50, 50)
    hp = side // 14
    features = np.zeros((8, hp, hp), dtype=np.float32)
    features[:, :, :] = np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1, 1)
    features[:, : hp // 2, : hp // 2] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1, 1)
    talc_mask = np.zeros((side, side), dtype=bool)
    talc_mask[0 : side // 2, 0 : side // 2] = True
    bg_mask = np.zeros((side, side), dtype=bool)
    bg_mask[side // 2 :, side // 2 :] = True
    class_masks = {CLS_TALC: talc_mask, CLS_MATRIX: bg_mask}
    talc_mean = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    matrix_mean = np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    meta = calibrate_talc_from_labeled_images(
        [(rgb, features, class_masks, "test")],
        reference_mean_embedding=talc_mean,
        reference_matrix_mean_embedding=matrix_mean,
    )
    assert meta["source"] == "sample_regions"
    assert meta["method"] == "calibrated_margin_midpoint"
    assert meta["min_cosine_margin"] > meta["background_margin_median"]


def test_resolve_uses_calibrated_margin_when_present():
    calib = CalibrationData(meta={"talc_contour": {"min_cosine_margin": 0.04}})
    assert resolve_talc_min_cosine_margin(calib, 0.0) == 0.04
    assert resolve_talc_intensity_max(calib, 45.0) == 45.0


def test_resolve_falls_back_to_config():
    calib = CalibrationData()
    assert resolve_talc_min_cosine_margin(calib, DEFAULT_TALC_MIN_COSINE_MARGIN) == 0.0
    assert resolve_talc_intensity_max(calib, 45.0) == 45.0
