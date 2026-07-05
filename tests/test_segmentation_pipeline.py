"""Unit tests for calibrated segmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ml.lib.calibration.histograms import build_rgb_histogram
from ml.lib.calibration.types import CalibrationData, ClassCalibrationStats
from ml.lib.constants import CLASS_COLORS, CLS_COARSE, SEGMENTATION_MODE_INTENSITY
from ml.lib.segmentation.pipeline import segment_image
from ml.lib.types import SegmentConfig


@pytest.fixture
def tiny_calib():
    coarse_rgb = np.array([[46, 125, 50], [48, 127, 52]], dtype=np.float32)
    return CalibrationData(
        rgb_samples={
            "coarse": coarse_rgb,
            "fine": np.zeros((0, 3), np.float32),
            "talc": np.zeros((0, 3), np.float32),
            "matrix": np.zeros((0, 3), np.float32),
        },
        embedding_samples={k: np.zeros((0, 384), np.float32) for k in ("coarse", "fine", "talc", "matrix")},
        rgb_histograms={
            "coarse": build_rgb_histogram(coarse_rgb),
            "fine": np.zeros((32, 32, 32)),
            "talc": np.zeros((32, 32, 32)),
            "matrix": np.zeros((32, 32, 32)),
        },
        stats={"coarse": ClassCalibrationStats(count=2), "fine": ClassCalibrationStats(count=0), "talc": ClassCalibrationStats(count=0), "matrix": ClassCalibrationStats(count=0)},
    )


def test_segment_image_shape_and_classes(tiny_calib):
    rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
    rgb[8:16, 8:16] = CLASS_COLORS[CLS_COARSE]
    result = segment_image(
        rgb,
        tiny_calib,
        SegmentConfig(segmentation_mode=SEGMENTATION_MODE_INTENSITY, min_backproj_score=1e-8),
    )
    assert result.labels.shape == rgb.shape[:2]
    assert result.labels.dtype == np.uint8
    assert set(np.unique(result.labels)).issubset({0, 1, 2, 3, 4})
    assert result.mask_to_native_scale == 1.0
    assert "final_class_counts" in result.metadata
