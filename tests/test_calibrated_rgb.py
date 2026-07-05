"""Tests for RGB backprojection segmentation."""

from __future__ import annotations

import numpy as np

from ml.lib.calibration.histograms import build_rgb_histogram
from ml.lib.calibration.types import CalibrationData, ClassCalibrationStats
from ml.lib.constants import CLASS_COLORS, CLS_BACKGROUND, CLS_COARSE, CLS_TALC
from ml.lib.segmentation.calibrated import segment_rgb_backprojection


def test_rgb_backprojection_assigns_calibrated_class(tmp_path):
    coarse_rgb = np.array([[46, 125, 50], [50, 130, 55]], dtype=np.float32)
    talc_rgb = np.array([[21, 101, 192], [25, 105, 195]], dtype=np.float32)
    calib = CalibrationData(
        rgb_samples={"coarse": coarse_rgb, "fine": np.zeros((0, 3)), "talc": talc_rgb, "matrix": np.zeros((0, 3))},
        embedding_samples={k: np.zeros((0, 384), np.float32) for k in ("coarse", "fine", "talc", "matrix")},
        rgb_histograms={
            "coarse": build_rgb_histogram(coarse_rgb),
            "fine": np.zeros((32, 32, 32)),
            "talc": build_rgb_histogram(talc_rgb),
            "matrix": np.zeros((32, 32, 32)),
        },
        stats={
            "coarse": ClassCalibrationStats(count=2),
            "fine": ClassCalibrationStats(count=0),
            "talc": ClassCalibrationStats(count=2),
            "matrix": ClassCalibrationStats(count=0),
        },
    )

    rgb = np.full((10, 10, 3), 180, dtype=np.uint8)
    rgb[2:5, 2:5] = CLASS_COLORS[CLS_COARSE]
    rgb[6:9, 6:9] = CLASS_COLORS[CLS_TALC]

    labels, meta = segment_rgb_backprojection(rgb, calib, min_score=1e-8)

    assert meta["method"] == "rgb_backprojection"
    assert labels[3, 3] == CLS_COARSE
    assert labels[7, 7] == CLS_TALC
    assert labels[0, 0] == CLS_BACKGROUND
