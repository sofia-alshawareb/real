"""Tests for embedding cosine segmentation."""

from __future__ import annotations

import numpy as np
import torch

from ml.lib.calibration.types import CalibrationData, ClassCalibrationStats
from ml.lib.constants import CLS_COARSE, PATCH_SIZE
from ml.lib.segmentation.calibrated import segment_embedding_cosine


def test_embedding_cosine_assigns_nearest_class():
    hp, wp = 4, 4
    c = 8
    mean_coarse = np.zeros(c, dtype=np.float32)
    mean_coarse[0] = 1.0
    mean_coarse /= np.linalg.norm(mean_coarse)

    feats = torch.zeros(c, hp, wp)
    feats[:, 1, 1] = torch.from_numpy(mean_coarse)

    calib = CalibrationData(
        rgb_samples={k: np.zeros((0, 3), np.float32) for k in ("coarse", "fine", "talc", "matrix")},
        embedding_samples={k: np.zeros((0, c), np.float32) for k in ("coarse", "fine", "talc", "matrix")},
        stats={
            "coarse": ClassCalibrationStats(count=1, mean_embedding=mean_coarse),
            "fine": ClassCalibrationStats(count=0),
            "talc": ClassCalibrationStats(count=0),
            "matrix": ClassCalibrationStats(count=0),
        },
    )

    h, w = hp * PATCH_SIZE, wp * PATCH_SIZE
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    labels, meta = segment_embedding_cosine(rgb, feats, calib, min_cosine=0.5)

    assert meta["method"] == "embedding_cosine"
    assert labels[PATCH_SIZE : PATCH_SIZE + 5, PATCH_SIZE : PATCH_SIZE + 5].max() == CLS_COARSE
