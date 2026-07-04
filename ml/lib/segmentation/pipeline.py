"""End-to-end calibrated segmentation pipeline."""

from __future__ import annotations

import numpy as np
import torch

from ml.lib.calibration.types import CalibrationData
from ml.lib.segmentation.calibrated import segment_calibrated
from ml.lib.types import SegmentConfig, SegmentationResult


def segment_image(
    rgb: np.ndarray,
    calib: CalibrationData,
    config: SegmentConfig | None = None,
    *,
    block01_activation: np.ndarray | None = None,
    block01_features: torch.Tensor | np.ndarray | None = None,
    block11_features: torch.Tensor | np.ndarray | None = None,
) -> SegmentationResult:
    """Run segmentation (RGB backprojection, embedding cosine, or hybrid)."""
    cfg = config or SegmentConfig()
    return segment_calibrated(
        rgb,
        calib,
        cfg,
        block01_activation=block01_activation,
        block01_features=block01_features,
        block11_features=block11_features,
    )
