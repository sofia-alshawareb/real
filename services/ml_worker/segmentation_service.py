"""Calibrated segmentation service."""

from __future__ import annotations

import numpy as np
import torch

from ml.lib.calibration.types import CalibrationData
from ml.lib.constants import SEGMENTATION_MODE_EMBEDDING, SEGMENTATION_MODE_HYBRID
from ml.lib.segmentation.pipeline import segment_image
from ml.lib.types import SegmentConfig, SegmentationResult
from services.ml_worker.config import SegmentationConfig


class SegmentationService:
    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config

    def _seg_cfg(self) -> SegmentConfig:
        return SegmentConfig(
            segmentation_mode=self.config.mode,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            close_radius=self.config.close_radius,
            min_backproj_score=self.config.min_backproj_score,
            min_cosine_sim=self.config.min_cosine_sim,
            rgb_hist_bins=self.config.rgb_hist_bins,
            preprocess=self.config.preprocess,
            denoise=self.config.denoise,
            illum_sigma=self.config.illum_sigma,
            region_overlap=self.config.region_overlap,
            block_index=self.config.block_index,
            fg_dilate_radius=self.config.fg_dilate_radius,
            talc_black_max=self.config.talc_black_max,
        )

    def run(
        self,
        rgb: np.ndarray,
        calib: CalibrationData,
        block01_activation: np.ndarray | None = None,
        block01_features: torch.Tensor | np.ndarray | None = None,
        block11_features: torch.Tensor | np.ndarray | None = None,
    ) -> SegmentationResult:
        mode = self.config.mode
        block_features = block01_features if block01_features is not None else block11_features
        if mode == SEGMENTATION_MODE_EMBEDDING and block_features is None:
            raise ValueError("block01_features required for embedding segmentation mode")
        if mode == SEGMENTATION_MODE_HYBRID and (
            block01_activation is None or block_features is None
        ):
            raise ValueError(
                "block01_activation and block01_features required for hybrid segmentation mode"
            )
        return segment_image(
            rgb,
            calib,
            self._seg_cfg(),
            block01_activation=block01_activation,
            block01_features=block_features,
        )
