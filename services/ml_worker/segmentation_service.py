"""Segmentation service consuming in-memory DINO block-1 activation."""

from __future__ import annotations

import numpy as np

from ml.lib.segmentation.pipeline import segment_image
from ml.lib.types import SegmentConfig, SegmentationResult
from services.ml_worker.config import SegmentationConfig


class SegmentationService:
    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        if config.region_map != "dino":
            raise ValueError("Service only supports region_map=dino in phase 1.")
        if config.block_index != 1:
            raise ValueError("Service requires block_index=1 in phase 1.")

    def run(
        self,
        rgb: np.ndarray,
        block01_activation: np.ndarray,
    ) -> SegmentationResult:
        seg_cfg = SegmentConfig(
            region_map=self.config.region_map,
            num_blocks=12,
            block_index=self.config.block_index,
            preprocess=self.config.preprocess,
            denoise=self.config.denoise,
            illum_sigma=self.config.illum_sigma,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            region_overlap=self.config.region_overlap,
            close_radius=self.config.close_radius,
        )
        if block01_activation.shape[:2] != rgb.shape[:2]:
            raise ValueError(
                f"Activation shape {block01_activation.shape[:2]} != RGB {rgb.shape[:2]}"
            )
        return segment_image(rgb, block01_activation.astype(np.float32), seg_cfg)
