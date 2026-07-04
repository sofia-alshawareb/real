from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ml.lib.constants import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_MIN_BACKPROJ_SCORE,
    DEFAULT_MIN_COSINE_SIM,
    DEFAULT_REGION_OVERLAP,
    DEFAULT_FG_DILATE_RADIUS,
    DEFAULT_TALC_BLACK_MAX,
    SEGMENTATION_MODE_INTENSITY,
)


@dataclass
class SegmentConfig:
    segmentation_mode: str = SEGMENTATION_MODE_INTENSITY
    num_blocks: int = 12
    block_index: int = 11
    preprocess: bool = False
    denoise: bool = True
    illum_sigma: float = 64.0
    max_samples: int = 300_000
    random_state: int = 0
    region_overlap: float = DEFAULT_REGION_OVERLAP
    close_radius: int = 3
    min_backproj_score: float = DEFAULT_MIN_BACKPROJ_SCORE
    min_cosine_sim: float = DEFAULT_MIN_COSINE_SIM
    fg_dilate_radius: int = DEFAULT_FG_DILATE_RADIUS
    talc_black_max: float = DEFAULT_TALC_BLACK_MAX
    rgb_hist_bins: int = 32


@dataclass
class SegmentationResult:
    labels: np.ndarray
    native_width: int
    native_height: int
    mask_width: int
    mask_height: int
    mask_to_native_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def class_names(self) -> dict[int, str]:
        return dict(CLASS_NAMES)

    @property
    def class_colors(self) -> dict[int, tuple[int, int, int]]:
        return dict(CLASS_COLORS)


@dataclass
class DinoBlockOutput:
    block_index: int
    features: torch.Tensor
    activation: np.ndarray


@dataclass
class DinoInferenceResult:
    blocks: dict[int, DinoBlockOutput]
    native_width: int
    native_height: int
    inference_blocks: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def activation(self, block_index: int) -> np.ndarray:
        return self.blocks[block_index].activation

    def features(self, block_index: int) -> torch.Tensor:
        return self.blocks[block_index].features


@dataclass
class DinoArtifacts:
    block01_activation: np.ndarray
    block11_activation: np.ndarray
    block01_features: np.ndarray
    block11_features: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_inference(cls, result: DinoInferenceResult) -> DinoArtifacts:
        h, w = result.native_height, result.native_width

        def _activation(block_index: int) -> np.ndarray:
            if block_index in result.blocks:
                return result.activation(block_index)
            return np.zeros((h, w), dtype=np.float32)

        def _features(block_index: int) -> np.ndarray:
            if block_index in result.blocks:
                return result.features(block_index).numpy()
            ref = result.blocks.get(1) or next(iter(result.blocks.values()))
            return np.zeros_like(ref.features.numpy())

        return cls(
            block01_activation=_activation(1),
            block11_activation=_activation(11),
            block01_features=_features(1),
            block11_features=_features(11),
            meta=dict(result.meta),
        )
