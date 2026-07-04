from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ml.lib.constants import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_REGION_OVERLAP,
    REGION_MAP_DINO,
)


@dataclass
class SegmentConfig:
    region_map: str = REGION_MAP_DINO
    num_blocks: int = 12
    block_index: int = 1
    preprocess: bool = False
    denoise: bool = True
    illum_sigma: float = 64.0
    max_samples: int = 300_000
    random_state: int = 0
    region_overlap: float = DEFAULT_REGION_OVERLAP
    close_radius: int = 3


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
        return cls(
            block01_activation=result.activation(1),
            block11_activation=result.activation(11),
            block01_features=result.features(1).numpy(),
            block11_features=result.features(11).numpy(),
            meta=dict(result.meta),
        )
