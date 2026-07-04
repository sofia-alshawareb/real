from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClassCalibrationStats:
    count: int = 0
    mean_rgb: np.ndarray | None = None
    std_rgb: np.ndarray | None = None
    mean_embedding: np.ndarray | None = None


@dataclass
class CalibrationData:
    rgb_samples: dict[str, np.ndarray] = field(default_factory=dict)
    embedding_samples: dict[str, np.ndarray] = field(default_factory=dict)
    rgb_histograms: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, ClassCalibrationStats] = field(default_factory=dict)
    source_images: list[str] = field(default_factory=list)
    schema_version: str = "1.0"
    meta: dict[str, Any] = field(default_factory=dict)

    def active_class_keys(self) -> list[str]:
        return [k for k, arr in self.rgb_samples.items() if arr.shape[0] > 0]

    def active_class_ids(self) -> list[int]:
        from ml.lib.constants import CALIB_CLASS_ID_BY_KEY

        return [CALIB_CLASS_ID_BY_KEY[k] for k in self.active_class_keys()]

    def background_mean_embedding(self) -> np.ndarray | None:
        from ml.lib.constants import CALIB_BACKGROUND_KEY

        bg = self.stats.get(CALIB_BACKGROUND_KEY)
        if bg is None or bg.mean_embedding is None:
            return None
        return bg.mean_embedding
