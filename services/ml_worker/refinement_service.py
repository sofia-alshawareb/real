"""Online calibration refinement: append user hint samples and re-segment."""

from __future__ import annotations

import numpy as np
import torch

from ml.lib.calibration.colors import class_key_to_id
from ml.lib.calibration.extract import extract_samples_from_image
from ml.lib.calibration.filters import apply_class_filter
from ml.lib.calibration.store import CalibrationStore
from ml.lib.calibration.types import CalibrationData
from ml.lib.constants import CALIB_CLASS_KEY_BY_ID, CLS_TALC
from ml.lib.types import SegmentationResult
from services.ml_worker.config import SegmentationConfig
from services.ml_worker.segmentation_service import SegmentationService


class CalibrationRefinementService:
    def __init__(
        self,
        config: SegmentationConfig,
        calib_store: CalibrationStore,
        segmentation: SegmentationService,
    ) -> None:
        self.config = config
        self.calib_store = calib_store
        self.segmentation = segmentation

    def refine(
        self,
        rgb: np.ndarray,
        block11_features: np.ndarray | torch.Tensor,
        hint_mask: np.ndarray,
        ui_class: str,
        *,
        block01_activation: np.ndarray | None = None,
    ) -> SegmentationResult:
        if hint_mask.shape[:2] != rgb.shape[:2]:
            raise ValueError(
                f"hint_mask shape {hint_mask.shape[:2]} != image {rgb.shape[:2]}"
            )
        hint_bool = hint_mask.astype(bool)
        if not np.any(hint_bool):
            raise ValueError("hint mask is empty")

        class_id = class_key_to_id(ui_class)
        class_key = CALIB_CLASS_KEY_BY_ID[class_id]
        from ml.lib.calibration.filters import rgb_to_gray

        gray = rgb_to_gray(rgb)
        filtered, filt_meta = apply_class_filter(
            class_id,
            gray,
            hint_bool,
            rgb=rgb,
            talc_black_max=self.config.talc_black_max,
        )
        if not np.any(filtered):
            raise ValueError("no valid samples after class filter")

        class_masks = {class_id: filtered}
        rgb_by_class, emb_by_class, extract_report = extract_samples_from_image(
            rgb,
            class_masks,
            block11_features,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )

        append_meta = self.calib_store.append_samples(
            class_key,
            rgb_by_class.get(class_key, np.zeros((0, 3), np.float32)),
            emb_by_class.get(class_key, np.zeros((0, 384), np.float32)),
            max_rgb_samples=self.config.max_rgb_samples,
            max_emb_samples=self.config.max_embedding_samples,
            random_state=self.config.random_state,
        )
        calib = self.calib_store.get()

        feats = (
            block11_features
            if isinstance(block11_features, torch.Tensor)
            else torch.from_numpy(block11_features.astype(np.float32))
        )
        result = self.segmentation.run(
            rgb,
            calib,
            block01_activation=block01_activation if self.config.mode == "hybrid" else None,
            block11_features=feats if self.config.mode == "embedding" else None,
        )
        result.metadata["refinement"] = {
            "ui_class": ui_class,
            "filter": filt_meta,
            "extract": extract_report.get("classes", {}).get(class_key, {}),
            "append": append_meta,
        }
        if class_id == CLS_TALC:
            result.metadata["refinement"]["talc_filter"] = filt_meta
        return result
