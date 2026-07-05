"""Tests for online calibration refinement."""

from __future__ import annotations

import numpy as np
import torch

from ml.lib.calibration.store import CalibrationStore
from ml.lib.constants import CLASS_COLORS, CLS_COARSE, PATCH_SIZE
from ml.lib.segmentation.pipeline import segment_image
from ml.lib.types import SegmentConfig
from services.ml_worker.config import SegmentationConfig
from services.ml_worker.refinement_service import CalibrationRefinementService
from services.ml_worker.segmentation_service import SegmentationService


def test_refinement_appends_samples_and_resegments(tmp_path):
    coarse_rgb = np.array([[46, 125, 50]], dtype=np.float32)
    store = CalibrationStore(tmp_path / "compiled")
    store.initialize_from_merged(
        {"coarse": coarse_rgb, "fine": np.zeros((0, 3)), "talc": np.zeros((0, 3)), "matrix": np.zeros((0, 3))},
        {k: np.zeros((0, 8), np.float32) for k in ("coarse", "fine", "talc", "matrix")},
        ["init"],
    )

    seg_cfg = SegmentationConfig(mode="intensity", calibration_dir=tmp_path / "compiled")
    segmentation = SegmentationService(seg_cfg)
    refinement = CalibrationRefinementService(seg_cfg, store, segmentation)

    h, w = 56, 56
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    rgb[10:20, 10:20] = CLASS_COLORS[CLS_COARSE]

    hp, wp = h // PATCH_SIZE, w // PATCH_SIZE
    feats = torch.randn(8, hp, wp)
    feats = feats / feats.norm(dim=0, keepdim=True).clamp(min=1e-6)

    hint = np.zeros((h, w), dtype=np.uint8)
    hint[10:20, 10:20] = 1

    result = refinement.refine(rgb, feats.numpy(), hint, "coarse")

    assert "refinement" in result.metadata
    assert store.summary_counts()["coarse"] > 1
    assert result.labels.shape == (h, w)

    cfg = SegmentConfig(segmentation_mode="intensity", min_backproj_score=1e-8)
    seg = segment_image(rgb, store.get(), cfg)
    assert seg.labels.shape == (h, w)
