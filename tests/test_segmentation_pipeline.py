"""Unit tests for segmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ml.lib.segmentation.pipeline import segment_image
from ml.lib.types import SegmentConfig


def test_segment_image_shape_and_classes(synthetic_rgb, synthetic_activation):
    result = segment_image(
        synthetic_rgb,
        synthetic_activation,
        SegmentConfig(random_state=0),
    )
    assert result.labels.shape == synthetic_rgb.shape[:2]
    assert result.labels.dtype == np.uint8
    assert set(np.unique(result.labels)).issubset({0, 1, 2, 3})
    assert result.mask_to_native_scale == 1.0
    assert "final_class_counts" in result.metadata


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA required for twice-in-process test",
)
def test_segment_twice_in_cuda_process(synthetic_rgb, synthetic_activation):
    import torch

    torch.cuda.init()
    cfg = SegmentConfig(random_state=0)
    r1 = segment_image(synthetic_rgb, synthetic_activation, cfg)
    r2 = segment_image(synthetic_rgb, synthetic_activation, cfg)
    assert np.array_equal(r1.labels, r2.labels)
