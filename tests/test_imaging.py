"""Tests for image normalization."""

from __future__ import annotations

import numpy as np

from ml.lib.imaging import load_rgb_from_bytes
from tests.conftest import make_16bit_tiff_bytes, make_multipage_tiff_bytes


def test_16bit_tiff_normalization():
    rgb = load_rgb_from_bytes(make_16bit_tiff_bytes())
    assert rgb.dtype == np.uint8
    assert rgb.ndim == 3 and rgb.shape[2] == 3


def test_multipage_tiff_first_page():
    rgb = load_rgb_from_bytes(make_multipage_tiff_bytes())
    assert rgb.shape == (32, 32, 3)
    assert tuple(rgb[0, 0]) == (255, 0, 0)
