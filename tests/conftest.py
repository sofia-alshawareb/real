"""Shared test fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _find_golden_image() -> Path | None:
    candidates = [
        ROOT
        / "data/raw/task3/Фото руд по сортам. ч1/Оталькованные руды/2550374-2 10х.JPG",
        ROOT / "task3-data/Фото руд по сортам. ч1/Оталькованные руды/2550374-2 10х.JPG",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


GOLDEN_IMAGE = _find_golden_image()


@pytest.fixture
def golden_image_path() -> Path:
    if GOLDEN_IMAGE is None:
        pytest.skip("Golden reference image not available")
    return GOLDEN_IMAGE


@pytest.fixture
def synthetic_rgb() -> np.ndarray:
    h, w = 128, 160
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[20:100, 30:130] = (180, 60, 40)
    rgb[60:90, 50:110] = (30, 30, 30)
    return rgb


@pytest.fixture
def synthetic_activation(synthetic_rgb: np.ndarray) -> np.ndarray:
    act = np.zeros(synthetic_rgb.shape[:2], dtype=np.float32)
    act[20:100, 30:130] = 0.8
    act[60:90, 50:110] = 0.2
    return act


def make_16bit_tiff_bytes() -> bytes:
    arr = (np.linspace(0, 65535, 64 * 64, dtype=np.uint16).reshape(64, 64))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="TIFF")
    return buf.getvalue()


def make_multipage_tiff_bytes() -> bytes:
    buf = io.BytesIO()
    frames = [
        Image.new("RGB", (32, 32), (255, 0, 0)),
        Image.new("RGB", (32, 32), (0, 255, 0)),
    ]
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()
