"""Tests for background embedding extraction during calibration prep."""

from __future__ import annotations

import numpy as np

from ml.lib.calibration.extract import extract_background_embeddings
from ml.lib.constants import CLS_COARSE, PATCH_SIZE


def test_extract_background_embeddings_ignores_painted_patches():
    h, w = 56, 56
    hp, wp = 4, 4
    c = 8
    class_masks = {
        CLS_COARSE: np.zeros((h, w), dtype=bool),
    }
    class_masks[CLS_COARSE][0:PATCH_SIZE, 0:PATCH_SIZE] = True

    block01 = np.zeros((c, hp, wp), dtype=np.float32)
    block01[:, 0, 0] = 1.0
    block01[:, 1, 1] = 2.0

    rows = extract_background_embeddings(class_masks, block01)
    assert rows.shape[0] == hp * wp - 1
    assert not np.allclose(rows, 1.0)
