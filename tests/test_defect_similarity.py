"""Tests for block-11 similarity defect detection."""

from __future__ import annotations

import numpy as np
import torch

from ml.lib.constants import PATCH_SIZE
from ml.lib.segmentation.defect_similarity import defect_mask_from_block11_similarity


def test_defect_band_picks_highest_activation_similarity_region():
    h, w = 56, 56
    hp, wp = h // PATCH_SIZE, w // PATCH_SIZE
    intensity = np.full((h, w), 0.5, dtype=np.float32)
    intensity[10:14, 10:14] = 0.05  # darkest non-FG patch (pr=0, pc=0)

    fg_mask = np.zeros((h, w), dtype=np.uint8)
    fg_mask[28:42, 28:42] = 1

    c = 4
    feats = torch.zeros(c, hp, wp)
    # Reference patch (0,0) gets moderate activation; band-2 patches get higher.
    feats[:, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for pr in range(hp):
        for pc in range(wp):
            if (pr, pc) == (0, 0):
                continue
            if pr + pc >= 3:
                feats[:, pr, pc] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # high act, low sim
            elif pr + pc >= 1:
                feats[:, pr, pc] = torch.tensor([0.9, 0.1, 0.0, 0.0])  # high sim
            else:
                feats[:, pr, pc] = torch.tensor([0.5, 0.5, 0.0, 0.0])  # mid sim

    mask, meta = defect_mask_from_block11_similarity(
        intensity,
        feats,
        fg_mask,
        max_samples=10_000,
        random_state=0,
    )

    assert meta["reference_patch_row_col"] == [0, 0]
    assert meta["defect_similarity_band"] is not None
    assert mask.dtype == np.uint8
    assert mask.shape == (h, w)
    assert not np.any(mask & fg_mask.astype(bool))
