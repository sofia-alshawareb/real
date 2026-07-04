"""DINO patch grid helpers shared by calibration extract and segmentation."""

from __future__ import annotations

import numpy as np
import torch

from ml.lib.constants import PATCH_SIZE


def pool_mask_to_patch_grid(mask: np.ndarray, hp: int, wp: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((hp, wp), dtype=bool)
    for pr in range(hp):
        y0 = pr * PATCH_SIZE
        y1 = min((pr + 1) * PATCH_SIZE, h)
        for pc in range(wp):
            x0 = pc * PATCH_SIZE
            x1 = min((pc + 1) * PATCH_SIZE, w)
            out[pr, pc] = bool(mask[y0:y1, x0:x1].any())
    return out


def unit_patch_vectors(
    block11_features: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, int, int, int]:
    feats = (
        block11_features
        if isinstance(block11_features, torch.Tensor)
        else torch.from_numpy(block11_features.astype(np.float32))
    )
    feats = feats.detach().float().cpu()
    c, hp, wp = feats.shape
    vectors = feats.reshape(c, -1).T.numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-8)
    return unit, c, hp, wp
