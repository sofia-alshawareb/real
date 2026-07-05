"""Extract RGB and embedding calibration samples from one image."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.lib.calibration.filters import apply_class_filter, rgb_to_gray
from ml.lib.calibration.patch_utils import pool_mask_to_patch_grid, unit_patch_vectors
from ml.lib.constants import CALIB_CLASS_KEY_BY_ID, CLS_TALC


def extract_background_embeddings(
    class_masks: dict[int, np.ndarray],
    block01_features: np.ndarray,
) -> np.ndarray:
    """Patch embeddings from calib regions not painted as any foreground class."""
    if not class_masks:
        return np.zeros((0, 384), dtype=np.float32)

    h, w = next(iter(class_masks.values())).shape
    painted = np.zeros((h, w), dtype=bool)
    for mask in class_masks.values():
        painted |= mask.astype(bool)

    unit, emb_dim, hp, wp = unit_patch_vectors(block01_features)
    painted_patch = pool_mask_to_patch_grid(painted, hp, wp)
    background_patch = ~painted_patch
    if not np.any(background_patch):
        return np.zeros((0, emb_dim), dtype=np.float32)
    return unit[background_patch.ravel()]


def extract_samples_from_image(
    rgb: np.ndarray,
    class_masks: dict[int, np.ndarray],
    block11_features: np.ndarray | None,
    *,
    max_samples: int = 300_000,
    random_state: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Return rgb_rows and embedding_rows keyed by class name."""
    gray = rgb_to_gray(rgb)
    rgb_out: dict[str, np.ndarray] = {}
    emb_out: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {"classes": {}}

    unit = None
    hp = wp = 0
    emb_dim = 384
    if block11_features is not None:
        unit, _, hp, wp = unit_patch_vectors(block11_features)
        emb_dim = unit.shape[1]

    for cls_id, mask in class_masks.items():
        class_key = CALIB_CLASS_KEY_BY_ID[cls_id]
        filtered, filt_meta = apply_class_filter(
            cls_id,
            gray,
            mask,
            rgb=rgb,
        )
        if not np.any(filtered):
            rgb_out[class_key] = np.zeros((0, 3), dtype=np.float32)
            emb_out[class_key] = np.zeros((0, emb_dim), dtype=np.float32)
            report["classes"][class_key] = {**filt_meta, "rgb_count": 0, "embedding_count": 0}
            continue

        rows = rgb[filtered].astype(np.float32)
        rgb_out[class_key] = rows

        emb_rows = np.zeros((0, emb_dim), dtype=np.float32)
        if unit is not None:
            patch_mask = pool_mask_to_patch_grid(filtered, hp, wp)
            if np.any(patch_mask):
                emb_rows = unit[patch_mask.ravel()]
        emb_out[class_key] = emb_rows

        report["classes"][class_key] = {
            **filt_meta,
            "rgb_count": int(rows.shape[0]),
            "embedding_count": int(emb_rows.shape[0]),
        }
        if cls_id == CLS_TALC:
            report["talc_filter"] = filt_meta

    return rgb_out, emb_out, report
