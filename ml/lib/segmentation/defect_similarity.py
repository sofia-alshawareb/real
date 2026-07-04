"""Defect detection via block-11 patch similarity."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ml.lib.constants import (
    CLS_FOREGROUND,
    CLS_PARTITIONS,
    N_GAUSSIANS_DEFECT_SIM,
    PATCH_SIZE,
)
from ml.lib.dino.inference import upsample_patch_map
from ml.lib.segmentation.intensity import (
    build_intensity,
    fit_gmm,
    ordered_gmm_params,
    thresholds_from_adjacent_intersections,
    two_gmm_threshold,
)
from ml.lib.segmentation.regions import (
    build_final_segmentation,
    segment_by_thresholds,
)
from ml.lib.types import SegmentConfig, SegmentationResult


def _rgb_to_gray01(rgb: np.ndarray) -> np.ndarray:
    return np.mean(rgb.astype(np.float32) / 255.0, axis=2)


def _pool_mask_to_patch_grid(mask: np.ndarray, hp: int, wp: int) -> np.ndarray:
    """True where any pixel in the patch overlaps the mask."""
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


def _unit_patch_vectors(block11_features: torch.Tensor) -> tuple[np.ndarray, int, int, int]:
    feats = block11_features.detach().float().cpu()
    c, hp, wp = feats.shape
    vectors = feats.reshape(c, -1).T.numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-8)
    return unit, c, hp, wp


def _patch_activation_grid(feats: torch.Tensor) -> np.ndarray:
    return torch.linalg.vector_norm(feats, dim=0).numpy().astype(np.float32)


def _normalize_reference(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)


def reference_from_min_intensity(
    intensity: np.ndarray,
    fg_mask: np.ndarray,
    unit: np.ndarray,
    hp: int,
    wp: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    masked = intensity.copy()
    masked[fg_mask.astype(bool)] = np.inf
    if not np.isfinite(masked).any():
        ref_vec = unit[0]
        return ref_vec, {
            "method": "block11_min_intensity_patch_similarity",
            "reference_patch_row_col": [0, 0],
            "reference_intensity": float("nan"),
        }
    flat_idx = int(np.argmin(masked))
    min_y, min_x = np.unravel_index(flat_idx, intensity.shape)
    pr = min(int(min_y // PATCH_SIZE), hp - 1)
    pc = min(int(min_x // PATCH_SIZE), wp - 1)
    ref_idx = pr * wp + pc
    return unit[ref_idx], {
        "method": "block11_min_intensity_patch_similarity",
        "reference_patch_row_col": [pr, pc],
        "reference_intensity": float(intensity[min_y, min_x]),
    }


def _mean_patch_intensity(
    intensity: np.ndarray,
    hp: int,
    wp: int,
) -> np.ndarray:
    h, w = intensity.shape
    out = np.zeros((hp, wp), dtype=np.float32)
    for pr in range(hp):
        y0 = pr * PATCH_SIZE
        y1 = min((pr + 1) * PATCH_SIZE, h)
        for pc in range(wp):
            x0 = pc * PATCH_SIZE
            x1 = min((pc + 1) * PATCH_SIZE, w)
            out[pr, pc] = float(intensity[y0:y1, x0:x1].mean())
    return out


def reference_from_user_hint(
    intensity: np.ndarray,
    hint_mask: np.ndarray,
    fg_mask: np.ndarray,
    block11_features: torch.Tensor,
    *,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Average block-11 vectors of lowest-intensity 2-GMM component inside hint."""
    unit, _c, hp, wp = _unit_patch_vectors(block11_features)
    hint_bool = hint_mask.astype(bool)
    if not np.any(hint_bool):
        raise ValueError("hint mask is empty")

    fg_patch = _pool_mask_to_patch_grid(fg_mask, hp, wp)
    hint_patch = _pool_mask_to_patch_grid(hint_bool.astype(np.uint8), hp, wp)
    candidate_patch = hint_patch & ~fg_patch

    threshold, means, variances, weights = two_gmm_threshold(
        intensity,
        max_samples=max_samples,
        random_state=random_state,
        pixel_mask=hint_bool & (fg_mask == 0),
    )

    patch_intensity = _mean_patch_intensity(intensity, hp, wp)
    low_intensity_patch = (patch_intensity < threshold) & candidate_patch

    if not np.any(low_intensity_patch):
        low_intensity_patch = candidate_patch

    if not np.any(low_intensity_patch):
        raise ValueError("no valid hint patches outside foreground")

    flat_indices = np.flatnonzero(low_intensity_patch.ravel())
    ref_vec = _normalize_reference(unit[flat_indices].mean(axis=0))

    return ref_vec, {
        "method": "user_hint_2gmm_mean",
        "hint_pixels": int(hint_bool.sum()),
        "reference_patch_count": int(flat_indices.size),
        "intensity_gmm2": {
            "threshold": float(threshold),
            "means": means.tolist(),
            "variances": variances.tolist(),
            "weights": weights.tolist(),
        },
    }


def defect_mask_from_block11_similarity(
    intensity: np.ndarray,
    block11_features: torch.Tensor,
    fg_mask: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
    reference_vector: np.ndarray | None = None,
    reference_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find defects using 4-band GMM on cosine similarity to a reference patch vector."""
    h, w = intensity.shape
    feats = block11_features.detach().float().cpu()
    unit, _c, hp, wp = _unit_patch_vectors(block11_features)

    fg_patch = _pool_mask_to_patch_grid(fg_mask, hp, wp)

    if reference_vector is not None:
        ref_vec = _normalize_reference(reference_vector)
        ref_meta = dict(reference_meta or {})
    else:
        ref_vec, ref_meta = reference_from_min_intensity(intensity, fg_mask, unit, hp, wp)

    sim_grid = (unit @ ref_vec).reshape(hp, wp).astype(np.float32)
    act_grid = _patch_activation_grid(feats)

    valid = ~fg_patch
    valid_sims = sim_grid[valid]
    if valid_sims.size < N_GAUSSIANS_DEFECT_SIM:
        empty = np.zeros((h, w), dtype=np.uint8)
        meta: dict[str, Any] = {
            **ref_meta,
            "error": "insufficient_non_fg_patches",
        }
        return empty, meta

    gmm = fit_gmm(
        sim_grid,
        N_GAUSSIANS_DEFECT_SIM,
        max_samples,
        random_state,
        pixel_mask=valid,
    )
    means, variances, weights = ordered_gmm_params(gmm)
    thresholds = thresholds_from_adjacent_intersections(means, variances, weights)
    sim_labels = segment_by_thresholds(sim_grid, thresholds)

    band_ids = list(range(N_GAUSSIANS_DEFECT_SIM))
    band_mean_activation: dict[int, float] = {}
    for band_id in band_ids:
        band_mask = (sim_labels == band_id) & valid
        if np.any(band_mask):
            band_mean_activation[band_id] = float(act_grid[band_mask].mean())
        else:
            band_mean_activation[band_id] = float("-inf")

    defect_band = max(band_ids, key=lambda b: band_mean_activation[b])
    defect_patch = (sim_labels == defect_band) & valid
    defect_up = upsample_patch_map(defect_patch.astype(np.float32), (h, w)) >= 0.5
    non_fg = fg_mask == 0
    defect_mask = (defect_up & non_fg).astype(np.uint8)

    meta = {
        **ref_meta,
        "similarity_gmm4": {
            "means": means.tolist(),
            "variances": variances.tolist(),
            "weights": weights.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "band_mean_block11_activation": {
            str(k): v for k, v in band_mean_activation.items()
        },
        "defect_similarity_band": int(defect_band),
        "defect_pixels": int(defect_mask.sum()),
    }
    return defect_mask, meta


def refine_defect_labels(
    rgb: np.ndarray,
    labels: np.ndarray,
    block11_features: torch.Tensor,
    hint_mask: np.ndarray,
    config: SegmentConfig | None = None,
) -> SegmentationResult:
    """Re-run global defect detection using a user-hint reference; keep FG/partitions."""
    cfg = config or SegmentConfig()
    h, w = rgb.shape[:2]

    fg_object = (labels == CLS_FOREGROUND).astype(np.uint8)
    partitions = (labels == CLS_PARTITIONS).astype(np.uint8)
    fg_mask = ((fg_object > 0) | (partitions > 0)).astype(np.uint8)

    gray = _rgb_to_gray01(rgb)
    intensity = build_intensity(
        gray,
        preprocess=cfg.preprocess,
        illum_sigma=cfg.illum_sigma,
        denoise=cfg.denoise,
    )

    ref_vec, ref_meta = reference_from_user_hint(
        intensity,
        hint_mask,
        fg_mask,
        block11_features,
        max_samples=cfg.max_samples,
        random_state=cfg.random_state,
    )

    defect_mask, defect_meta = defect_mask_from_block11_similarity(
        intensity,
        block11_features,
        fg_mask,
        max_samples=cfg.max_samples,
        random_state=cfg.random_state,
        reference_vector=ref_vec,
        reference_meta=ref_meta,
    )

    seg_raw = build_final_segmentation(fg_object, partitions, defect_mask)
    new_labels = seg_raw.astype(np.uint8)

    refinement_meta = {
        "reference": ref_meta,
        "defect_detection": defect_meta,
    }

    return SegmentationResult(
        labels=new_labels.astype(np.uint8),
        native_width=w,
        native_height=h,
        mask_width=w,
        mask_height=h,
        mask_to_native_scale=1.0,
        metadata={
            "refinement": refinement_meta,
            "defect_detection": defect_meta,
        },
    )
