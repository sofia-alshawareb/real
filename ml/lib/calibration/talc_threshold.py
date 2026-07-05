"""Calibrate talc embedding cosine thresholds from sampled talc vs matrix labels.

Labels are partial examples only — blue samples talc-like areas (may include some
background), grey samples smooth matrix background. Thresholds are derived from
pooled pixel statistics, not spatial coverage of the full image.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.lib.calibration.filters import rgb_to_gray
from ml.lib.calibration.patch_utils import pool_mask_to_patch_grid, unit_patch_vectors
from ml.lib.calibration.types import CalibrationData
from ml.lib.constants import (
    CLS_MATRIX,
    CLS_TALC,
    DEFAULT_TALC_BLACK_MAX,
    DEFAULT_TALC_MIN_COSINE,
    DEFAULT_TALC_MIN_COSINE_MARGIN,
)
from ml.lib.segmentation.talc_intensity import patch_cosine_similarity_map


def background_mask_for_talc_calibration(class_masks: dict[int, np.ndarray]) -> np.ndarray | None:
    """Matrix (grey) sample regions, excluding talc overlap — background for calib."""
    if CLS_MATRIX not in class_masks:
        return None
    bg = class_masks[CLS_MATRIX].astype(bool).copy()
    if CLS_TALC in class_masks:
        bg &= ~class_masks[CLS_TALC].astype(bool)
    if not np.any(bg):
        return None
    return bg


def collect_embedding_vectors(
    block01_features: np.ndarray,
    class_masks: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (talc_unit_vectors, background_unit_vectors, meta) from sample masks."""
    meta: dict[str, Any] = {"skipped": False}
    talc_mask = class_masks.get(CLS_TALC)
    bg_mask = background_mask_for_talc_calibration(class_masks)
    if talc_mask is None or not np.any(talc_mask):
        return np.zeros((0, 384), np.float32), np.zeros((0, 384), np.float32), {
            **meta,
            "skipped": True,
            "reason": "no_talc",
        }
    if bg_mask is None:
        return np.zeros((0, 384), np.float32), np.zeros((0, 384), np.float32), {
            **meta,
            "skipped": True,
            "reason": "no_background",
        }

    unit, emb_dim, hp, wp = unit_patch_vectors(block01_features)
    talc_patch = pool_mask_to_patch_grid(talc_mask.astype(bool), hp, wp)
    bg_patch = pool_mask_to_patch_grid(bg_mask, hp, wp)
    talc_vecs = unit[talc_patch.ravel()]
    bg_vecs = unit[bg_patch.ravel()]
    meta.update(
        {
            "n_talc_patches": int(talc_vecs.shape[0]),
            "n_background_patches": int(bg_vecs.shape[0]),
            "embedding_dim": int(emb_dim),
        }
    )
    return talc_vecs, bg_vecs, meta


def collect_margin_samples(
    block01_features: np.ndarray,
    class_masks: dict[int, np.ndarray],
    talc_mean_embedding: np.ndarray,
    matrix_mean_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (talc_margins, background_margins, meta) where margin = cos(talc) − cos(matrix)."""
    meta: dict[str, Any] = {"skipped": False}
    talc_mask = class_masks.get(CLS_TALC)
    bg_mask = background_mask_for_talc_calibration(class_masks)
    if talc_mask is None or not np.any(talc_mask):
        return np.zeros(0, np.float32), np.zeros(0, np.float32), {**meta, "skipped": True, "reason": "no_talc"}
    if bg_mask is None:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), {**meta, "skipped": True, "reason": "no_background"}

    cos_talc, (hp, wp) = patch_cosine_similarity_map(block01_features, talc_mean_embedding)
    cos_matrix, _ = patch_cosine_similarity_map(block01_features, matrix_mean_embedding)
    margin = (cos_talc - cos_matrix).astype(np.float32)
    talc_patch = pool_mask_to_patch_grid(talc_mask.astype(bool), hp, wp)
    bg_patch = pool_mask_to_patch_grid(bg_mask, hp, wp)
    talc_vals = margin[talc_patch]
    bg_vals = margin[bg_patch]
    meta.update(
        {
            "n_talc_patches": int(talc_vals.size),
            "n_background_patches": int(bg_vals.size),
            "talc_margin_median": float(np.median(talc_vals)) if talc_vals.size else None,
            "background_margin_median": float(np.median(bg_vals)) if bg_vals.size else None,
            "talc_cosine_median": float(np.median(cos_talc[talc_patch])) if talc_vals.size else None,
            "background_cosine_median": float(np.median(cos_talc[bg_patch])) if bg_vals.size else None,
        }
    )
    return talc_vals, bg_vals, meta


def collect_cosine_samples(
    block01_features: np.ndarray,
    class_masks: dict[int, np.ndarray],
    talc_mean_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (talc_cosines, background_cosines, meta) vs mean talc embedding."""
    meta: dict[str, Any] = {"skipped": False}
    talc_mask = class_masks.get(CLS_TALC)
    bg_mask = background_mask_for_talc_calibration(class_masks)
    if talc_mask is None or not np.any(talc_mask):
        return np.zeros(0, np.float32), np.zeros(0, np.float32), {**meta, "skipped": True, "reason": "no_talc"}
    if bg_mask is None:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), {**meta, "skipped": True, "reason": "no_background"}

    sim, (hp, wp) = patch_cosine_similarity_map(block01_features, talc_mean_embedding)
    talc_patch = pool_mask_to_patch_grid(talc_mask.astype(bool), hp, wp)
    bg_patch = pool_mask_to_patch_grid(bg_mask, hp, wp)
    talc_vals = sim[talc_patch].astype(np.float32)
    bg_vals = sim[bg_patch].astype(np.float32)
    meta.update(
        {
            "n_talc_patches": int(talc_vals.size),
            "n_background_patches": int(bg_vals.size),
            "talc_median": float(np.median(talc_vals)) if talc_vals.size else None,
            "background_median": float(np.median(bg_vals)) if bg_vals.size else None,
        }
    )
    return talc_vals, bg_vals, meta


def collect_intensity_samples(
    rgb: np.ndarray,
    class_masks: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (talc_gray, background_gray, meta) from painted sample masks."""
    meta: dict[str, Any] = {"skipped": False}
    talc_mask = class_masks.get(CLS_TALC)
    bg_mask = background_mask_for_talc_calibration(class_masks)
    if talc_mask is None or not np.any(talc_mask):
        return np.zeros(0, np.float32), np.zeros(0, np.float32), {**meta, "skipped": True, "reason": "no_talc"}
    if bg_mask is None:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), {**meta, "skipped": True, "reason": "no_background"}

    gray = rgb_to_gray(rgb)
    talc_vals = gray[talc_mask.astype(bool)].astype(np.float32)
    bg_vals = gray[bg_mask].astype(np.float32)
    meta.update(
        {
            "n_talc_pixels": int(talc_vals.size),
            "n_background_pixels": int(bg_vals.size),
            "talc_median": float(np.median(talc_vals)) if talc_vals.size else None,
            "background_median": float(np.median(bg_vals)) if bg_vals.size else None,
        }
    )
    return talc_vals, bg_vals, meta


def compute_talc_min_cosine_margin(
    talc_margins: np.ndarray,
    background_margins: np.ndarray,
    *,
    fallback: float = DEFAULT_TALC_MIN_COSINE_MARGIN,
    min_patches: int = 20,
) -> tuple[float, dict[str, Any]]:
    """Derive margin cutoff: cos(talc) − cos(matrix) separates sample patches."""
    meta: dict[str, Any] = {
        "method": "calibrated_margin_midpoint",
        "n_talc_patches": int(talc_margins.size),
        "n_background_patches": int(background_margins.size),
    }
    if talc_margins.size < min_patches or background_margins.size < min_patches:
        meta["method"] = "fallback_insufficient_samples"
        meta["min_cosine_margin"] = float(fallback)
        meta["fallback"] = float(fallback)
        return float(fallback), meta

    med_t = float(np.median(talc_margins))
    med_b = float(np.median(background_margins))
    p25_t = float(np.percentile(talc_margins, 25))
    p75_b = float(np.percentile(background_margins, 75))

    if med_t > med_b:
        threshold = (med_b + med_t) / 2.0
    else:
        threshold = (p75_b + p25_t) / 2.0

    min_region = float((med_b + med_t) / 2.0)
    meta.update(
        {
            "min_cosine_margin": threshold,
            "min_region_mean_margin": min_region,
            "talc_margin_median": med_t,
            "background_margin_median": med_b,
            "talc_margin_p25": p25_t,
            "background_margin_p75": p75_b,
        }
    )
    return threshold, meta


def compute_talc_min_cosine(
    talc_cosines: np.ndarray,
    background_cosines: np.ndarray,
    *,
    fallback: float = DEFAULT_TALC_MIN_COSINE,
    min_patches: int = 20,
) -> tuple[float, dict[str, Any]]:
    """Derive cosine cutoff between matrix background and talc sample patches."""
    meta: dict[str, Any] = {
        "method": "calibrated_midpoint",
        "n_talc_patches": int(talc_cosines.size),
        "n_background_patches": int(background_cosines.size),
    }
    if talc_cosines.size < min_patches or background_cosines.size < min_patches:
        meta["method"] = "fallback_insufficient_samples"
        meta["min_cosine"] = float(fallback)
        meta["fallback"] = float(fallback)
        return float(fallback), meta

    med_t = float(np.median(talc_cosines))
    med_b = float(np.median(background_cosines))
    p90_b = float(np.percentile(background_cosines, 90))
    p75_t = float(np.percentile(talc_cosines, 75))

    if med_t > med_b:
        threshold = (med_b + med_t) / 2.0
    else:
        threshold = (p90_b + p75_t) / 2.0

    threshold = float(np.clip(threshold, -1.0, 1.0))
    min_region = float((med_b + med_t) / 2.0)
    meta.update(
        {
            "min_cosine": threshold,
            "min_region_mean_cosine": min_region,
            "talc_median": med_t,
            "background_median": med_b,
            "talc_p75": p75_t,
            "background_p90": p90_b,
        }
    )
    return threshold, meta


def compute_talc_intensity_max(
    talc_intensities: np.ndarray,
    background_intensities: np.ndarray,
    *,
    fallback: float = DEFAULT_TALC_BLACK_MAX,
    min_pixels: int = 50,
) -> tuple[float, dict[str, Any]]:
    """Derive gray ceiling for talc search region from sample pixel intensities."""
    meta: dict[str, Any] = {
        "n_talc_pixels": int(talc_intensities.size),
        "n_background_pixels": int(background_intensities.size),
    }
    if talc_intensities.size < min_pixels:
        meta["method"] = "fallback_insufficient_talc_samples"
        meta["talc_intensity_max"] = float(fallback)
        return float(fallback), meta

    p90_t = float(np.percentile(talc_intensities, 90))
    med_t = float(np.median(talc_intensities))
    med_b = float(np.median(background_intensities)) if background_intensities.size >= min_pixels else None

    intensity_max = p90_t
    if med_b is not None and med_b > med_t:
        intensity_max = float((med_t + med_b) / 2.0)

    meta.update(
        {
            "method": "calibrated_p90_or_midpoint",
            "talc_intensity_max": intensity_max,
            "talc_median": med_t,
            "talc_p90": p90_t,
            "background_median": med_b,
        }
    )
    return intensity_max, meta


def _normalize_mean(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.zeros(384, dtype=np.float32)
    mean = rows.mean(axis=0).astype(np.float32)
    return mean / max(float(np.linalg.norm(mean)), 1e-8)


def calibrate_talc_from_labeled_images(
    image_samples: list[tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], str]],
    *,
    fallback_cosine: float = DEFAULT_TALC_MIN_COSINE,
    fallback_intensity: float = DEFAULT_TALC_BLACK_MAX,
    reference_mean_embedding: np.ndarray | None = None,
    reference_matrix_mean_embedding: np.ndarray | None = None,
) -> dict[str, Any]:
    """Aggregate sample patches across images → talc_contour calibration block."""
    talc_vec_parts: list[np.ndarray] = []
    talc_int_parts: list[np.ndarray] = []
    bg_int_parts: list[np.ndarray] = []
    per_image: dict[str, Any] = {}

    for rgb, block01_features, class_masks, image_id in image_samples:
        talc_vecs, _bg_vecs, vec_meta = collect_embedding_vectors(block01_features, class_masks)
        talc_i, bg_i, int_meta = collect_intensity_samples(rgb, class_masks)
        per_image[image_id] = {"embedding": vec_meta, "intensity": int_meta}
        if vec_meta.get("skipped"):
            continue
        if talc_vecs.size:
            talc_vec_parts.append(talc_vecs)
        talc_int_parts.append(talc_i)
        bg_int_parts.append(bg_i)

    if reference_mean_embedding is not None:
        talc_mean = reference_mean_embedding.astype(np.float32)
        talc_mean /= max(float(np.linalg.norm(talc_mean)), 1e-8)
    elif talc_vec_parts:
        talc_mean = _normalize_mean(np.vstack(talc_vec_parts))
    else:
        talc_mean = None

    if talc_mean is None:
        return {
            "method": "fallback_no_labeled_pairs",
            "min_cosine": float(fallback_cosine),
            "talc_intensity_max": float(fallback_intensity),
            "min_region_mean_cosine": float(fallback_cosine),
            "per_image": per_image,
            "source": "sample_regions",
        }

    if talc_mean is None:
        return {
            "method": "fallback_no_labeled_pairs",
            "min_cosine_margin": float(DEFAULT_TALC_MIN_COSINE_MARGIN),
            "talc_intensity_max": float(fallback_intensity),
            "min_region_mean_margin": float(DEFAULT_TALC_MIN_COSINE_MARGIN),
            "per_image": per_image,
            "source": "sample_regions",
        }

    if reference_matrix_mean_embedding is not None:
        matrix_mean = reference_matrix_mean_embedding.astype(np.float32)
        matrix_mean /= max(float(np.linalg.norm(matrix_mean)), 1e-8)
    else:
        matrix_parts: list[np.ndarray] = []
        for _rgb, block01_features, class_masks, _image_id in image_samples:
            _talc_vecs, bg_vecs, vec_meta = collect_embedding_vectors(block01_features, class_masks)
            if not vec_meta.get("skipped") and bg_vecs.size:
                matrix_parts.append(bg_vecs)
        if not matrix_parts:
            return {
                "method": "fallback_no_matrix_mean",
                "min_cosine_margin": float(DEFAULT_TALC_MIN_COSINE_MARGIN),
                "talc_intensity_max": float(fallback_intensity),
                "min_region_mean_margin": float(DEFAULT_TALC_MIN_COSINE_MARGIN),
                "per_image": per_image,
                "source": "sample_regions",
            }
        matrix_mean = _normalize_mean(np.vstack(matrix_parts))

    talc_margin_parts: list[np.ndarray] = []
    bg_margin_parts: list[np.ndarray] = []
    for _rgb, block01_features, class_masks, image_id in image_samples:
        talc_m, bg_m, margin_meta = collect_margin_samples(
            block01_features, class_masks, talc_mean, matrix_mean
        )
        per_image.setdefault(image_id, {})["margin"] = margin_meta
        if margin_meta.get("skipped"):
            continue
        talc_margin_parts.append(talc_m)
        bg_margin_parts.append(bg_m)

    if not talc_margin_parts or not bg_margin_parts:
        return {
            "method": "fallback_no_margin_pairs",
            "min_cosine_margin": float(DEFAULT_TALC_MIN_COSINE_MARGIN),
            "talc_intensity_max": float(fallback_intensity),
            "min_region_mean_margin": float(DEFAULT_TALC_MIN_COSINE_MARGIN),
            "per_image": per_image,
            "source": "sample_regions",
        }

    talc_margin_all = np.concatenate(talc_margin_parts)
    bg_margin_all = np.concatenate(bg_margin_parts)
    _, margin_meta = compute_talc_min_cosine_margin(talc_margin_all, bg_margin_all)

    talc_int_all = np.concatenate(talc_int_parts) if talc_int_parts else np.zeros(0, np.float32)
    bg_int_all = np.concatenate(bg_int_parts) if bg_int_parts else np.zeros(0, np.float32)
    intensity_max, int_meta = compute_talc_intensity_max(
        talc_int_all, bg_int_all, fallback=fallback_intensity
    )

    return {
        **margin_meta,
        **{k: v for k, v in int_meta.items() if k not in margin_meta},
        "talc_intensity_max": intensity_max,
        "per_image": per_image,
        "source": "sample_regions",
        "n_images": len(talc_margin_parts),
        "embedding_block": 11,
    }


def _talc_contour_meta(calib: CalibrationData) -> dict[str, Any] | None:
    if not calib.meta:
        return None
    tc = calib.meta.get("talc_contour")
    return tc if isinstance(tc, dict) else None


def resolve_talc_min_cosine_margin(
    calib: CalibrationData,
    config_fallback: float = DEFAULT_TALC_MIN_COSINE_MARGIN,
) -> float:
    """Use compiled talc-vs-matrix margin threshold when available."""
    tc = _talc_contour_meta(calib)
    if tc and tc.get("min_cosine_margin") is not None:
        return float(tc["min_cosine_margin"])
    return float(config_fallback)


def resolve_min_region_mean_margin(
    calib: CalibrationData,
    min_cosine_margin: float,
) -> float:
    """Minimum mean talc-matrix margin for a kept talc blob."""
    tc = _talc_contour_meta(calib)
    if tc:
        if tc.get("min_region_mean_margin") is not None:
            return float(tc["min_region_mean_margin"])
        if tc.get("min_region_mean_cosine") is not None:
            return float(tc["min_region_mean_cosine"])
    return float(min_cosine_margin)


def resolve_talc_min_cosine(
    calib: CalibrationData,
    config_fallback: float,
) -> float:
    """Use compiled calibration cosine threshold when available, else config default."""
    tc = _talc_contour_meta(calib)
    if tc:
        if tc.get("min_cosine") is not None:
            return float(tc["min_cosine"])
        if tc.get("gradient_threshold") is not None:
            return float(tc["gradient_threshold"])
    return float(config_fallback)


def resolve_talc_intensity_max(
    calib: CalibrationData,
    config_fallback: float,
) -> float:
    """Use calibrated gray ceiling for talc search region when available."""
    tc = _talc_contour_meta(calib)
    if tc and tc.get("talc_intensity_max") is not None:
        return float(tc["talc_intensity_max"])
    if tc and tc.get("talc_black_max") is not None:
        return float(tc["talc_black_max"])
    return float(config_fallback)


def resolve_min_region_mean_cosine(
    calib: CalibrationData,
    min_cosine: float,
) -> float:
    """Minimum mean patch cosine for a kept talc blob."""
    tc = _talc_contour_meta(calib)
    if tc:
        if tc.get("min_region_mean_cosine") is not None:
            return float(tc["min_region_mean_cosine"])
        if tc.get("min_region_mean_gradient") is not None:
            return float(tc["min_region_mean_gradient"])
    return float(min_cosine)


def resolve_talc_matrix_mean_embedding(calib: CalibrationData) -> np.ndarray | None:
    """Mean matrix embedding from compiled calibration stats."""
    stats = calib.stats.get("matrix")
    if stats is None or stats.mean_embedding is None:
        return None
    vec = stats.mean_embedding.astype(np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-8)


def resolve_talc_mean_embedding(calib: CalibrationData) -> np.ndarray | None:
    """Mean talc embedding from compiled calibration stats."""
    stats = calib.stats.get("talc")
    if stats is None or stats.mean_embedding is None:
        return None
    vec = stats.mean_embedding.astype(np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-8)
