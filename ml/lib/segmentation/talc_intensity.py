"""Talc detection: dark pixels that match talc more than matrix in embedding space."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, dilation, disk, erosion

from ml.lib.calibration.filters import is_visually_black, rgb_to_gray
from ml.lib.calibration.patch_utils import unit_patch_vectors
from ml.lib.dino.inference import upsample_patch_map
from ml.lib.constants import DEFAULT_TALC_GMM_THRESHOLD_HIGH_BIAS, N_GAUSSIANS_BINARY, TALC_GMM_MIN_ACTIVATION
from ml.lib.segmentation.intensity import (
    adjacent_intersection,
    build_intensity,
    fit_gmm,
    intensity_gradient_map,
    ordered_gmm_params,
)


def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
    out = vec.astype(np.float32)
    return out / max(float(np.linalg.norm(out)), 1e-8)


def patch_cosine_similarity_map(
    block01_features: torch.Tensor | np.ndarray,
    mean_embedding: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Per-patch cosine similarity to a class mean embedding."""
    unit, _c, hp, wp = unit_patch_vectors(block01_features)
    mean_vec = _normalize_embedding(mean_embedding)
    sim = (unit @ mean_vec).reshape(hp, wp).astype(np.float32)
    return sim, (hp, wp)


def upsample_patch_cosine(
    block01_features: torch.Tensor | np.ndarray,
    mean_embedding: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Upsample patch cosine map to full image resolution."""
    sim, (hp, wp) = patch_cosine_similarity_map(block01_features, mean_embedding)
    h, w = output_shape
    sim_up = upsample_patch_map(sim, (h, w))
    return sim_up.astype(np.float32), (hp, wp)


def upsample_talc_matrix_margin(
    block01_features: torch.Tensor | np.ndarray,
    mean_talc_embedding: np.ndarray,
    mean_matrix_embedding: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Upsampled cos(talc), cos(matrix), and margin = cos(talc) − cos(matrix)."""
    cos_talc, patch_grid = upsample_patch_cosine(block01_features, mean_talc_embedding, output_shape)
    cos_matrix, _ = upsample_patch_cosine(block01_features, mean_matrix_embedding, output_shape)
    margin = (cos_talc - cos_matrix).astype(np.float32)
    return cos_talc, cos_matrix, margin, patch_grid


def segment_talc_intensity_coarse(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    *,
    fg_dilate_radius: int,
    talc_intensity_max: float,
    close_radius: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coarse talc gate: dark pixels outside dilated foreground (no embedding similarity)."""
    h, w = rgb.shape[:2]
    gray = rgb_to_gray(rgb)
    intensity_max = float(talc_intensity_max)

    fg = fg_mask.astype(bool)
    if fg_dilate_radius > 0:
        fg = dilation(fg, disk(int(fg_dilate_radius)))
    non_fg = ~fg

    dark = gray < intensity_max
    candidates = non_fg & dark

    meta: dict[str, Any] = {
        "method": "talc_intensity_coarse",
        "fg_dilate_radius": int(fg_dilate_radius),
        "talc_intensity_max": intensity_max,
        "close_radius": int(close_radius),
        "candidate_pixel_count": int(candidates.sum()),
        "pixel_count": 0,
    }

    if not np.any(candidates):
        meta["warning"] = "no dark candidates outside foreground"
        return np.zeros((h, w), dtype=np.uint8), meta

    if close_radius > 0:
        candidates = binary_closing(candidates, disk(int(close_radius)))

    meta["pixel_count"] = int(candidates.sum())
    return candidates.astype(np.uint8), meta


def segment_talc_black_threshold(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    *,
    fg_dilate_radius: int,
    talc_black_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Outside dilated FG: perceptually black pixels → talc seed mask."""
    h, w = rgb.shape[:2]
    gray = rgb_to_gray(rgb)
    black_max = float(talc_black_max)

    fg = fg_mask.astype(bool)
    if fg_dilate_radius > 0:
        fg = dilation(fg, disk(int(fg_dilate_radius)))

    non_fg = ~fg
    if not np.any(non_fg):
        return np.zeros((h, w), dtype=np.uint8), {
            "method": "talc_black_threshold",
            "warning": "empty non-foreground",
        }

    black = is_visually_black(gray, rgb, black_max=black_max)
    talc_mask = np.zeros((h, w), dtype=np.uint8)
    talc_mask[non_fg & black] = 1

    meta = {
        "method": "talc_black_threshold",
        "fg_dilate_radius": int(fg_dilate_radius),
        "talc_black_max": black_max,
        "non_fg_pixels": int(non_fg.sum()),
        "seed_pixel_count": int(talc_mask.sum()),
        "pixel_count": int(talc_mask.sum()),
    }
    return talc_mask, meta


def _filter_components_by_mean_score(
    mask: np.ndarray,
    score_map: np.ndarray,
    min_mean: float,
) -> tuple[np.ndarray, int]:
    labeled, n_regions = ndi.label(mask)
    if n_regions == 0:
        return mask, 0
    kept = np.zeros_like(mask, dtype=bool)
    rejected = 0
    for region_id in range(1, n_regions + 1):
        region = labeled == region_id
        if float(score_map[region].mean()) >= min_mean:
            kept |= region
        else:
            rejected += 1
    return kept, rejected


def segment_talc_embedding(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    block01_features: torch.Tensor | np.ndarray,
    mean_talc_embedding: np.ndarray,
    mean_matrix_embedding: np.ndarray,
    *,
    fg_dilate_radius: int,
    talc_intensity_max: float,
    min_cosine_margin: float,
    min_region_mean_margin: float,
    close_radius: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Block-11 coarse gate from talc-vs-matrix embedding similarity.

    A pixel is a candidate when it is dark, outside dilated foreground,
    cos(talc) > cos(matrix), and margin = cos(talc) − cos(matrix) meets
    min_cosine_margin (calibrated similarity strictness).
    """
    h, w = rgb.shape[:2]
    gray = rgb_to_gray(rgb)
    intensity_max = float(talc_intensity_max)
    margin_threshold = float(min_cosine_margin)
    min_region = float(min_region_mean_margin)

    fg = fg_mask.astype(bool)
    if fg_dilate_radius > 0:
        fg = dilation(fg, disk(int(fg_dilate_radius)))
    non_fg = ~fg

    cos_talc, cos_matrix, margin, patch_grid = upsample_talc_matrix_margin(
        block01_features,
        mean_talc_embedding,
        mean_matrix_embedding,
        (h, w),
    )
    dark = gray < intensity_max
    talc_wins = cos_talc > cos_matrix
    separated = margin >= margin_threshold
    candidates = non_fg & dark & talc_wins & separated

    meta: dict[str, Any] = {
        "method": "talc_embedding_margin",
        "embedding_block": 11,
        "fg_dilate_radius": int(fg_dilate_radius),
        "talc_intensity_max": intensity_max,
        "min_cosine_margin": margin_threshold,
        "min_region_mean_margin": min_region,
        "close_radius": int(close_radius),
        "patch_grid": list(patch_grid),
        "candidate_pixel_count": int(candidates.sum()),
        "rejected_low_margin_components": 0,
        "pixel_count": 0,
    }

    if not np.any(candidates):
        meta["warning"] = "no talc-favoring dark candidates"
        return np.zeros((h, w), dtype=np.uint8), meta

    if close_radius > 0:
        candidates = binary_closing(candidates, disk(int(close_radius)))

    talc_mask, rejected = _filter_components_by_mean_score(candidates, margin, min_region)
    meta["rejected_low_margin_components"] = rejected
    meta["pixel_count"] = int(talc_mask.sum())
    return talc_mask.astype(np.uint8), meta


MIN_TALC_GMM_FIT_PIXELS = 64


def compute_talc_refine_gradient_map(
    rgb: np.ndarray,
    *,
    preprocess: bool,
    denoise: bool,
    illum_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradient magnitude in [0, 1] from image intensity."""
    gray = rgb_to_gray(rgb)
    intensity = build_intensity(
        gray,
        preprocess=preprocess,
        illum_sigma=illum_sigma,
        denoise=denoise,
    )
    return intensity_gradient_map(intensity), gray


def _talc_two_gmm_threshold(
    value_map: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
    pixel_mask: np.ndarray | None,
    high_bias: float = 0.0,
    use_high_component_mean: bool = False,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """2-GMM threshold: high-component mean (DINO) or biased intersection (gradient)."""
    gmm = fit_gmm(
        value_map,
        N_GAUSSIANS_BINARY,
        max_samples,
        random_state,
        pixel_mask=pixel_mask,
    )
    means, variances, weights = ordered_gmm_params(gmm)
    unbiased = adjacent_intersection(
        means[0],
        variances[0],
        weights[0],
        means[1],
        variances[1],
        weights[1],
    )
    mean_lo = float(means[0])
    mean_hi = float(means[1])
    if use_high_component_mean:
        threshold = mean_hi
    else:
        bias = float(np.clip(high_bias, 0.0, 1.0))
        threshold = unbiased + bias * (mean_hi - unbiased)
        threshold = float(np.clip(threshold, mean_lo, mean_hi))
    return threshold, float(unbiased), means, variances, weights


def _interior_gmm_fit_mask(
    gate: np.ndarray,
    fg_mask: np.ndarray | None,
    *,
    fg_buffer_radius: int,
    gate_erode_radius: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Interior of coarse gate, excluding foreground and FG border band."""
    fit = gate.astype(bool)
    meta: dict[str, Any] = {
        "gate_pixel_count": int(fit.sum()),
        "excluded_foreground_pixels": 0,
        "excluded_fg_buffer_pixels": 0,
        "excluded_gate_rim_pixels": 0,
    }
    if fg_mask is not None:
        fg = fg_mask.astype(bool)
        before = int(fit.sum())
        fit &= ~fg
        meta["excluded_foreground_pixels"] = before - int(fit.sum())
        if fg_buffer_radius > 0:
            fg_band = dilation(fg, disk(int(fg_buffer_radius)))
            before = int(fit.sum())
            fit &= ~fg_band
            meta["excluded_fg_buffer_pixels"] = before - int(fit.sum())
    if gate_erode_radius > 0:
        before = int(fit.sum())
        fit &= erosion(gate.astype(bool), disk(int(gate_erode_radius)))
        meta["excluded_gate_rim_pixels"] = before - int(fit.sum())
    meta["gmm_fit_pixel_count"] = int(fit.sum())
    meta["gmm_fit_foreground_overlap"] = (
        int((fit & fg_mask.astype(bool)).sum()) if fg_mask is not None else 0
    )
    return fit, meta


def _fill_talc_connected_components(talc_band: np.ndarray) -> tuple[np.ndarray, int]:
    """Label high-gradient CCs and fill internal holes within each component."""
    labeled, n_comp = ndi.label(talc_band.astype(bool))
    if n_comp == 0:
        return talc_band.astype(bool), 0
    filled = np.zeros_like(talc_band, dtype=bool)
    for comp_id in range(1, n_comp + 1):
        component = labeled == comp_id
        filled |= ndi.binary_fill_holes(component)
    return filled, n_comp


def _filter_components_by_seed_overlap(
    mask: np.ndarray,
    seed: np.ndarray,
    overlap_threshold: float,
) -> tuple[np.ndarray, int, int]:
    """Keep CCs whose overlap with seed exceeds overlap_threshold."""
    labeled, n_comp = ndi.label(mask.astype(bool))
    if n_comp == 0:
        return mask.astype(bool), 0, 0
    seed_weights = seed.astype(bool).astype(np.float64).ravel()
    flat = labeled.ravel()
    sizes = np.bincount(flat)
    overlap_counts = np.bincount(flat, weights=seed_weights, minlength=sizes.shape[0])
    kept = np.zeros_like(mask, dtype=bool)
    kept_count = 0
    for comp_id in range(1, n_comp + 1):
        if sizes[comp_id] <= 0:
            continue
        overlap = overlap_counts[comp_id] / float(sizes[comp_id])
        if overlap >= overlap_threshold:
            kept |= labeled == comp_id
            kept_count += 1
    return kept, kept_count, n_comp - kept_count


def _talc_bands_from_gradient_gmm(
    gradient: np.ndarray,
    *,
    region_mask: np.ndarray | None = None,
    gmm_fit_mask: np.ndarray | None = None,
    fg_mask: np.ndarray | None = None,
    max_samples: int,
    random_state: int,
    threshold_high_bias: float = DEFAULT_TALC_GMM_THRESHOLD_HIGH_BIAS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """2-GMM on interior gradient; low gradient → background, high → talc."""
    gate = (
        np.ones(gradient.shape, dtype=bool)
        if region_mask is None
        else region_mask.astype(bool)
    )
    fg = fg_mask.astype(bool) if fg_mask is not None else None
    safe_gate = gate & ~fg if fg is not None else gate

    if not np.any(gate):
        empty = np.zeros(gradient.shape, dtype=bool)
        return empty, empty, {
            "skipped": True,
            "reason": "empty_region_mask",
            "selection": "high_gradient_talc",
        }, empty

    fit = safe_gate if gmm_fit_mask is None else gmm_fit_mask.astype(bool)
    if fg is not None:
        fit &= ~fg

    fit_fallback = False
    if int(fit.sum()) < MIN_TALC_GMM_FIT_PIXELS:
        fit = safe_gate.copy()
        fit_fallback = True

    threshold, unbiased_threshold, means, variances, weights = _talc_two_gmm_threshold(
        gradient,
        max_samples=max_samples,
        random_state=random_state,
        pixel_mask=fit,
        high_bias=threshold_high_bias,
        use_high_component_mean=False,
    )
    low_gate = (gradient < threshold) & gate
    high_gate = (gradient >= threshold) & gate
    if fg is not None:
        low_gate &= ~fg
        high_gate &= ~fg

    background_band = low_gate
    talc_band = high_gate
    meta = {
        "gmm_threshold": float(threshold),
        "gmm_threshold_unbiased": float(unbiased_threshold),
        "gmm_threshold_rule": "biased_intersection",
        "gmm_threshold_high_bias": float(np.clip(threshold_high_bias, 0.0, 1.0)),
        "gmm_means": means.tolist(),
        "gmm_variances": variances.tolist(),
        "gmm_weights": weights.tolist(),
        "gate_pixel_count": int(gate.sum()),
        "gmm_fit_pixel_count": int(fit.sum()),
        "gmm_fit_foreground_overlap": int((fit & fg).sum()) if fg is not None else 0,
        "gmm_fit_fallback": fit_fallback,
        "low_area": int(low_gate.sum()),
        "high_area": int(high_gate.sum()),
        "gradient_mean_low_band": float(means[0]),
        "gradient_mean_high_band": float(means[1]),
        "selection": "high_gradient_talc",
        "background_side": "low",
        "talc_side": "high",
    }
    return talc_band, background_band, meta, fit


def _talc_band_from_two_gmm_activation(
    activation: np.ndarray,
    gray: np.ndarray,
    *,
    region_mask: np.ndarray | None = None,
    gmm_fit_mask: np.ndarray | None = None,
    fg_mask: np.ndarray | None = None,
    max_samples: int,
    random_state: int,
    min_activation: float = TALC_GMM_MIN_ACTIVATION,
    threshold_high_bias: float = DEFAULT_TALC_GMM_THRESHOLD_HIGH_BIAS,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """2-GMM on interior activation; low activation → talc, high → background."""
    del gray, threshold_high_bias  # DINO threshold is high-component mean, not biased
    gate = (
        np.ones(activation.shape, dtype=bool)
        if region_mask is None
        else region_mask.astype(bool)
    )
    fg = fg_mask.astype(bool) if fg_mask is not None else None
    safe_gate = gate & ~fg if fg is not None else gate

    if not np.any(gate):
        empty = np.zeros(activation.shape, dtype=bool)
        return empty, {
            "skipped": True,
            "reason": "empty_region_mask",
            "selection": "low_activation_talc",
        }, empty

    act_floor = float(min_activation)
    act_ok = activation >= act_floor
    gate_low_act_excluded = int((gate & ~act_ok).sum())

    fit = safe_gate if gmm_fit_mask is None else gmm_fit_mask.astype(bool)
    if fg is not None:
        fit &= ~fg

    before_act = int(fit.sum())
    fit &= act_ok
    fit_meta: dict[str, Any] = {
        "gmm_min_activation": act_floor,
        "excluded_low_activation_pixels": gate_low_act_excluded,
        "excluded_low_activation_pixels_in_fit": before_act - int(fit.sum()),
    }
    fit_fallback = False
    if int(fit.sum()) < MIN_TALC_GMM_FIT_PIXELS:
        fit = safe_gate & act_ok
        fit_fallback = True
        fit_meta["gmm_fit_fallback"] = "insufficient_pixels_after_activation_floor"

    threshold, unbiased_threshold, means, variances, weights = _talc_two_gmm_threshold(
        activation,
        max_samples=max_samples,
        random_state=random_state,
        pixel_mask=fit,
        use_high_component_mean=True,
    )
    low_gate = (activation < threshold) & gate & act_ok
    high_gate = (activation >= threshold) & gate & act_ok
    if fg is not None:
        low_gate &= ~fg
        high_gate &= ~fg

    talc_band = low_gate
    meta = {
        "gmm_threshold": float(threshold),
        "gmm_threshold_unbiased": float(unbiased_threshold),
        "gmm_threshold_rule": "high_component_mean",
        "gmm_means": means.tolist(),
        "gmm_variances": variances.tolist(),
        "gmm_weights": weights.tolist(),
        "gate_pixel_count": int(gate.sum()),
        "gmm_fit_pixel_count": int(fit.sum()),
        "gmm_fit_foreground_overlap": int((fit & fg).sum()) if fg is not None else 0,
        "gmm_fit_fallback": fit_fallback,
        "low_area": int(low_gate.sum()),
        "high_area": int(high_gate.sum()),
        "activation_mean_low_band": float(means[0]),
        "activation_mean_high_band": float(means[1]),
        "selection": "low_activation_talc",
        "background_side": "high",
        "talc_side": "low",
        **fit_meta,
    }
    return talc_band, meta, fit


def refine_talc_with_block01_activation(
    rough_talc_mask: np.ndarray,
    block01_activation: np.ndarray,
    gray: np.ndarray,
    *,
    overlap_threshold: float = 0.4,
    max_samples: int = 300_000,
    random_state: int = 0,
    close_radius: int = 0,
    gate_dilate_radius: int = 0,
    fg_mask: np.ndarray | None = None,
    fg_dilate_radius: int = 0,
    gmm_fg_buffer_radius: int = 0,
    gmm_gate_erode_radius: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine coarse talc inside dilated gate: 2-GMM on block-1 activation."""
    del overlap_threshold  # legacy param; polarity fixed by activation
    rough = rough_talc_mask.astype(bool)
    fg_exclusion_dilated: np.ndarray | None = None
    fg_bool: np.ndarray | None = None
    if fg_mask is not None:
        fg_bool = fg_mask.astype(bool)
        fg_exclusion_dilated = fg_bool
        if fg_dilate_radius > 0:
            fg_exclusion_dilated = dilation(fg_bool, disk(int(fg_dilate_radius)))
    if gate_dilate_radius > 0:
        rough = dilation(rough, disk(int(gate_dilate_radius)))

    if not np.any(rough):
        return np.zeros(rough.shape, dtype=np.uint8), {
            "method": "talc_block01_activation_refine",
            "skipped": True,
            "reason": "empty_rough_mask",
            "pixel_count": 0,
        }

    gmm_fit_mask, fit_mask_meta = _interior_gmm_fit_mask(
        rough,
        fg_bool,
        fg_buffer_radius=gmm_fg_buffer_radius,
        gate_erode_radius=gmm_gate_erode_radius,
    )

    talc_band, gmm_meta, _gmm_fit_used = _talc_band_from_two_gmm_activation(
        block01_activation,
        gray,
        region_mask=rough,
        gmm_fit_mask=gmm_fit_mask,
        fg_mask=fg_bool,
        max_samples=max_samples,
        random_state=random_state,
    )
    if gmm_meta.get("skipped"):
        return np.zeros(rough.shape, dtype=np.uint8), {
            "method": "talc_block01_activation_refine",
            "skipped": True,
            "reason": gmm_meta.get("reason", "empty_region_mask"),
            "pixel_count": 0,
        }

    if close_radius > 0:
        talc_band = binary_closing(talc_band, disk(int(close_radius)))
    act_ok = block01_activation >= TALC_GMM_MIN_ACTIVATION
    talc_band = talc_band & rough & act_ok

    refined = talc_band
    if fg_exclusion_dilated is not None:
        refined = refined & ~fg_exclusion_dilated

    _, kept_components = ndi.label(refined.astype(bool))
    meta: dict[str, Any] = {
        "method": "talc_block01_activation_refine",
        "gate_dilate_radius": int(gate_dilate_radius),
        "fg_dilate_radius": int(fg_dilate_radius),
        "rough_pixel_count": int(rough.sum()),
        "kept_components": int(kept_components),
        "pixel_count": int(refined.sum()),
        **fit_mask_meta,
        **gmm_meta,
    }
    return refined.astype(np.uint8), meta


def refine_talc_with_image_gradient(
    rough_talc_mask: np.ndarray,
    rgb: np.ndarray,
    *,
    preprocess: bool,
    denoise: bool,
    illum_sigma: float,
    overlap_threshold: float = 0.4,
    max_samples: int = 300_000,
    random_state: int = 0,
    close_radius: int = 0,
    gate_dilate_radius: int = 0,
    fg_mask: np.ndarray | None = None,
    fg_dilate_radius: int = 0,
    gmm_fg_buffer_radius: int = 0,
    gmm_gate_erode_radius: int = 0,
    gmm_threshold_high_bias: float = DEFAULT_TALC_GMM_THRESHOLD_HIGH_BIAS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine coarse talc: 2-GMM on Sobel gradient, CC fill, background = smooth."""
    coarse_seed = rough_talc_mask.astype(bool)
    rough = coarse_seed
    fg_exclusion_dilated: np.ndarray | None = None
    fg_bool: np.ndarray | None = None
    if fg_mask is not None:
        fg_bool = fg_mask.astype(bool)
        fg_exclusion_dilated = fg_bool
        if fg_dilate_radius > 0:
            fg_exclusion_dilated = dilation(fg_bool, disk(int(fg_dilate_radius)))
    if gate_dilate_radius > 0:
        rough = dilation(rough, disk(int(gate_dilate_radius)))

    if not np.any(rough):
        return np.zeros(rough.shape, dtype=np.uint8), {
            "method": "talc_gradient_refine",
            "skipped": True,
            "reason": "empty_rough_mask",
            "pixel_count": 0,
        }

    gradient, _gray = compute_talc_refine_gradient_map(
        rgb,
        preprocess=preprocess,
        denoise=denoise,
        illum_sigma=illum_sigma,
    )

    gmm_fit_mask, fit_mask_meta = _interior_gmm_fit_mask(
        rough,
        fg_bool,
        fg_buffer_radius=gmm_fg_buffer_radius,
        gate_erode_radius=gmm_gate_erode_radius,
    )

    talc_band, _background_band, gmm_meta, _gmm_fit_used = _talc_bands_from_gradient_gmm(
        gradient,
        region_mask=rough,
        gmm_fit_mask=gmm_fit_mask,
        fg_mask=fg_bool,
        max_samples=max_samples,
        random_state=random_state,
        threshold_high_bias=gmm_threshold_high_bias,
    )
    if gmm_meta.get("skipped"):
        return np.zeros(rough.shape, dtype=np.uint8), {
            "method": "talc_gradient_refine",
            "skipped": True,
            "reason": gmm_meta.get("reason", "empty_region_mask"),
            "pixel_count": 0,
        }

    talc_band, n_components = _fill_talc_connected_components(talc_band)
    if overlap_threshold > 0 and np.any(coarse_seed):
        talc_band, kept_components, rejected_components = _filter_components_by_seed_overlap(
            talc_band,
            coarse_seed,
            overlap_threshold,
        )
    else:
        labeled, kept_components = ndi.label(talc_band.astype(bool))
        del labeled
        rejected_components = 0

    if close_radius > 0:
        talc_band = binary_closing(talc_band, disk(int(close_radius)))
    talc_band = talc_band & rough

    refined = talc_band
    if fg_exclusion_dilated is not None:
        refined = refined & ~fg_exclusion_dilated

    meta: dict[str, Any] = {
        "method": "talc_gradient_refine",
        "gate_dilate_radius": int(gate_dilate_radius),
        "fg_dilate_radius": int(fg_dilate_radius),
        "overlap_threshold": float(overlap_threshold),
        "rough_pixel_count": int(rough.sum()),
        "filled_components": int(n_components),
        "kept_components": int(kept_components),
        "rejected_components": int(rejected_components),
        "pixel_count": int(refined.sum()),
        **fit_mask_meta,
        **gmm_meta,
    }
    return refined.astype(np.uint8), meta


def segment_talc_hybrid(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    *,
    fg_dilate_radius: int,
    talc_black_max: float,
    close_radius: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coarse talc gate without embedding similarity, then optional downstream refine."""
    talc_mask, coarse_meta = segment_talc_intensity_coarse(
        rgb,
        fg_mask,
        fg_dilate_radius=fg_dilate_radius,
        talc_intensity_max=talc_black_max,
        close_radius=close_radius,
    )
    meta = {
        "method": "talc_hybrid",
        "coarse": coarse_meta,
        "pixel_count": int(talc_mask.sum()),
    }
    return talc_mask, meta
