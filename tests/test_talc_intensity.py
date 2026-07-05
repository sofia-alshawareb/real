"""Tests for talc detection: dark coarse gate + gradient refine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.lib.segmentation.talc_intensity import (
    _interior_gmm_fit_mask,
    _talc_bands_from_gradient_gmm,
    _talc_two_gmm_threshold,
    compute_talc_refine_gradient_map,
    patch_cosine_similarity_map,
    refine_talc_with_block01_activation,
    refine_talc_with_image_gradient,
    segment_talc_black_threshold,
    segment_talc_embedding,
    segment_talc_hybrid,
)
from skimage.morphology import dilation, disk


def _textured_patch(rgb: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    for i in range(r0, r1):
        for j in range(c0, c1):
            v = int(60 + (i * 7 + j * 11) % 80)
            rgb[i, j] = (v, v, v)


def test_dark_pixels_outside_dilated_fg_become_talc_seed():
    h, w = 128, 128
    rgb = np.full((h, w, 3), (40, 180, 40), dtype=np.uint8)
    rgb[20:40, 20:40] = (10, 10, 10)
    rgb[60:80, 60:80] = (35, 170, 35)

    fg = np.zeros((h, w), dtype=bool)
    fg[60:80, 60:80] = True

    talc_mask, meta = segment_talc_black_threshold(
        rgb,
        fg,
        fg_dilate_radius=3,
        talc_black_max=45.0,
    )

    assert meta["method"] == "talc_black_threshold"
    assert talc_mask[fg].sum() == 0
    assert talc_mask[20:40, 20:40].any()
    assert meta["seed_pixel_count"] > 0


def test_green_background_stays_empty_seed():
    h, w = 64, 64
    rgb = np.full((h, w, 3), (50, 200, 50), dtype=np.uint8)
    fg = np.zeros((h, w), dtype=bool)
    fg[20:40, 20:40] = True

    talc_mask, meta = segment_talc_black_threshold(
        rgb,
        fg,
        fg_dilate_radius=2,
        talc_black_max=45.0,
    )

    assert meta["seed_pixel_count"] == 0


def test_patch_cosine_similarity_to_mean():
    feats = torch.zeros(4, 2, 2)
    feats[:, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mean = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sim, (hp, wp) = patch_cosine_similarity_map(feats, mean)
    assert hp == 2 and wp == 2
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[1, 1] == pytest.approx(0.0)


def test_embedding_margin_rejects_matrix_like_dark_region():
    h, w = 112, 112
    rgb = np.full((h, w, 3), (40, 180, 40), dtype=np.uint8)
    rgb[10:50, 10:50] = (8, 8, 8)
    fg = np.zeros((h, w), dtype=bool)
    fg[60:80, 60:80] = True
    embed_dim = 8
    feats = torch.zeros(embed_dim, 8, 8)
    talc_mean = np.zeros(embed_dim, dtype=np.float32)
    talc_mean[0] = 1.0
    matrix_mean = np.zeros(embed_dim, dtype=np.float32)
    matrix_mean[1] = 1.0
    feats[:, 1:7, 1:7] = torch.from_numpy(talc_mean).reshape(embed_dim, 1, 1)

    talc_mask, meta = segment_talc_embedding(
        rgb,
        fg,
        feats,
        talc_mean,
        matrix_mean,
        fg_dilate_radius=3,
        talc_intensity_max=45.0,
        min_cosine_margin=0.05,
        min_region_mean_margin=0.04,
        close_radius=1,
    )

    assert meta["method"] == "talc_embedding_margin"
    assert meta["pixel_count"] > 0
    assert talc_mask[30, 30]

    rgb_matrix = rgb.copy()
    feats_matrix = torch.zeros(embed_dim, 8, 8)
    feats_matrix[:, 1:7, 1:7] = torch.from_numpy(matrix_mean).reshape(embed_dim, 1, 1)
    bg_mask, bg_meta = segment_talc_embedding(
        rgb_matrix,
        fg,
        feats_matrix,
        talc_mean,
        matrix_mean,
        fg_dilate_radius=3,
        talc_intensity_max=45.0,
        min_cosine_margin=0.05,
        min_region_mean_margin=0.04,
        close_radius=1,
    )
    assert bg_meta["pixel_count"] == 0
    assert bg_mask.sum() == 0


def test_hybrid_pipeline_uses_intensity_coarse_gate():
    h, w = 112, 112
    rgb = np.full((h, w, 3), (40, 180, 40), dtype=np.uint8)
    rgb[10:50, 10:50] = (8, 8, 8)
    fg = np.zeros((h, w), dtype=bool)
    fg[60:80, 60:80] = True

    talc_mask, meta = segment_talc_hybrid(
        rgb,
        fg,
        fg_dilate_radius=3,
        talc_black_max=45.0,
        close_radius=0,
    )

    assert meta["method"] == "talc_hybrid"
    assert meta["coarse"]["method"] == "talc_intensity_coarse"
    assert meta["pixel_count"] > 0
    assert talc_mask[30, 30]
    assert talc_mask[fg].sum() == 0


def test_gradient_refine_selects_high_gradient_as_talc():
    h, w = 128, 128
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    _textured_patch(rgb, 20, 50, 20, 50)

    rough = np.zeros((h, w), dtype=np.uint8)
    rough[10:70, 10:70] = 1

    refined, meta = refine_talc_with_image_gradient(
        rough,
        rgb,
        preprocess=False,
        denoise=False,
        illum_sigma=30.0,
        overlap_threshold=0.0,
        close_radius=0,
        random_state=0,
    )

    assert meta["method"] == "talc_gradient_refine"
    assert meta["selection"] == "high_gradient_talc"
    assert meta["talc_side"] == "high"
    assert meta["background_side"] == "low"
    assert refined[35, 35]
    assert meta["pixel_count"] > 0
    assert meta["pixel_count"] < int(rough.sum())


def test_gradient_refine_smooth_region_is_background():
    h, w = 128, 128
    rgb = np.full((h, w, 3), 80, dtype=np.uint8)
    _textured_patch(rgb, 70, 100, 70, 100)

    rough = np.ones((h, w), dtype=np.uint8)

    refined, meta = refine_talc_with_image_gradient(
        rough,
        rgb,
        preprocess=False,
        denoise=False,
        illum_sigma=30.0,
        overlap_threshold=0.0,
        close_radius=0,
        random_state=0,
    )

    assert meta["talc_side"] == "high"
    assert meta["background_side"] == "low"
    assert not refined[40, 40]
    assert refined[85, 85]


def test_gmm_fit_excludes_fg_border_gradient_spikes():
    h, w = 128, 128
    gate = np.zeros((h, w), dtype=bool)
    gate[15:95, 15:95] = True

    fg = np.zeros((h, w), dtype=bool)
    fg[40:80, 40:80] = True

    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    _textured_patch(rgb, 25, 35, 25, 35)
    fg_edge = dilation(fg, disk(3)) & ~fg
    edge_in_gate = fg_edge & gate
    for idx in np.argwhere(edge_in_gate):
        i, j = int(idx[0]), int(idx[1])
        v = 20 + (i + j) % 200
        rgb[i, j] = (v, v, v)
    gradient, _ = compute_talc_refine_gradient_map(
        rgb, preprocess=False, denoise=False, illum_sigma=30.0
    )

    fit, fit_meta = _interior_gmm_fit_mask(
        gate,
        fg,
        fg_buffer_radius=8,
        gate_erode_radius=2,
    )
    assert not np.any(fit & fg_edge)
    assert fit_meta["excluded_fg_buffer_pixels"] > 0

    talc_band, _bg, meta, _fit = _talc_bands_from_gradient_gmm(
        gradient,
        region_mask=gate,
        gmm_fit_mask=fit,
        fg_mask=fg,
        max_samples=10_000,
        random_state=0,
    )
    assert meta["gmm_fit_foreground_overlap"] == 0
    assert meta["talc_side"] == "high"
    assert meta["background_side"] == "low"
    assert meta["gmm_fit_pixel_count"] < meta["gate_pixel_count"]
    assert talc_band[30, 30]
    assert not np.any(_fit & fg_edge)


def test_gmm_fit_never_includes_foreground_mask():
    h, w = 64, 64
    gate = np.zeros((h, w), dtype=bool)
    gate[10:50, 10:50] = True
    fg = np.zeros((h, w), dtype=bool)
    fg[25:40, 25:40] = True

    fit, meta = _interior_gmm_fit_mask(
        gate,
        fg,
        fg_buffer_radius=3,
        gate_erode_radius=0,
    )
    assert meta["excluded_foreground_pixels"] > 0
    assert meta["gmm_fit_foreground_overlap"] == 0
    assert not np.any(fit & fg)


def test_gradient_refine_stays_inside_dilated_coarse_gate():
    h, w = 128, 128
    rough = np.zeros((h, w), dtype=np.uint8)
    rough[20:60, 20:60] = 1

    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    _textured_patch(rgb, 25, 45, 25, 45)
    _textured_patch(rgb, 5, 15, 5, 15)

    refined, meta = refine_talc_with_image_gradient(
        rough,
        rgb,
        preprocess=False,
        denoise=False,
        illum_sigma=30.0,
        overlap_threshold=0.0,
        close_radius=0,
        random_state=0,
    )

    assert refined[10, 10] == 0
    assert int(refined.sum()) > 0


def test_talc_gmm_threshold_equals_high_component_mean_for_dino():
    activation = np.concatenate(
        [np.full(500, 0.55, dtype=np.float32), np.full(500, 0.90, dtype=np.float32)]
    ).reshape(10, 100)
    fit = np.ones((10, 100), dtype=bool)

    threshold, unbiased, means, _, _ = _talc_two_gmm_threshold(
        activation,
        max_samples=10_000,
        random_state=0,
        pixel_mask=fit,
        use_high_component_mean=True,
    )

    assert threshold == pytest.approx(float(means[1]))
    assert threshold > unbiased


def test_talc_gmm_threshold_bias_raises_cutoff_for_gradient():
    gradient = np.concatenate(
        [np.full(500, 0.05, dtype=np.float32), np.full(500, 0.45, dtype=np.float32)]
    ).reshape(10, 100)
    fit = np.ones((10, 100), dtype=bool)

    thresh_unbiased, unbiased, _, _, _ = _talc_two_gmm_threshold(
        gradient,
        max_samples=10_000,
        random_state=0,
        pixel_mask=fit,
        high_bias=0.0,
        use_high_component_mean=False,
    )
    thresh_biased, _, means, _, _ = _talc_two_gmm_threshold(
        gradient,
        max_samples=10_000,
        random_state=0,
        pixel_mask=fit,
        high_bias=0.5,
        use_high_component_mean=False,
    )

    assert thresh_unbiased == pytest.approx(unbiased)
    assert thresh_biased > unbiased
    assert thresh_biased <= float(means[1])


def test_block01_refine_selects_low_activation_as_talc():
    h, w = 128, 128
    rough = np.zeros((h, w), dtype=np.uint8)
    rough[10:70, 10:70] = 1

    activation = np.full((h, w), 0.85, dtype=np.float32)
    activation[20:50, 20:50] = 0.55
    activation[80:100, 80:100] = 0.52

    gray = np.full((h, w), 180.0, dtype=np.float32)

    refined, meta = refine_talc_with_block01_activation(
        rough,
        activation,
        gray,
        close_radius=0,
        random_state=0,
    )

    assert meta["method"] == "talc_block01_activation_refine"
    assert meta["gmm_threshold_rule"] == "high_component_mean"
    assert meta["selection"] == "low_activation_talc"
    assert refined[30, 30]
    assert meta["pixel_count"] > 0


def test_save_two_gmm_histogram_writes_image(tmp_path):
    from ml.lib.segmentation.gmm_histogram import save_two_gmm_histogram

    gradient = np.concatenate(
        [np.random.normal(0.05, 0.01, 500), np.random.normal(0.4, 0.05, 500)]
    ).astype(np.float32).reshape(10, 100)
    fit = np.ones((10, 100), dtype=bool)
    meta = save_two_gmm_histogram(
        gradient,
        fit,
        means=np.array([0.05, 0.4]),
        variances=np.array([0.001, 0.002]),
        weights=np.array([0.5, 0.5]),
        threshold=0.2,
        dest_path=tmp_path / "gmm.png",
        talc_side="high",
        random_state=0,
    )
    assert (tmp_path / "gmm.png").exists()
    assert meta["threshold"] == 0.2
