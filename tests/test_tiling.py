"""Tests for large-image 2x2 tiling utilities."""

from __future__ import annotations

import numpy as np
import torch

from ml.lib.tiling import (
    DEFAULT_TILE_GRID,
    DEFAULT_TILE_THRESHOLD,
    TileBounds,
    axis_splits,
    merge_2d_tiles,
    merge_dino_inference_results,
    merge_feature_tiles,
    merge_segmentation_results,
    needs_tiling,
    split_image_grid,
)
from ml.lib.types import DinoBlockOutput, DinoInferenceResult, SegmentationResult


def test_needs_tiling_both_axes():
    assert not needs_tiling(2000, 2000, threshold=DEFAULT_TILE_THRESHOLD)
    assert not needs_tiling(2000, 3000, threshold=DEFAULT_TILE_THRESHOLD)
    assert not needs_tiling(3000, 2000, threshold=DEFAULT_TILE_THRESHOLD)
    assert needs_tiling(2001, 2001, threshold=DEFAULT_TILE_THRESHOLD)


def test_split_covers_full_image_without_overlap():
    rgb = np.arange(2100 * 2200 * 3, dtype=np.uint8).reshape(2100, 2200, 3)
    tiles = split_image_grid(rgb, DEFAULT_TILE_GRID)
    assert len(tiles) == DEFAULT_TILE_GRID * DEFAULT_TILE_GRID
    canvas = np.zeros(rgb.shape[:2], dtype=np.int32)
    for tile, bounds in tiles:
        assert tile.shape == (bounds.height, bounds.width, 3)
        canvas[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] += 1
    assert np.all(canvas == 1)


def test_axis_splits_are_monotonic_and_patch_aligned():
    splits = axis_splits(2500, 2)
    assert splits[0] == 0
    assert splits[-1] == 2500
    assert splits[1] % 14 == 0


def test_merge_2d_tiles_roundtrip():
    parts = [
        (np.full((50, 60), 1, dtype=np.uint8), TileBounds(0, 50, 0, 60)),
        (np.full((50, 60), 2, dtype=np.uint8), TileBounds(0, 50, 60, 120)),
        (np.full((50, 60), 3, dtype=np.uint8), TileBounds(50, 100, 0, 60)),
        (np.full((50, 60), 4, dtype=np.uint8), TileBounds(50, 100, 60, 120)),
    ]
    merged = merge_2d_tiles(parts, (100, 120))
    expected = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.repeat(np.repeat(expected, 50, axis=0), 60, axis=1)
    np.testing.assert_array_equal(merged, expected)


def test_merge_feature_tiles_stitches_patch_grid():
    bounds_tl = TileBounds(0, 28, 0, 28)
    bounds_br = TileBounds(28, 56, 28, 56)
    feat_tl = torch.ones(4, 2, 2)
    feat_br = torch.full((4, 2, 2), 2.0)
    merged = merge_feature_tiles(
        [(feat_tl, bounds_tl), (feat_br, bounds_br)],
        full_height=56,
        full_width=56,
    )
    assert merged.shape == (4, 4, 4)
    assert merged[:, 0, 0].eq(1.0).all()
    assert merged[:, 2, 2].eq(2.0).all()


def test_merge_dino_and_segmentation_results():
    h, w = 40, 40
    block = DinoBlockOutput(
        block_index=1,
        features=torch.zeros(8, 3, 3),
        activation=np.zeros((20, 20), dtype=np.float32),
    )
    dino = DinoInferenceResult(
        blocks={1: block},
        native_width=20,
        native_height=20,
        inference_blocks=[1],
        meta={},
    )
    seg = SegmentationResult(
        labels=np.full((20, 20), 3, dtype=np.uint8),
        native_width=20,
        native_height=20,
        mask_width=20,
        mask_height=20,
        metadata={"final_class_counts": {3: 400}},
    )
    bounds = TileBounds(0, 20, 0, 20)
    merged_dino = merge_dino_inference_results([(dino, bounds)], h, w)
    assert merged_dino.native_height == h
    assert merged_dino.activation(1).shape == (h, w)

    merged_seg = merge_segmentation_results([(seg, bounds)], h, w)
    assert merged_seg.labels.shape == (h, w)
    assert merged_seg.metadata["tiled"] is True
