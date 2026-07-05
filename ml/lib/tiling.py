"""Patch-aligned image tiling for large inputs (> threshold on both axes)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ml.lib.constants import PATCH_SIZE
from ml.lib.types import DinoBlockOutput, DinoInferenceResult, SegmentationResult

DEFAULT_TILE_THRESHOLD = 2000
DEFAULT_TILE_GRID = 2


@dataclass(frozen=True)
class TileBounds:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def needs_tiling(
    height: int,
    width: int,
    *,
    threshold: int = DEFAULT_TILE_THRESHOLD,
) -> bool:
    """Tile when both dimensions exceed the threshold."""
    return height > threshold and width > threshold


def axis_splits(size: int, grid: int, patch_size: int = PATCH_SIZE) -> list[int]:
    """Return monotonic split positions [0, ..., size] aligned to patch boundaries."""
    if grid <= 1:
        return [0, size]
    splits = [0]
    for i in range(1, grid):
        raw = size * i // grid
        aligned = (raw // patch_size) * patch_size
        min_next = splits[-1] + patch_size
        max_next = size - (grid - i) * patch_size
        aligned = max(min_next, min(aligned, max_next))
        splits.append(aligned)
    splits.append(size)
    return splits


def split_image_grid(
    rgb: np.ndarray,
    grid: int = DEFAULT_TILE_GRID,
    *,
    patch_size: int = PATCH_SIZE,
) -> list[tuple[np.ndarray, TileBounds]]:
    """Split RGB into a grid x grid set of non-overlapping tiles."""
    h, w = rgb.shape[:2]
    ys = axis_splits(h, grid, patch_size)
    xs = axis_splits(w, grid, patch_size)
    tiles: list[tuple[np.ndarray, TileBounds]] = []
    for row in range(grid):
        for col in range(grid):
            bounds = TileBounds(ys[row], ys[row + 1], xs[col], xs[col + 1])
            tiles.append((rgb[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1].copy(), bounds))
    return tiles


def merge_2d_tiles(
    parts: Sequence[tuple[np.ndarray, TileBounds]],
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Stitch equal-resolution 2D arrays (activations, label maps) into one canvas."""
    h, w = full_shape
    out = np.zeros((h, w), dtype=parts[0][0].dtype)
    for arr, bounds in parts:
        th, tw = bounds.height, bounds.width
        if arr.shape != (th, tw):
            raise ValueError(f"Tile shape {arr.shape} != expected {(th, tw)}")
        out[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] = arr
    return out


def merge_feature_tiles(
    parts: Sequence[tuple[torch.Tensor | np.ndarray, TileBounds]],
    full_height: int,
    full_width: int,
    *,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Stitch DINO patch feature maps (C, hp, wp) into a full-image grid."""
    hp = math.ceil(full_height / patch_size)
    wp = math.ceil(full_width / patch_size)
    first = parts[0][0]
    if isinstance(first, torch.Tensor):
        c = int(first.shape[0])
    else:
        c = int(first.shape[0])
    out = np.zeros((c, hp, wp), dtype=np.float32)
    for feats, bounds in parts:
        arr = feats.detach().float().cpu().numpy() if isinstance(feats, torch.Tensor) else feats.astype(np.float32)
        py0 = bounds.y0 // patch_size
        px0 = bounds.x0 // patch_size
        fh, fw = arr.shape[1], arr.shape[2]
        out[:, py0 : py0 + fh, px0 : px0 + fw] = arr
    return torch.from_numpy(out)


def merge_dino_inference_results(
    parts: Sequence[tuple[DinoInferenceResult, TileBounds]],
    full_height: int,
    full_width: int,
) -> DinoInferenceResult:
    """Combine per-tile DINO outputs into one full-image result."""
    if not parts:
        raise ValueError("merge_dino_inference_results requires at least one tile")
    block_indices = parts[0][0].inference_blocks
    merged_blocks: dict[int, DinoBlockOutput] = {}
    for block_idx in block_indices:
        act_parts = [(result.activation(block_idx), bounds) for result, bounds in parts]
        feat_parts = [(result.features(block_idx), bounds) for result, bounds in parts]
        merged_blocks[block_idx] = DinoBlockOutput(
            block_index=block_idx,
            features=merge_feature_tiles(feat_parts, full_height, full_width),
            activation=merge_2d_tiles(act_parts, (full_height, full_width)),
        )
    meta = dict(parts[0][0].meta)
    meta["tiled"] = True
    meta["tile_grid"] = int(round(len(parts) ** 0.5))
    meta["tile_count"] = len(parts)
    return DinoInferenceResult(
        blocks=merged_blocks,
        native_width=full_width,
        native_height=full_height,
        inference_blocks=list(block_indices),
        meta=meta,
    )


def merge_segmentation_results(
    parts: Sequence[tuple[SegmentationResult, TileBounds]],
    full_height: int,
    full_width: int,
    *,
    grid: int = DEFAULT_TILE_GRID,
) -> SegmentationResult:
    """Combine per-tile segmentation labels and metadata."""
    labels = merge_2d_tiles([(r.labels, b) for r, b in parts], (full_height, full_width))
    tile_meta = []
    class_totals: dict[int, int] = {}
    for result, bounds in parts:
        counts = result.metadata.get("final_class_counts", {})
        tile_meta.append(
            {
                "bounds": [bounds.y0, bounds.y1, bounds.x0, bounds.x1],
                "final_class_counts": counts,
            }
        )
        for key, value in counts.items():
            cls_id = int(key)
            class_totals[cls_id] = class_totals.get(cls_id, 0) + int(value)

    metadata = dict(parts[0][0].metadata)
    metadata["tiled"] = True
    metadata["tile_grid"] = grid
    metadata["tile_count"] = len(parts)
    metadata["tiles"] = tile_meta
    metadata["final_class_counts"] = class_totals
    return SegmentationResult(
        labels=labels.astype(np.uint8),
        native_width=full_width,
        native_height=full_height,
        mask_width=full_width,
        mask_height=full_height,
        mask_to_native_scale=1.0,
        metadata=metadata,
    )
