"""Region banding, promotion, and final label map assembly."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import closing, disk

from ml.lib.constants import (
    CLS_BACKGROUND,
    CLS_DEFECT,
    CLS_FOREGROUND,
    CLS_PARTITIONS,
    CLASS_NAMES,
    N_GAUSSIANS_REGIONS,
    REGION_MAP_INTENSITY_GRADIENT,
    REGION_NAMES,
)


def segment_by_thresholds(value_map: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    labels = np.zeros(value_map.shape, dtype=np.int32)
    for i, t in enumerate(thresholds):
        labels[value_map >= t] = i + 1
    return labels


def morphological_close_label_map(
    labels: np.ndarray,
    radius: int,
    class_ids: list[int] | None = None,
) -> np.ndarray:
    if radius <= 0:
        return labels.astype(np.int32, copy=True)

    labels = labels.astype(np.int32, copy=False)
    ids = class_ids if class_ids is not None else [int(c) for c in np.unique(labels)]
    footprint = disk(int(radius))
    radius = int(radius)

    steal_dist = np.full((len(ids),) + labels.shape, np.inf, dtype=np.float64)
    for i, cls_id in enumerate(ids):
        mask = labels == cls_id
        if not np.any(mask):
            continue
        closed = closing(mask.astype(np.uint8), footprint=footprint).astype(bool)
        dist_to_c = ndi.distance_transform_edt(~mask)
        steal = closed & (labels != cls_id) & (dist_to_c <= radius)
        steal_dist[i] = np.where(steal, dist_to_c, np.inf)

    out = labels.copy()
    any_steal = np.isfinite(steal_dist).any(axis=0)
    if np.any(any_steal):
        best = np.argmin(steal_dist, axis=0)
        for i, cls_id in enumerate(ids):
            out[any_steal & (best == i)] = cls_id
    return out


def promote_regions_by_overlap(
    region_labels: np.ndarray,
    seed_mask: np.ndarray,
    overlap_threshold: float,
    n_regions: int = N_GAUSSIANS_REGIONS,
) -> np.ndarray:
    final = (seed_mask > 0).astype(np.uint8)
    seed_bool = final.astype(bool)
    seed_weights = seed_bool.astype(np.float64).ravel()

    for region_id in range(n_regions):
        band = region_labels == region_id
        if not np.any(band):
            continue
        labeled, n_comp = ndi.label(band)
        if n_comp == 0:
            continue
        flat = labeled.ravel()
        sizes = np.bincount(flat)
        overlap_counts = np.bincount(
            flat, weights=seed_weights, minlength=sizes.shape[0]
        )
        selected_ids = [
            comp_id
            for comp_id in range(1, n_comp + 1)
            if sizes[comp_id] > 0
            and (overlap_counts[comp_id] / float(sizes[comp_id])) >= overlap_threshold
        ]
        if not selected_ids:
            continue
        keep = np.isin(labeled, np.asarray(selected_ids, dtype=labeled.dtype))
        final[keep] = 1
    return final


def select_regions_by_overlap(
    region_labels: np.ndarray,
    reference_mask: np.ndarray,
    overlap_threshold: float,
    n_regions: int = N_GAUSSIANS_REGIONS,
) -> np.ndarray:
    final = np.zeros(region_labels.shape, dtype=np.uint8)
    ref_bool = reference_mask.astype(bool)
    ref_weights = ref_bool.astype(np.float64).ravel()

    for region_id in range(n_regions):
        band = region_labels == region_id
        if not np.any(band):
            continue
        labeled, n_comp = ndi.label(band)
        if n_comp == 0:
            continue
        flat = labeled.ravel()
        sizes = np.bincount(flat)
        overlap_counts = np.bincount(
            flat, weights=ref_weights, minlength=sizes.shape[0]
        )
        selected_ids = [
            comp_id
            for comp_id in range(1, n_comp + 1)
            if sizes[comp_id] > 0
            and (overlap_counts[comp_id] / float(sizes[comp_id])) >= overlap_threshold
        ]
        if not selected_ids:
            continue
        keep = np.isin(labeled, np.asarray(selected_ids, dtype=labeled.dtype))
        final[keep] = 1
    return final


def split_foreground_object_and_partitions(
    foreground_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fg = foreground_mask.astype(bool)
    if not np.any(fg):
        z = np.zeros(fg.shape, dtype=np.uint8)
        return z, z

    labeled, n_comp = ndi.label(fg)
    if n_comp == 0:
        z = np.zeros(fg.shape, dtype=np.uint8)
        return z, z

    sizes = ndi.sum(fg, labeled, index=np.arange(1, n_comp + 1))
    largest_id = int(np.argmax(sizes)) + 1
    fg_object = (labeled == largest_id).astype(np.uint8)
    partitions = (fg & (labeled != largest_id)).astype(np.uint8)
    return fg_object, partitions


def build_final_segmentation(
    fg_object: np.ndarray,
    partitions: np.ndarray,
    defect: np.ndarray,
) -> np.ndarray:
    seg = np.full(fg_object.shape, CLS_BACKGROUND, dtype=np.uint8)
    seg[defect.astype(bool)] = CLS_DEFECT
    seg[partitions.astype(bool)] = CLS_PARTITIONS
    seg[fg_object.astype(bool)] = CLS_FOREGROUND
    return seg


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[i]: int((labels == i).sum()) for i in CLASS_NAMES}


def region_counts(labels: np.ndarray) -> dict[str, int]:
    return {REGION_NAMES[i]: int((labels == i).sum()) for i in REGION_NAMES}


def defect_mask_from_regions(
    region_labels: np.ndarray,
    defect_seed: np.ndarray,
    *,
    region_map: str,
    region_overlap: float,
) -> tuple[np.ndarray, str]:
    if region_map == REGION_MAP_INTENSITY_GRADIENT:
        mask = select_regions_by_overlap(
            region_labels, defect_seed, region_overlap
        )
        return mask, "gradient_region_select"
    mask = promote_regions_by_overlap(region_labels, defect_seed, region_overlap)
    return mask, "seed_plus_region_promote"
