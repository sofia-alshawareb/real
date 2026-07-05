"""Calibrated segmentation: RGB backprojection and embedding cosine matching."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ml.lib.calibration.patch_utils import unit_patch_vectors
from ml.lib.calibration.histograms import backproject_rgb_histogram
from ml.lib.calibration.types import CalibrationData
from ml.lib.constants import (
    CALIB_CLASS_ID_BY_KEY,
    CALIB_CLASS_KEYS,
    CLASS_NAMES,
    CLASS_TIE_PRIORITY,
    CLS_BACKGROUND,
    DEFAULT_EMBEDDING_BLOCK,
    DEFAULT_MIN_BACKPROJ_SCORE,
    DEFAULT_MIN_COSINE_SIM,
    SEGMENTATION_MODE_EMBEDDING,
    SEGMENTATION_MODE_HYBRID,
    SEGMENTATION_MODE_INTENSITY,
)
from ml.lib.dino.inference import upsample_patch_map
from ml.lib.calibration.filters import rgb_to_gray
from ml.lib.calibration.talc_threshold import resolve_talc_intensity_max
from ml.lib.segmentation.intensity_fg import segment_coarse_fine_intensity
from ml.lib.constants import TALC_REFINE_MODE_DINO, TALC_REFINE_MODE_GRADIENT
from ml.lib.segmentation.talc_intensity import (
    refine_talc_with_block01_activation,
    refine_talc_with_image_gradient,
    segment_talc_hybrid,
)
from ml.lib.segmentation.regions import (
    build_final_segmentation,
    class_counts,
    morphological_close_label_map,
)
from ml.lib.types import SegmentConfig, SegmentationResult


def _priority_class_ids(active_keys: list[str]) -> list[int]:
    active_ids = {CALIB_CLASS_ID_BY_KEY[k] for k in active_keys}
    return [c for c in CLASS_TIE_PRIORITY if c in active_ids]


def segment_rgb_backprojection(
    rgb: np.ndarray,
    calib: CalibrationData,
    *,
    min_score: float = DEFAULT_MIN_BACKPROJ_SCORE,
    close_radius: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = rgb.shape[:2]
    active_keys = calib.active_class_keys()
    if not active_keys:
        labels = np.zeros((h, w), dtype=np.uint8)
        return labels, {"method": "rgb_backprojection", "active_classes": [], "warning": "empty calibration"}

    priority_ids = _priority_class_ids(active_keys)
    score_stack = []
    for cls_id in priority_ids:
        key = next(k for k, cid in CALIB_CLASS_ID_BY_KEY.items() if cid == cls_id)
        hist = calib.rgb_histograms.get(key)
        if hist is None or hist.sum() == 0:
            score_stack.append(np.zeros((h, w), dtype=np.float32))
        else:
            score_stack.append(backproject_rgb_histogram(rgb, hist))
    scores = np.stack(score_stack, axis=-1)
    best_idx = np.argmax(scores, axis=-1)
    max_score = scores.max(axis=-1)
    labels = np.zeros((h, w), dtype=np.uint8)
    for i, cls_id in enumerate(priority_ids):
        labels[best_idx == i] = cls_id
    labels[max_score < min_score] = CLS_BACKGROUND

    if close_radius > 0:
        labels = morphological_close_label_map(
            labels,
            radius=close_radius,
            class_ids=[CLS_BACKGROUND, *priority_ids],
        ).astype(np.uint8)

    meta = {
        "method": "rgb_backprojection",
        "active_classes": active_keys,
        "min_score": float(min_score),
        "final_class_counts": class_counts(labels),
    }
    return labels.astype(np.uint8), meta


def segment_embedding_cosine(
    rgb: np.ndarray,
    block_features: torch.Tensor | np.ndarray,
    calib: CalibrationData,
    *,
    min_cosine: float = DEFAULT_MIN_COSINE_SIM,
    close_radius: int = 0,
    class_keys: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = rgb.shape[:2]
    if class_keys is None:
        active_keys = [
            k
            for k in CALIB_CLASS_KEYS
            if calib.stats.get(k) and (calib.stats[k].count or 0) > 0
        ]
        active_keys = [k for k in active_keys if calib.stats[k].mean_embedding is not None]
    else:
        active_keys = [
            k
            for k in class_keys
            if calib.stats.get(k)
            and (calib.stats[k].count or 0) > 0
            and calib.stats[k].mean_embedding is not None
        ]
    if not active_keys:
        labels = np.zeros((h, w), dtype=np.uint8)
        return labels, {"method": "embedding_cosine", "active_classes": [], "warning": "empty calibration"}

    priority_ids = _priority_class_ids(active_keys)
    unit, _c, hp, wp = unit_patch_vectors(block_features)
    score_stack = []
    for cls_id in priority_ids:
        key = next(k for k, cid in CALIB_CLASS_ID_BY_KEY.items() if cid == cls_id)
        mean_vec = calib.stats[key].mean_embedding
        assert mean_vec is not None
        sim = (unit @ mean_vec.astype(np.float32)).reshape(hp, wp)
        score_stack.append(sim.astype(np.float32))
    scores = np.stack(score_stack, axis=-1)
    best_idx = np.argmax(scores, axis=-1)
    max_score = scores.max(axis=-1)
    patch_labels = np.zeros((hp, wp), dtype=np.uint8)
    for i, cls_id in enumerate(priority_ids):
        patch_labels[best_idx == i] = cls_id
    patch_labels[max_score < min_cosine] = CLS_BACKGROUND

    labels_up = upsample_patch_map(patch_labels.astype(np.float32), (h, w))
    labels = np.rint(labels_up).astype(np.uint8)

    if close_radius > 0:
        labels = morphological_close_label_map(
            labels,
            radius=close_radius,
            class_ids=[CLS_BACKGROUND, *priority_ids],
        ).astype(np.uint8)

    meta = {
        "method": "embedding_cosine",
        "active_classes": active_keys,
        "min_cosine": float(min_cosine),
        "patch_grid": [hp, wp],
        "final_class_counts": class_counts(labels),
    }
    return labels.astype(np.uint8), meta


def segment_hybrid(
    rgb: np.ndarray,
    block01_activation: np.ndarray,
    calib: CalibrationData,
    config: SegmentConfig,
    *,
    block11_features: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coarse/fine via block-1 activation; talc via dark coarse gate + DINO or gradient refine."""
    del block11_features  # hybrid talc no longer uses block-11 embedding similarity
    fg_object, partitions, fg_mask, fg_meta = segment_coarse_fine_intensity(
        rgb,
        block01_activation,
        preprocess=config.preprocess,
        denoise=config.denoise,
        illum_sigma=config.illum_sigma,
        max_samples=config.max_samples,
        random_state=config.random_state,
        region_overlap=config.region_overlap,
        close_radius=config.close_radius,
    )
    talc_mask, talc_meta = segment_talc_hybrid(
        rgb,
        fg_mask.astype(bool),
        fg_dilate_radius=config.fg_dilate_radius,
        talc_black_max=resolve_talc_intensity_max(calib, config.talc_black_max),
        close_radius=config.close_radius,
    )
    refine_mode = config.talc_refine_mode
    if refine_mode == TALC_REFINE_MODE_DINO:
        talc_mask, refine_meta = refine_talc_with_block01_activation(
            talc_mask,
            block01_activation,
            rgb_to_gray(rgb),
            overlap_threshold=config.talc_block01_overlap,
            max_samples=config.max_samples,
            random_state=config.random_state,
            close_radius=config.close_radius,
            gate_dilate_radius=config.talc_contour_dilate,
            fg_mask=fg_mask.astype(bool),
            fg_dilate_radius=config.talc_refine_fg_dilate_radius,
            gmm_fg_buffer_radius=config.talc_gmm_fg_buffer_radius,
            gmm_gate_erode_radius=config.talc_gmm_gate_erode,
        )
        talc_meta["refine_mode"] = refine_mode
        talc_meta["block01_refine"] = refine_meta
    elif refine_mode == TALC_REFINE_MODE_GRADIENT:
        talc_mask, refine_meta = refine_talc_with_image_gradient(
            talc_mask,
            rgb,
            preprocess=config.preprocess,
            denoise=config.denoise,
            illum_sigma=config.illum_sigma,
            overlap_threshold=config.talc_block01_overlap,
            max_samples=config.max_samples,
            random_state=config.random_state,
            close_radius=config.close_radius,
            gate_dilate_radius=config.talc_contour_dilate,
            fg_mask=fg_mask.astype(bool),
            fg_dilate_radius=config.talc_refine_fg_dilate_radius,
            gmm_fg_buffer_radius=config.talc_gmm_fg_buffer_radius,
            gmm_gate_erode_radius=config.talc_gmm_gate_erode,
            gmm_threshold_high_bias=config.talc_gmm_threshold_high_bias,
        )
        talc_meta["refine_mode"] = refine_mode
        talc_meta["gradient_refine"] = refine_meta
    else:
        raise ValueError(
            f"Unknown talc_refine_mode {refine_mode!r}; "
            f"expected {TALC_REFINE_MODE_DINO!r} or {TALC_REFINE_MODE_GRADIENT!r}"
        )
    talc_meta["pixel_count"] = int(talc_mask.sum())
    labels = build_final_segmentation(fg_object, partitions, talc_mask)
    if config.close_radius > 0:
        labels = morphological_close_label_map(
            labels,
            radius=config.close_radius,
            class_ids=list(CLASS_NAMES.keys()),
        ).astype(np.uint8)

    meta = {
        "method": "hybrid",
        "foreground": fg_meta,
        "talc": talc_meta,
        "final_class_counts": class_counts(labels),
    }
    return labels.astype(np.uint8), meta


def segment_calibrated(
    rgb: np.ndarray,
    calib: CalibrationData,
    config: SegmentConfig,
    *,
    block01_activation: np.ndarray | None = None,
    block01_features: torch.Tensor | np.ndarray | None = None,
    block11_features: torch.Tensor | np.ndarray | None = None,
) -> SegmentationResult:
    t0 = time.perf_counter()
    mode = config.segmentation_mode
    close_radius = config.close_radius
    if config.block_index == DEFAULT_EMBEDDING_BLOCK:
        block_features = block11_features if block11_features is not None else block01_features
    else:
        block_features = block01_features if block01_features is not None else block11_features

    if mode == SEGMENTATION_MODE_INTENSITY:
        labels, seg_meta = segment_rgb_backprojection(
            rgb,
            calib,
            min_score=config.min_backproj_score,
            close_radius=close_radius,
        )
    elif mode == SEGMENTATION_MODE_EMBEDDING:
        if block_features is None:
            raise ValueError("block11_features required for embedding segmentation mode")
        labels, seg_meta = segment_embedding_cosine(
            rgb,
            block_features,
            calib,
            min_cosine=config.min_cosine_sim,
            close_radius=close_radius,
        )
    elif mode == SEGMENTATION_MODE_HYBRID:
        if block01_activation is None:
            raise ValueError("block01_activation required for hybrid segmentation mode")
        labels, seg_meta = segment_hybrid(
            rgb,
            block01_activation,
            calib,
            config,
            block11_features=block11_features,
        )
    else:
        raise ValueError(f"Unknown segmentation mode: {mode!r}")

    elapsed = time.perf_counter() - t0
    h, w = rgb.shape[:2]
    metadata = {
        "segmentation_mode": mode,
        "calibrated": seg_meta,
        "elapsed_s": elapsed,
        "final_class_counts": class_counts(labels),
    }
    return SegmentationResult(
        labels=labels.astype(np.uint8),
        native_width=w,
        native_height=h,
        mask_width=w,
        mask_height=h,
        mask_to_native_scale=1.0,
        metadata=metadata,
    )
