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
    CLS_TALC,
    DEFAULT_MIN_BACKPROJ_SCORE,
    DEFAULT_MIN_COSINE_SIM,
    PATCH_SIZE,
    SEGMENTATION_MODE_EMBEDDING,
    SEGMENTATION_MODE_HYBRID,
    SEGMENTATION_MODE_INTENSITY,
)
from ml.lib.dino.inference import upsample_patch_map
from ml.lib.segmentation.intensity_fg import segment_coarse_fine_intensity
from ml.lib.segmentation.talc_intensity import segment_talc_intensity_gmm
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


def segment_talc_embedding(
    rgb: np.ndarray,
    block01_features: torch.Tensor | np.ndarray,
    calib: CalibrationData,
    *,
    min_cosine: float = DEFAULT_MIN_COSINE_SIM,
    exclude_mask: np.ndarray | None = None,
    close_radius: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Talc patches via nearest-neighbor: talc vs calibrated background embedding."""
    h, w = rgb.shape[:2]
    talc_stats = calib.stats.get("talc")
    if (
        talc_stats is None
        or (talc_stats.count or 0) == 0
        or talc_stats.mean_embedding is None
    ):
        return np.zeros((h, w), dtype=np.uint8), {
            "method": "talc_embedding",
            "warning": "no talc calibration",
        }

    unit, _c, hp, wp = unit_patch_vectors(block01_features)
    mean_talc = talc_stats.mean_embedding.astype(np.float32)
    sim_talc = (unit @ mean_talc).reshape(hp, wp)

    mean_bg = calib.background_mean_embedding()
    if mean_bg is not None:
        sim_bg = (unit @ mean_bg.astype(np.float32)).reshape(hp, wp)
        patch_mask = sim_talc > sim_bg
        classifier = "talc_vs_background_nn"
    else:
        patch_mask = sim_talc >= min_cosine
        classifier = "talc_threshold_fallback"

    talc_up = upsample_patch_map(patch_mask.astype(np.float32), (h, w)) >= 0.5
    talc_mask = talc_up.astype(np.uint8)

    if exclude_mask is not None:
        talc_mask[exclude_mask.astype(bool)] = 0

    if close_radius > 0 and np.any(talc_mask):
        label_map = np.zeros((h, w), dtype=np.int32)
        label_map[talc_mask.astype(bool)] = CLS_TALC
        label_map = morphological_close_label_map(
            label_map,
            radius=close_radius,
            class_ids=[CLS_BACKGROUND, CLS_TALC],
        )
        talc_mask = (label_map == CLS_TALC).astype(np.uint8)

    meta = {
        "method": "talc_embedding",
        "classifier": classifier,
        "min_cosine": float(min_cosine),
        "patch_grid": [hp, wp],
        "pixel_count": int(talc_mask.sum()),
        "background_calibrated": mean_bg is not None,
    }
    return talc_mask, meta


def segment_hybrid(
    rgb: np.ndarray,
    block01_activation: np.ndarray,
    block01_features: torch.Tensor | np.ndarray,
    calib: CalibrationData,
    config: SegmentConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coarse/fine via intensity GMM; talc via 2-GMM on intensity outside dilated FG."""
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
    talc_mask, talc_meta = segment_talc_intensity_gmm(
        rgb,
        fg_mask.astype(bool),
        fg_dilate_radius=config.fg_dilate_radius,
        talc_black_max=config.talc_black_max,
        preprocess=config.preprocess,
        denoise=config.denoise,
        illum_sigma=config.illum_sigma,
        max_samples=config.max_samples,
        random_state=config.random_state,
        close_radius=config.close_radius,
    )
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
            raise ValueError("block01_features required for embedding segmentation mode")
        labels, seg_meta = segment_embedding_cosine(
            rgb,
            block_features,
            calib,
            min_cosine=config.min_cosine_sim,
            close_radius=close_radius,
        )
    elif mode == SEGMENTATION_MODE_HYBRID:
        if block01_activation is None or block_features is None:
            raise ValueError(
                "block01_activation and block01_features required for hybrid segmentation mode"
            )
        labels, seg_meta = segment_hybrid(
            rgb,
            block01_activation,
            block_features,
            calib,
            config,
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
