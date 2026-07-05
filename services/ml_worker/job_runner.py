"""Job orchestration: DINO + calibrated segmentation + artifact persistence."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from queue import Empty, Queue

import numpy as np
import torch

from ml.lib.calibration.store import CalibrationStore
from ml.lib.constants import COARSE_FINE_DINO_BLOCK, TALC_EMBEDDING_BLOCK, SEGMENTATION_MODE_EMBEDDING, SEGMENTATION_MODE_HYBRID
from ml.lib.tiling import (
    TileBounds,
    merge_dino_inference_results,
    merge_segmentation_results,
    needs_tiling,
    split_image_grid,
)
from ml.lib.types import DinoArtifacts, DinoInferenceResult, SegmentationResult

from services.ml_worker.config import ServiceConfig, load_config
from services.ml_worker.dino_service import DinoInferenceService
from services.ml_worker.refinement_service import CalibrationRefinementService
from services.ml_worker.segmentation_service import SegmentationService
from services.ml_worker.storage.filesystem import FilesystemArtifactStore


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    image_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error: str | None = None
    save_activations: bool = False


@dataclass
class WorkerState:
    config: ServiceConfig
    store: FilesystemArtifactStore
    calib_store: CalibrationStore
    dino: DinoInferenceService
    segmentation: SegmentationService
    refinement: CalibrationRefinementService
    jobs: dict[str, JobRecord] = field(default_factory=dict)
    _queue: Queue = field(default_factory=Queue)
    _thread: threading.Thread | None = None


def create_worker(config_path: str | None = None) -> WorkerState:
    import os

    config = load_config(config_path)
    env_save = os.environ.get("ML_SAVE_ACTIVATIONS")
    if env_save is not None:
        config.dino.save_activations = env_save.lower() in ("1", "true", "yes")
    store = FilesystemArtifactStore(
        config.storage.root,
        images_subdir=config.storage.images_subdir,
    )
    calib_store = CalibrationStore(
        config.segmentation.calibration_dir,
        rgb_hist_bins=config.segmentation.rgb_hist_bins,
    )
    if not calib_store.exists():
        raise FileNotFoundError(
            f"Calibration not found at {config.segmentation.calibration_dir}. "
            "Run: python ml/prepare_calibration.py"
        )
    calib_store.load()
    dino = DinoInferenceService(config.dino)
    dino.warmup()
    segmentation = SegmentationService(config.segmentation)
    refinement = CalibrationRefinementService(config.segmentation, calib_store, segmentation)
    state = WorkerState(
        config=config,
        store=store,
        calib_store=calib_store,
        dino=dino,
        segmentation=segmentation,
        refinement=refinement,
    )
    state._thread = threading.Thread(target=_consumer_loop, args=(state,), daemon=True)
    state._thread.start()
    return state


def enqueue_segment_job(
    state: WorkerState,
    image_id: str,
    *,
    save_activations: bool = False,
) -> JobRecord:
    if not state.store.image_exists(image_id):
        raise FileNotFoundError(f"Unknown image_id: {image_id}")
    job = JobRecord(
        job_id=str(uuid.uuid4()),
        image_id=image_id,
        save_activations=save_activations,
    )
    state.jobs[job.job_id] = job
    state._queue.put(job.job_id)
    return job


def get_job(state: WorkerState, job_id: str) -> JobRecord:
    if job_id not in state.jobs:
        raise KeyError(job_id)
    return state.jobs[job_id]


def _ensure_dino_blocks(
    state: WorkerState, image_id: str, rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or compute block-1 activation + block-1/11 features for hybrid pipeline."""
    dino_dir = state.store._dino_dir(image_id)
    if state.store.dino_features_ready(image_id):
        return (
            state.store.load_block01_activation(image_id),
            state.store.load_block01_features(image_id),
            state.store.load_block11_features(image_id),
        )
    dino_result, _ = _run_dino(state, rgb, save_activations=False)
    block01_act = dino_result.activation(COARSE_FINE_DINO_BLOCK)
    block01 = dino_result.features(COARSE_FINE_DINO_BLOCK).numpy()
    block11 = dino_result.features(TALC_EMBEDDING_BLOCK).numpy()
    meta = {"inference_blocks": dino_result.inference_blocks, **dino_result.meta}
    state.store.write_block01_features(
        dino_dir,
        block01,
        activation=block01_act,
        meta=meta,
    )
    state.store.write_block11_features(dino_dir, block11, meta=meta)
    return block01_act, block01, block11


def _run_dino(
    state: WorkerState,
    rgb: np.ndarray,
    *,
    save_activations: bool,
) -> tuple[DinoInferenceResult, DinoArtifacts | None]:
    """Run DINO on full image or 2x2 tiles when both dimensions exceed threshold."""
    seg_cfg = state.config.segmentation
    h, w = rgb.shape[:2]
    if not needs_tiling(h, w, threshold=seg_cfg.tile_threshold):
        return state.dino.run(rgb, save_activations=save_activations)

    tile_parts: list[tuple[DinoInferenceResult, TileBounds]] = []
    tile_artifacts: list[DinoArtifacts | None] = []
    for tile_rgb, bounds in split_image_grid(rgb, seg_cfg.tile_grid):
        result, artifacts = state.dino.run(tile_rgb, save_activations=save_activations)
        tile_parts.append((result, bounds))
        tile_artifacts.append(artifacts)

    merged = merge_dino_inference_results(tile_parts, h, w)
    merged_artifacts = None
    if save_activations and all(a is not None for a in tile_artifacts):
        merged_artifacts = DinoArtifacts.from_inference(merged)
    return merged, merged_artifacts


def _run_segmentation_pipeline(
    state: WorkerState,
    rgb: np.ndarray,
    calib,
    *,
    save_activations: bool,
) -> tuple[SegmentationResult, DinoInferenceResult | None, DinoArtifacts | None]:
    """Run segmentation once, or on a 2x2 tile grid for large images."""
    seg_cfg = state.config.segmentation
    h, w = rgb.shape[:2]
    mode = seg_cfg.mode
    needs_dino = mode in (SEGMENTATION_MODE_HYBRID, SEGMENTATION_MODE_EMBEDDING)

    if not needs_tiling(h, w, threshold=seg_cfg.tile_threshold):
        dino_result = None
        dino_artifacts = None
        block01_activation = None
        block01_features = None
        block11_features = None
        if needs_dino:
            dino_result, dino_artifacts = state.dino.run(
                rgb, save_activations=save_activations
            )
            block01_activation = dino_result.activation(COARSE_FINE_DINO_BLOCK)
            block01_features = dino_result.features(COARSE_FINE_DINO_BLOCK)
            block11_features = dino_result.features(TALC_EMBEDDING_BLOCK)
        seg_result = state.segmentation.run(
            rgb,
            calib,
            block01_activation=block01_activation,
            block01_features=block01_features,
            block11_features=block11_features,
        )
        return seg_result, dino_result, dino_artifacts

    tile_seg: list[tuple[SegmentationResult, TileBounds]] = []
    tile_dino: list[tuple[DinoInferenceResult, TileBounds]] = []
    for tile_rgb, bounds in split_image_grid(rgb, seg_cfg.tile_grid):
        block01_activation = None
        block01_features = None
        block11_features = None
        dino_result = None
        if needs_dino:
            dino_result, _ = state.dino.run(tile_rgb, save_activations=False)
            tile_dino.append((dino_result, bounds))
            block01_activation = dino_result.activation(COARSE_FINE_DINO_BLOCK)
            block01_features = dino_result.features(COARSE_FINE_DINO_BLOCK)
            block11_features = dino_result.features(TALC_EMBEDDING_BLOCK)
        tile_seg.append(
            (
                state.segmentation.run(
                    tile_rgb,
                    calib,
                    block01_activation=block01_activation,
                    block01_features=block01_features,
                    block11_features=block11_features,
                ),
                bounds,
            )
        )

    seg_result = merge_segmentation_results(tile_seg, h, w, grid=seg_cfg.tile_grid)
    merged_dino = merge_dino_inference_results(tile_dino, h, w) if tile_dino else None
    merged_artifacts = (
        DinoArtifacts.from_inference(merged_dino) if save_activations and merged_dino else None
    )
    return seg_result, merged_dino, merged_artifacts


def refine_calibration_from_hint(
    state: WorkerState,
    image_id: str,
    hint_mask: np.ndarray,
    ui_class: str,
) -> SegmentationResult:
    if not state.store.image_exists(image_id):
        raise FileNotFoundError(f"Unknown image_id: {image_id}")

    rgb = state.store.load_rgb(image_id)
    block01_act, _block01, block11 = _ensure_dino_blocks(state, image_id, rgb)

    result = state.refinement.refine(
        rgb, block11, hint_mask, ui_class, block01_activation=block01_act
    )

    prev_summary: dict = {}
    if state.store.segmentation_ready(image_id):
        try:
            prev_summary = state.store.get_mask_metadata(image_id)
        except FileNotFoundError:
            pass
    history = list(prev_summary.get("refinement_history", []))
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ui_class": ui_class,
            **result.metadata.get("refinement", {}),
        }
    )
    result.metadata["refinement_history"] = history

    state.store.save_segmentation(image_id, result)
    return result


def refine_defect_from_hint(
    state: WorkerState,
    image_id: str,
    hint_mask: np.ndarray,
) -> SegmentationResult:
    """Backward-compatible alias: talc hint refinement."""
    return refine_calibration_from_hint(state, image_id, hint_mask, "talc")


def _consumer_loop(state: WorkerState) -> None:
    while True:
        try:
            job_id = state._queue.get(timeout=0.5)
        except Empty:
            continue
        job = state.jobs[job_id]
        job.status = JobStatus.RUNNING
        job.progress = 0.1
        try:
            _run_segment_job(state, job)
            job.status = JobStatus.DONE
            job.progress = 1.0
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            state.store.cleanup_temp(job.image_id)
        finally:
            state._queue.task_done()


def _run_segment_job(state: WorkerState, job: JobRecord) -> None:
    image_id = job.image_id
    rgb = state.store.load_rgb(image_id)
    temp_dir = state.store.begin_temp(image_id)
    job.progress = 0.2
    calib = state.calib_store.get()
    mode = state.config.segmentation.mode
    needs_dino = mode in (SEGMENTATION_MODE_HYBRID, SEGMENTATION_MODE_EMBEDDING)

    try:
        seg_result, dino_result, dino_artifacts = _run_segmentation_pipeline(
            state,
            rgb,
            calib,
            save_activations=job.save_activations,
        )
        job.progress = 0.5

        if needs_dino and dino_result is not None:
            dino_temp = temp_dir / "dino"
            meta = {"inference_blocks": dino_result.inference_blocks, **dino_result.meta}
            state.store.write_block01_features(
                dino_temp,
                dino_result.features(COARSE_FINE_DINO_BLOCK).numpy(),
                activation=dino_result.activation(COARSE_FINE_DINO_BLOCK),
                meta=meta,
            )
            state.store.write_block11_features(
                dino_temp,
                dino_result.features(TALC_EMBEDDING_BLOCK).numpy(),
                meta=meta,
            )
            if dino_artifacts is not None:
                state.store.write_dino(dino_temp, dino_artifacts)

        job.progress = 0.8
        state.store.write_segmentation(temp_dir / "segmentation", image_id, seg_result)
        state.store.commit_temp(image_id, temp_dir)
    except Exception:
        state.store.cleanup_temp(image_id)
        raise
