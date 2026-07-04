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
from ml.lib.constants import SEGMENTATION_MODE_EMBEDDING, SEGMENTATION_MODE_HYBRID
from ml.lib.types import SegmentationResult

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


def _ensure_block01(state: WorkerState, image_id: str, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if state.store.dino_features_ready(image_id):
        return (
            state.store.load_block01_activation(image_id),
            state.store.load_block01_features(image_id),
        )
    dino_result, _ = state.dino.run(rgb, save_activations=False)
    dino_dir = state.store._dino_dir(image_id)
    block01_act = dino_result.activation(1)
    block01 = dino_result.features(1).numpy()
    state.store.write_block01_features(
        dino_dir,
        block01,
        activation=block01_act,
        meta={"inference_blocks": dino_result.inference_blocks},
    )
    return block01_act, block01


def refine_calibration_from_hint(
    state: WorkerState,
    image_id: str,
    hint_mask: np.ndarray,
    ui_class: str,
) -> SegmentationResult:
    if not state.store.image_exists(image_id):
        raise FileNotFoundError(f"Unknown image_id: {image_id}")

    rgb = state.store.load_rgb(image_id)
    block01_act, block01 = _ensure_block01(state, image_id, rgb)

    result = state.refinement.refine(
        rgb, block01, hint_mask, ui_class, block01_activation=block01_act
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
        block01_activation = None
        block01_features = None
        if needs_dino:
            dino_result, dino_artifacts = state.dino.run(
                rgb, save_activations=job.save_activations
            )
            job.progress = 0.5
            block01_activation = dino_result.activation(1)
            block01_features = dino_result.features(1)
            block01_np = block01_features.numpy()
            dino_temp = temp_dir / "dino"
            state.store.write_block01_features(
                dino_temp,
                block01_np,
                activation=block01_activation,
                meta={"inference_blocks": dino_result.inference_blocks},
            )
            if dino_artifacts is not None:
                state.store.write_dino(dino_temp, dino_artifacts)

        seg_result = state.segmentation.run(
            rgb,
            calib,
            block01_activation=block01_activation,
            block01_features=block01_features,
        )
        job.progress = 0.8

        state.store.write_segmentation(
            temp_dir / "segmentation", image_id, seg_result
        )
        state.store.commit_temp(image_id, temp_dir)
    except Exception:
        state.store.cleanup_temp(image_id)
        raise
