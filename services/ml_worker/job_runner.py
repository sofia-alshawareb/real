"""Job orchestration: DINO + segmentation + artifact persistence."""

from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue

from services.ml_worker.config import ServiceConfig, load_config
from services.ml_worker.dino_service import DinoInferenceService
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
    dino: DinoInferenceService
    segmentation: SegmentationService
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
    dino = DinoInferenceService(config.dino)
    dino.warmup()
    segmentation = SegmentationService(config.segmentation)
    state = WorkerState(
        config=config,
        store=store,
        dino=dino,
        segmentation=segmentation,
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

    try:
        dino_result, dino_artifacts = state.dino.run(
            rgb, save_activations=job.save_activations
        )
        job.progress = 0.5

        block01 = dino_result.activation(1)
        seg_result = state.segmentation.run(rgb, block01)
        job.progress = 0.8

        if dino_artifacts is not None:
            state.store.write_dino(temp_dir / "dino", dino_artifacts)

        state.store.write_segmentation(
            temp_dir / "segmentation", image_id, seg_result
        )
        state.store.commit_temp(image_id, temp_dir)
    except Exception:
        state.store.cleanup_temp(image_id)
        raise
