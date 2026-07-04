"""FastAPI application for ML segmentation service."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

# R7: set BLAS threads before numpy import in worker/API process.
for _env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml.lib.constants import API_SCHEMA_VERSION, CLASS_NAMES
from ml.lib.imaging import image_id_from_bytes, load_rgb_from_bytes
from services.ml_worker.job_runner import (
    JobStatus,
    WorkerState,
    create_worker,
    enqueue_segment_job,
    get_job,
)

app = FastAPI(title="ML Segmentation Service", version=API_SCHEMA_VERSION)
_worker: WorkerState | None = None


class SegmentJobRequest(BaseModel):
    image_id: str
    save_activations: bool = False


class JobResponse(BaseModel):
    job_id: str
    image_id: str
    status: str
    progress: float
    error: str | None = None


class UploadResponse(BaseModel):
    image_id: str
    width: int
    height: int
    schema_version: str = API_SCHEMA_VERSION


@app.on_event("startup")
def startup() -> None:
    global _worker
    config_path = os.environ.get("ML_SERVICE_CONFIG")
    _worker = create_worker(config_path)


def _require_worker() -> WorkerState:
    if _worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    return _worker


@app.get("/api/v1/health")
def health() -> dict:
    worker = _require_worker()
    return {
        "status": "ok" if worker.dino.is_ready else "starting",
        "schema_version": API_SCHEMA_VERSION,
        "model_ready": worker.dino.is_ready,
        "device": str(worker.dino.device),
    }


@app.post("/api/v1/images", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    worker = _require_worker()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        rgb = load_rgb_from_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported image: {exc}") from exc

    image_id = image_id_from_bytes(raw)
    if not worker.store.image_exists(image_id):
        worker.store.save_image(image_id, rgb, raw)

    h, w = rgb.shape[:2]
    return UploadResponse(image_id=image_id, width=w, height=h)


@app.post("/api/v1/jobs/segment", response_model=JobResponse)
def create_segment_job(body: SegmentJobRequest) -> JobResponse:
    worker = _require_worker()
    try:
        job = enqueue_segment_job(
            worker,
            body.image_id,
            save_activations=body.save_activations,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobResponse(
        job_id=job.job_id,
        image_id=job.image_id,
        status=job.status.value,
        progress=job.progress,
        error=job.error,
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def poll_job(job_id: str) -> JobResponse:
    worker = _require_worker()
    try:
        job = get_job(worker, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return JobResponse(
        job_id=job.job_id,
        image_id=job.image_id,
        status=job.status.value,
        progress=job.progress,
        error=job.error,
    )


@app.get("/api/v1/images/{image_id}/mask")
def get_mask(image_id: str) -> JSONResponse:
    worker = _require_worker()
    if not worker.store.image_exists(image_id):
        raise HTTPException(status_code=404, detail="Image not found")
    if not worker.store.segmentation_ready(image_id):
        raise HTTPException(status_code=409, detail="Segmentation not ready")
    meta = worker.store.get_mask_metadata(image_id)
    png_bytes = worker.store.get_mask_bytes(image_id)
    classes = {str(k): v for k, v in CLASS_NAMES.items()}
    payload = {
        "schema_version": API_SCHEMA_VERSION,
        "image_id": image_id,
        "mask_width": meta.get("mask_width", meta.get("native_width")),
        "mask_height": meta.get("mask_height", meta.get("native_height")),
        "native_width": meta.get("native_width"),
        "native_height": meta.get("native_height"),
        "mask_to_native_scale": meta.get("mask_to_native_scale", 1.0),
        "classes": meta.get("classes", classes),
        "encoding": "png-p",
        "data": base64.b64encode(png_bytes).decode("ascii"),
    }
    return JSONResponse(content=payload)


def main() -> None:
    import uvicorn

    host = os.environ.get("ML_API_HOST", "0.0.0.0")
    port = int(os.environ.get("ML_API_PORT", "8000"))
    uvicorn.run("services.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
