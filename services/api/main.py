"""FastAPI application for ML segmentation service."""

from __future__ import annotations

import base64
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml.lib.constants import API_SCHEMA_VERSION, CALIB_CLASS_KEYS, CLASS_NAMES
from ml.lib.imaging import image_id_from_bytes, load_rgb_from_bytes
from services.ml_worker.job_runner import (
    JobStatus,
    WorkerState,
    create_worker,
    enqueue_segment_job,
    get_job,
    refine_calibration_from_hint,
    refine_defect_from_hint,
)

app = FastAPI(title="ML Segmentation Service", version=API_SCHEMA_VERSION)

_cors_origins = os.environ.get(
    "ML_API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class RefineCalibrationRequest(BaseModel):
    hint_mask: str = Field(..., description="Base64-encoded PNG (mode L), 1=hint pixels")
    ui_class: str = Field(..., description="coarse | fine | talc | matrix")


class RefineDefectRequest(RefineCalibrationRequest):
    ui_class: str = "talc"


class SaveManualMaskRequest(BaseModel):
    mask: str = Field(..., description="Base64 PNG (mode L), pixel value = UI class index 0–4")


def _decode_hint_mask_png(b64: str, expected_hw: tuple[int, int]) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
        arr = np.asarray(Image.open(BytesIO(raw)).convert("L"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid hint_mask PNG: {exc}") from exc
    eh, ew = expected_hw
    if arr.shape != (eh, ew):
        raise HTTPException(
            status_code=400,
            detail=f"hint_mask size {arr.shape[::-1]} != image size {(ew, eh)}",
        )
    return (arr > 0).astype(np.uint8)


def _decode_ui_manual_mask_png(b64: str, expected_hw: tuple[int, int]) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
        arr = np.asarray(Image.open(BytesIO(raw)).convert("L"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid mask PNG: {exc}") from exc
    eh, ew = expected_hw
    if arr.shape != (eh, ew):
        raise HTTPException(
            status_code=400,
            detail=f"mask size {arr.shape[::-1]} != image size {(ew, eh)}",
        )
    if int(arr.max()) > 4:
        arr = np.minimum(4, (arr // 64).astype(np.uint8))
    return arr.astype(np.uint8)


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
    calib_counts = worker.calib_store.summary_counts()
    return {
        "status": "ok" if worker.dino.is_ready else "starting",
        "schema_version": API_SCHEMA_VERSION,
        "model_ready": worker.dino.is_ready,
        "device": str(worker.dino.device),
        "calibration_loaded": worker.calib_store.exists(),
        "segmentation_mode": worker.config.segmentation.mode,
        "calibration_counts": calib_counts,
    }


@app.get("/api/v1/calibration/summary")
def calibration_summary() -> JSONResponse:
    worker = _require_worker()
    summary_path = worker.calib_store.summary_path
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Calibration not found")
    import json

    return JSONResponse(content=json.loads(summary_path.read_text(encoding="utf-8")))


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


@app.post("/api/v1/images/{image_id}/refine/calibration")
def refine_calibration(image_id: str, body: RefineCalibrationRequest) -> JSONResponse:
    worker = _require_worker()
    if body.ui_class not in CALIB_CLASS_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ui_class: {body.ui_class!r} (expected one of {CALIB_CLASS_KEYS})",
        )
    if not worker.store.image_exists(image_id):
        raise HTTPException(status_code=404, detail="Image not found")

    rgb = worker.store.load_rgb(image_id)
    hint = _decode_hint_mask_png(body.hint_mask, rgb.shape[:2])
    if not np.any(hint):
        raise HTTPException(status_code=400, detail="Empty hint mask")

    try:
        result = refine_calibration_from_hint(worker, image_id, hint, body.ui_class)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    meta = worker.store.get_mask_metadata(image_id)
    png_bytes = worker.store.get_mask_bytes(image_id)
    classes = {str(k): v for k, v in CLASS_NAMES.items()}
    payload = {
        "schema_version": API_SCHEMA_VERSION,
        "image_id": image_id,
        "mask_width": result.mask_width,
        "mask_height": result.mask_height,
        "native_width": result.native_width,
        "native_height": result.native_height,
        "mask_to_native_scale": result.mask_to_native_scale,
        "classes": meta.get("classes", classes),
        "encoding": "png-p",
        "data": base64.b64encode(png_bytes).decode("ascii"),
        "refinement": result.metadata.get("refinement"),
    }
    return JSONResponse(content=payload)


@app.post("/api/v1/images/{image_id}/refine/defect")
def refine_defect(image_id: str, body: RefineDefectRequest) -> JSONResponse:
    """Backward-compatible alias for talc calibration refinement."""
    return refine_calibration(
        image_id,
        RefineCalibrationRequest(hint_mask=body.hint_mask, ui_class=body.ui_class),
    )


@app.post("/api/v1/images/{image_id}/manual-mask")
def save_manual_mask(image_id: str, body: SaveManualMaskRequest) -> JSONResponse:
    """Save hand-drawn UI mask to artifact store (dev tooling)."""
    worker = _require_worker()
    if not worker.store.image_exists(image_id):
        raise HTTPException(status_code=404, detail="Image not found")

    rgb = worker.store.load_rgb(image_id)
    labels = _decode_ui_manual_mask_png(body.mask, rgb.shape[:2])
    manual_dir = worker.store.save_user_drawn_mask(image_id, labels)
    rel = manual_dir.relative_to(worker.store.root)
    return JSONResponse(
        content={
            "schema_version": API_SCHEMA_VERSION,
            "image_id": image_id,
            "artifact_path": str(rel),
            "width": int(labels.shape[1]),
            "height": int(labels.shape[0]),
            "files": [
                "user_drawn_labels.npy",
                "user_drawn_colored.png",
                "user_drawn_grayscale.png",
                "meta.json",
            ],
        }
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("ML_API_HOST", "0.0.0.0")
    port = int(os.environ.get("ML_API_PORT", "8000"))
    uvicorn.run("services.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
