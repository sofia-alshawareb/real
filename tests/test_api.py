"""API integration tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.api import main as api_main
from services.ml_worker.job_runner import JobStatus


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    cfg = tmp_path / "ml_service.yaml"
    cfg.write_text(
        f"""
dino:
  repo: data/models/dinov2
  weights: data/models/checkpoints/dinov2_vits14_reg4_pretrain.pth
  num_blocks: 12
  inference_blocks: [1, 11]
  save_activations: false
segmentation:
  region_map: dino
  block_index: 1
storage:
  root: {tmp_path / "artifacts"}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ML_SERVICE_CONFIG", str(cfg))
    api_main._worker = None
    with TestClient(api_main.app) as client:
        api_main.startup()
        yield client


def test_health(api_client):
    resp = api_client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "starting")
    assert "schema_version" in body


def test_upload_and_idempotent(golden_image_path, api_client):
    data = golden_image_path.read_bytes()
    r1 = api_client.post(
        "/api/v1/images",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    assert r1.status_code == 200
    image_id = r1.json()["image_id"]
    r2 = api_client.post(
        "/api/v1/images",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    assert r2.json()["image_id"] == image_id


def test_mask_409_before_job(golden_image_path, api_client):
    data = golden_image_path.read_bytes()
    up = api_client.post(
        "/api/v1/images",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    image_id = up.json()["image_id"]
    resp = api_client.get(f"/api/v1/images/{image_id}/mask")
    assert resp.status_code == 409


@pytest.mark.slow
def test_segment_job_flow(golden_image_path, api_client):
    data = golden_image_path.read_bytes()
    up = api_client.post(
        "/api/v1/images",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    image_id = up.json()["image_id"]
    job_resp = api_client.post(
        "/api/v1/jobs/segment",
        json={"image_id": image_id, "save_activations": False},
    )
    assert job_resp.status_code == 200
    job_id = job_resp.json()["job_id"]

    deadline = time.time() + 300
    poll: dict = {"status": "pending"}
    while time.time() < deadline:
        poll = api_client.get(f"/api/v1/jobs/{job_id}").json()
        if poll["status"] in (JobStatus.DONE.value, JobStatus.FAILED.value):
            break
        time.sleep(1)

    assert poll["status"] == JobStatus.DONE.value, poll.get("error")
    mask = api_client.get(f"/api/v1/images/{image_id}/mask")
    assert mask.status_code == 200
    body = mask.json()
    assert body["encoding"] == "png-p"
    assert body["mask_to_native_scale"] == 1.0
    assert "classes" in body
    assert body["data"]
