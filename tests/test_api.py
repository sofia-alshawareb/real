"""API integration tests."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ml.lib.constants import CLASS_COLORS, CLS_COARSE, CLS_TALC
from ml.prepare_calibration import compile_calibration
from services.api import main as api_main
from services.ml_worker.job_runner import JobStatus

_ROOT = Path(__file__).resolve().parents[1]


def _write_mini_calib(root: Path) -> None:
    img_dir = root / "img1"
    img_dir.mkdir(parents=True)
    h, w = 64, 64
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    rgb[20:40, 20:40] = 30
    Image.fromarray(rgb).save(img_dir / "normalized.png")
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[20:40, 20:45] = CLASS_COLORS[CLS_COARSE]
    mask[20:35, 35:40] = CLASS_COLORS[CLS_TALC]
    Image.fromarray(mask).save(img_dir / "user_drawn_colored.png")
    np.save(img_dir / "block01_features.npy", np.random.randn(384, 5, 5).astype(np.float32))


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    calib_src = tmp_path / "calib_src"
    calib_out = tmp_path / "calib_compiled"
    _write_mini_calib(calib_src)
    compile_calibration(calib_src, calib_out, random_state=0)

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
  mode: intensity
  calibration_dir: {calib_out}
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
    assert body["calibration_loaded"] is True
    assert body["segmentation_mode"] == "intensity"
    assert "calibration_counts" in body


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


def test_refine_calibration_bad_hint(golden_image_path, api_client):
    data = golden_image_path.read_bytes()
    up = api_client.post(
        "/api/v1/images",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    image_id = up.json()["image_id"]
    resp = api_client.post(
        f"/api/v1/images/{image_id}/refine/calibration",
        json={"hint_mask": "", "ui_class": "talc"},
    )
    assert resp.status_code == 400


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


@pytest.mark.slow
def test_refine_calibration_after_segment(golden_image_path, api_client):
    import base64
    from io import BytesIO

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
    job_id = job_resp.json()["job_id"]

    deadline = time.time() + 300
    poll: dict = {"status": "pending"}
    while time.time() < deadline:
        poll = api_client.get(f"/api/v1/jobs/{job_id}").json()
        if poll["status"] in (JobStatus.DONE.value, JobStatus.FAILED.value):
            break
        time.sleep(1)
    assert poll["status"] == JobStatus.DONE.value, poll.get("error")

    worker = api_main._worker
    assert worker is not None
    rgb = worker.store.load_rgb(image_id)
    h, w = rgb.shape[:2]
    hint = np.zeros((h, w), dtype=np.uint8)
    hint[h // 4 : h // 2, w // 4 : w // 2] = 1
    buf = BytesIO()
    Image.fromarray(hint * 255, mode="L").save(buf, format="PNG")
    hint_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    refine = api_client.post(
        f"/api/v1/images/{image_id}/refine/calibration",
        json={"hint_mask": hint_b64, "ui_class": "coarse"},
    )
    assert refine.status_code == 200, refine.text
    body = refine.json()
    assert body["encoding"] == "png-p"
    assert body.get("refinement") is not None


def test_save_manual_mask(golden_image_path, api_client):
    import base64
    from io import BytesIO

    data = golden_image_path.read_bytes()
    up = api_client.post(
        "/api/v1/images",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    image_id = up.json()["image_id"]
    worker = api_main._worker
    assert worker is not None
    rgb = worker.store.load_rgb(image_id)
    h, w = rgb.shape[:2]
    ui = np.zeros((h, w), dtype=np.uint8)
    ui[h // 3 : h // 2, w // 3 : w // 2] = 3
    buf = BytesIO()
    Image.fromarray(ui, mode="L").save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    resp = api_client.post(
        f"/api/v1/images/{image_id}/manual-mask",
        json={"mask": mask_b64},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["image_id"] == image_id
    assert body["artifact_path"].endswith("manual")
    assert worker.store.user_drawn_mask_ready(image_id)
    loaded = np.load(worker.store.root / image_id / "manual" / "user_drawn_labels.npy")
    assert int(loaded[h // 3, w // 3]) == 3
