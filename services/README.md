# ML segmentation backend service

Standalone FastAPI service for DINO-based GMM segmentation. Phase 1 delivers masks via REST; ore-classifier integration is deferred to WITHML.

## Layout

```
ml/lib/                 # Pure algorithm library
services/api/           # FastAPI REST endpoints
services/ml_worker/     # DINO + segmentation orchestration
configs/ml_service.yaml # Service configuration
data/artifacts/         # Persisted masks (gitignored)
```

## Requirements

Install Python dependencies (PyTorch with CUDA separately if needed):

```bash
pip install -r requirements.txt
```

Ensure DINOv2 repo and weights exist under `data/models/`.

## Run the API

```bash
export ML_SERVICE_CONFIG=configs/ml_service.yaml
python -m services.api.main
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

## API (v1)

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/images` | Upload image → `{ image_id, width, height }` |
| `POST /api/v1/jobs/segment` | `{ image_id, save_activations? }` → `{ job_id }` |
| `GET /api/v1/jobs/{job_id}` | Poll job status |
| `GET /api/v1/images/{image_id}/mask` | Paletted PNG mask (409 until job done) |

Mask response includes `mask_width`, `mask_height`, `native_width`, `native_height`, `mask_to_native_scale`, and explicit `classes` map.

## Configuration

See `configs/ml_service.yaml`:

- `dino.save_activations` (default `false`) — persist block 1+11 `.npy` artifacts
- `dino.num_blocks: 12` — required for block 11 extraction
- `segmentation.block_index: 1` — block-1 DINO activation for region GMM

## Tests

```bash
pytest tests/ -v
```

Slow integration tests (DINO forward) require the golden fixture image and GPU; they skip automatically when unavailable.

## CLI (unchanged)

The original segmentation CLI still works:

```bash
python ml/intensity_gmm_segment.py --input path/to/image.jpg
```

GMM fitting now uses `ThreadPoolExecutor` instead of `fork` for CUDA-safe server reuse.
