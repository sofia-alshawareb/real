# ML segmentation backend service

Standalone FastAPI service for **calibration-driven segmentation**. The ore-classifier frontend calls it when the demo switch is off.

## Layout

```
ml/lib/                 # Algorithm library + calibration modules
ml/prepare_calibration.py  # One-time compile of data/calib/img*
data/calib/compiled/    # Compiled RGB + embedding calibration (generated)
services/api/           # FastAPI REST endpoints
services/ml_worker/     # DINO + calibrated segmentation orchestration
configs/ml_service.yaml # Service configuration
data/artifacts/         # Persisted inference masks (gitignored)
```

## Requirements

```bash
pip install -r requirements.txt
```

Ensure DINOv2 repo and weights exist under `data/models/` (required for `hybrid` mode and talc embedding refinement).

## Prepare calibration (one-time)

Place per-image folders under `data/calib/img*/`:

- `normalized.png`
- `user_drawn_colored.png` (editor palette colors)
- `block01_features.npy` (run extraction script below)

Extract block-1 features for all calibration images:

```bash
python ml/extract_calib_block01_features.py --calib-root data/calib
```

Compile:

```bash
python ml/prepare_calibration.py \
  --calib-root data/calib \
  --output data/calib/compiled
```

Output:

```
data/calib/compiled/
  summary.json
  rgb/{coarse,fine,talc,matrix}.npy
  embeddings/{coarse,fine,talc,matrix}.npy
  rgb_histograms.npz
```

**Talc decontamination:** blue hand-drawn masks often include bright background. The prep script applies a 2-GMM dark-component filter so only darker talc pixels enter calibration.

## Run the API

```bash
export ML_SERVICE_CONFIG=configs/ml_service.yaml
python -m services.api.main
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Returns `calibration_loaded`, `segmentation_mode`, and per-class sample counts.

## Segmentation modes

Configure `segmentation.mode` in `configs/ml_service.yaml`:

| Mode | Algorithm | DINO required |
|------|-----------|---------------|
| `hybrid` (default) | Coarse/fine: intensity 2-GMM + block-1 region map; talc: perceptual black threshold (RGB mean + max channel) outside dilated foreground, capped by `talc_black_max` | Yes (block 1) |
| `intensity` | RGB histogram backprojection vs compiled calib | No |
| `embedding` | Block-1 patch cosine similarity vs class mean embeddings | Yes (block 1) |

Output mask classes (UI indices): `0=background`, `1=coarse`, `2=fine`, `3=talc`, `4=matrix`.

## API (v1)

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/images` | Upload image → `{ image_id, width, height }` |
| `POST /api/v1/jobs/segment` | Run calibrated segmentation job |
| `GET /api/v1/jobs/{job_id}` | Poll job status |
| `GET /api/v1/images/{image_id}/mask` | Paletted PNG mask (409 until job done) |
| `POST /api/v1/images/{image_id}/refine/calibration` | Append hint samples to calib + re-segment |
| `POST /api/v1/images/{image_id}/refine/defect` | Alias for talc refine (backward compatible) |
| `GET /api/v1/calibration/summary` | Read compiled calibration summary |
| `POST /api/v1/images/{image_id}/manual-mask` | Dev: save hand-drawn mask to artifacts |

### Interactive refinement

User stroke → hint mask → backend appends RGB + embedding samples to `data/calib/compiled/` → full image re-segmented.

```json
{
  "hint_mask": "<base64 PNG, mode L, 1=hint pixel>",
  "ui_class": "coarse | fine | talc | matrix"
}
```

Talc hints are filtered with the same dark-component GMM before append.

## Configuration

See `configs/ml_service.yaml`:

- `segmentation.mode` — `intensity` or `embedding`
- `segmentation.calibration_dir` — path to compiled calibration
- `segmentation.min_backproj_score` — RGB mode background threshold
- `segmentation.min_cosine_sim` — embedding mode background threshold
- `segmentation.rgb_hist_bins` — histogram resolution (default 32)

## Frontend integration (ore-classifier)

1. Run calibration prep (above).
2. Start ML API on port 8000.
3. `cd ore-classifier && npm run dev`
4. Leave **«Демо: локальная имитация ML»** **OFF**.

Drawing with coarse/fine/talc triggers debounced `POST /refine/calibration`; the returned mask replaces the editor overlay entirely.

## Tests

```bash
pytest tests/ -v
```

Slow integration tests (DINO forward) require the golden fixture image; they skip when unavailable.
