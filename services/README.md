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

Ensure DINOv2 repo and weights exist under `data/models/` (required for `hybrid` mode and online calibration hints).

## Prepare calibration (one-time)

Place per-image folders under `data/calib/img*/`:

- `normalized.png`
- `user_drawn_colored.png` (editor palette colors)
- `block01_features.npy`, `block01_activation.npy`, and `block11_features.npy` (run extraction script below)

For **talc intensity calibration**, paint **sample regions** in `user_drawn_colored.png` (partial examples — not every talc/background pixel):
- **Talc** — blue sample areas (may include some smooth background at edges)
- **Background** — matrix grey sample areas only (smooth matrix behind talc)

Then re-run compile; `summary.json` stores calibrated `talc_intensity_max`. **Hybrid talc refine** (`segmentation.talc_refine_mode`):

- **`dino`** (default): dark coarse gate → 2-GMM on block-1 activation inside gate; threshold = mean of high-activation Gaussian; low activation = talc.
- **`gradient`**: dark coarse gate → 2-GMM on Sobel gradient; biased intersection threshold; high gradient = talc; CC hole-fill.

### Artifact sample calibration

After labeling sample regions in the UI, artifacts under `data/artifacts/<id>/manual/` can be used directly:

```bash
python ml/calibrate_talc_from_artifacts.py \
  --artifacts-root data/artifacts \
  --calibration-dir data/calib/compiled
```

Requires per artifact: `images/normalized.png`, `manual/user_drawn_colored.png`, `dino/block11_features.npy`, with both blue (talc) and grey (matrix) sample strokes.

Verify block-11 talc margin on labeled artifacts:

```bash
python ml/verify_talc_block11.py
```

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
| `hybrid` (default) | Coarse/fine: block-1 activation; talc: coarse gate + `talc_refine_mode` (`dino` or `gradient`) | Yes (block 1; block 11 for online calibration hints) |
| `intensity` | RGB histogram backprojection vs compiled calib | No |
| `embedding` | Block-1 patch cosine similarity vs class mean embeddings | Yes (block 1) |

Output mask classes (UI indices): `0=background`, `1=coarse`, `2=fine`, `3=talc`, `4=matrix`.

**Large images:** when both width and height exceed **2000 px**, the worker splits the image into a **2×2** patch-aligned grid, runs DINO + segmentation on each tile, and stitches masks and DINO features back together. Configure via `segmentation.tile_threshold` and `segmentation.tile_grid` in `configs/ml_service.yaml`.

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

Talc hints are filtered with the same black threshold (`talc_black_max`) before append.

## Configuration

See `configs/ml_service.yaml`:

- `segmentation.mode` — `intensity` or `embedding`
- `segmentation.calibration_dir` — path to compiled calibration
- `segmentation.min_backproj_score` — RGB mode background threshold
- `segmentation.min_cosine_sim` — embedding mode background threshold
- `segmentation.talc_black_max` — fallback gray ceiling for talc search region (default 45; calibrated as `talc_intensity_max`)
- `segmentation.fg_dilate_radius` — FG dilation for coarse talc gate exclusion (default 7)
- `segmentation.talc_refine_mode` — `dino` (block-1 activation 2-GMM) or `gradient` (Sobel + CC fill); default `dino`
- `segmentation.talc_gmm_threshold_high_bias` — gradient mode only: shift 2-GMM cutoff toward high gradient (default 0.35)
- `segmentation.talc_refine_fg_dilate_radius` — FG dilation for talc refine clip (default 10)
- `segmentation.talc_contour_dilate` — dilation of coarse talc gate before refine (default 5)
- `segmentation.talc_gmm_fg_buffer_radius` — exclude FG border band from step-7 GMM fit (default 8 px)
- `segmentation.talc_gmm_gate_erode` — erode step-5 gate rim before GMM fit (default 2 px)
- `segmentation.tile_threshold` — both axes must exceed this to enable 2×2 tiling (default 2000)
- `segmentation.tile_grid` — grid size per axis when tiling (default 2)
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

### Debug hybrid talc masks (local)

Save every pipeline step as PNG overlays under `outputs/hybrid_talc_debug/<image_stem>/`:

```bash
python ml/debug_hybrid_talc.py data/calib/img1/normalized.png \
  --block01-activation data/calib/img1/block01_activation.npy
```

For your own image (runs DINO block-1 if activation not provided):

```bash
python ml/debug_hybrid_talc.py path/to/normalized.png \
  --config configs/ml_service.yaml \
  --out-dir outputs/hybrid_talc_debug
```

| File | Meaning |
|------|---------|
| `04_coarse_talc_gate.png` | Dark-pixel coarse talc |
| `05_coarse_gate_dilated_for_refine.png` | Coarse gate after `talc_contour_dilate` — steps 7–10 run only inside this mask |
| `05b_gmm_fit_interior.png` | Interior used for step-7 GMM fit (excludes FG border + gate rim) |
| `06_intensity_gradient.png` | Sobel gradient magnitude heatmap |
| `06b_talc_gmm_histogram.png` | 2-GMM on interior gradient |
| `07b_talc_band_filled_cc.png` | High-gradient CCs after hole fill |
| `06b_talc_gmm_histogram.png` | 2-GMM histogram on interior + chosen threshold |
| `07_talc_band_2gmm.png` | Inside step 5: activation band with lower mean image intensity (= talc) |
| `08_background_band_2gmm.png` | Inside step 5: the other 2-GMM band |
| `09_refined_talc_before_fg_clip.png` | Talc band after morph close (still inside step 5) |
| `10_refined_talc_final.png` | After FG exclusion |
| `13_final_segmentation.png` | Full palette mask |
| `summary.json` | Pixel counts + metadata per step |

Restart the API after config changes.
