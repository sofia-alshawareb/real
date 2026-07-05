from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DINO_REPO = _ROOT / "data/models/dinov2"
DEFAULT_DINO_WEIGHTS = _ROOT / "data/models/checkpoints/dinov2_vits14_reg4_pretrain.pth"
FALLBACK_DINO_WEIGHTS = (
    Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth"
)

PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

N_GAUSSIANS_REGIONS = 4
N_GAUSSIANS_BINARY = 2
N_GAUSSIANS_DEFECT_SIM = 4
DEFAULT_REGION_OVERLAP = 0.60
DEFAULT_DEFECT_THRESHOLD = 0.15

COARSE_FINE_DINO_BLOCK = 1
TALC_EMBEDDING_BLOCK = 11
DEFAULT_EMBEDDING_BLOCK = 11
REGION_MAP_INTENSITY_GRADIENT = "intensity_gradient"

SEGMENTATION_MODE_INTENSITY = "intensity"
SEGMENTATION_MODE_EMBEDDING = "embedding"
SEGMENTATION_MODE_HYBRID = "hybrid"

# UI mask class indices (ore-classifier + calibrated segmentation output).
CLS_BACKGROUND = 0
CLS_COARSE = 1
CLS_FINE = 2
CLS_TALC = 3
CLS_MATRIX = 4

# Legacy aliases (deprecated GMM pipeline).
CLS_FOREGROUND = CLS_COARSE
CLS_PARTITIONS = CLS_FINE
CLS_DEFECT = CLS_TALC

CLASS_NAMES = {
    CLS_BACKGROUND: "background",
    CLS_COARSE: "coarse",
    CLS_FINE: "fine",
    CLS_TALC: "talc",
    CLS_MATRIX: "matrix",
}

CLASS_COLORS = {
    CLS_BACKGROUND: (0, 0, 0),
    CLS_COARSE: (46, 125, 50),
    CLS_FINE: (198, 40, 40),
    CLS_TALC: (21, 101, 192),
    CLS_MATRIX: (158, 158, 158),
}

UI_CLASS_NAMES = CLASS_NAMES
UI_CLASS_COLORS = CLASS_COLORS

CALIB_CLASS_KEYS = ("coarse", "fine", "talc", "matrix")
# Embedding-only reference for talc vs background nearest-neighbor (from unlabeled calib pixels).
CALIB_BACKGROUND_KEY = "background"
CALIB_CLASS_ID_BY_KEY = {
    "coarse": CLS_COARSE,
    "fine": CLS_FINE,
    "talc": CLS_TALC,
    "matrix": CLS_MATRIX,
}
CALIB_CLASS_KEY_BY_ID = {v: k for k, v in CALIB_CLASS_ID_BY_KEY.items()}

# Tie-break priority (higher index wins on equal score).
CLASS_TIE_PRIORITY = (CLS_MATRIX, CLS_COARSE, CLS_FINE, CLS_TALC)

MASK_COLOR_TOLERANCE = 1

DEFAULT_CALIBRATION_DIR = _ROOT / "data/calib/compiled"
DEFAULT_RGB_HIST_BINS = 32
DEFAULT_MIN_BACKPROJ_SCORE = 1e-6
DEFAULT_MIN_COSINE_SIM = 0.3
DEFAULT_FG_DILATE_RADIUS = 7
DEFAULT_TALC_REFINE_FG_DILATE_RADIUS = 10
DEFAULT_TALC_GMM_FG_BUFFER_RADIUS = 8
DEFAULT_TALC_GMM_GATE_ERODE = 2
DEFAULT_TALC_BLACK_MAX = 45.0
DEFAULT_TALC_MIN_COSINE = 0.3
DEFAULT_TALC_MIN_COSINE_MARGIN = 0.0
DEFAULT_TALC_CONTOUR_DILATE = 5
DEFAULT_TALC_BLOCK01_OVERLAP = 0.4
DEFAULT_TALC_MARGIN_RELAX = 0.0
# Shift 2-GMM intersection toward the high component mean (gradient refine only).
DEFAULT_TALC_GMM_THRESHOLD_HIGH_BIAS = 0.35
TALC_GMM_MIN_ACTIVATION = 0.5
TALC_REFINE_MODE_DINO = "dino"
TALC_REFINE_MODE_GRADIENT = "gradient"
DEFAULT_TALC_REFINE_MODE = TALC_REFINE_MODE_DINO
DEFAULT_MAX_RGB_SAMPLES = 500_000
DEFAULT_MAX_EMBEDDING_SAMPLES = 50_000

REGION_NAMES = {
    0: "dino_region_0",
    1: "dino_region_1",
    2: "dino_region_2",
    3: "dino_region_3",
}

MASK_SCHEMA_VERSION = "1.0"
API_SCHEMA_VERSION = "1.0"
CALIBRATION_SCHEMA_VERSION = "1.0"
