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
DEFAULT_REGION_OVERLAP = 0.60

REGION_MAP_DINO = "dino"
REGION_MAP_INTENSITY_GRADIENT = "intensity_gradient"

CLS_BACKGROUND = 0
CLS_FOREGROUND = 1
CLS_PARTITIONS = 2
CLS_DEFECT = 3

CLASS_NAMES = {
    CLS_BACKGROUND: "background",
    CLS_FOREGROUND: "foreground_object",
    CLS_PARTITIONS: "partitions",
    CLS_DEFECT: "defect",
}

CLASS_COLORS = {
    CLS_BACKGROUND: (40, 40, 40),
    CLS_FOREGROUND: (220, 50, 50),
    CLS_PARTITIONS: (50, 140, 255),
    CLS_DEFECT: (240, 200, 40),
}

REGION_NAMES = {
    0: "dino_region_0",
    1: "dino_region_1",
    2: "dino_region_2",
    3: "dino_region_3",
}

MASK_SCHEMA_VERSION = "1.0"
API_SCHEMA_VERSION = "1.0"
