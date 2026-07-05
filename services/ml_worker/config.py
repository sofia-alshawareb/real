"""Load ML service configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _ROOT / "configs/ml_service.yaml"


@dataclass
class DinoConfig:
    repo: Path
    weights: Path
    num_blocks: int = 12
    inference_blocks: list[int] = field(default_factory=lambda: [1, 11])
    save_activations: bool = False
    device: str = ""


@dataclass
class SegmentationConfig:
    mode: str = "hybrid"
    calibration_dir: Path = field(default_factory=lambda: _ROOT / "data/calib/compiled")
    rgb_hist_bins: int = 32
    min_backproj_score: float = 1e-6
    min_cosine_sim: float = 0.3
    close_radius: int = 3
    random_state: int = 0
    max_samples: int = 300_000
    max_rgb_samples: int = 500_000
    max_embedding_samples: int = 50_000
    preprocess: bool = False
    denoise: bool = True
    illum_sigma: float = 64.0
    region_overlap: float = 0.60
    block_index: int = 1
    fg_dilate_radius: int = 7
    talc_refine_fg_dilate_radius: int = 10
    talc_gmm_fg_buffer_radius: int = 8
    talc_gmm_gate_erode: int = 2
    talc_black_max: float = 45.0
    talc_min_cosine: float = 0.3
    talc_min_cosine_margin: float = 0.0
    talc_contour_dilate: int = 4
    tile_threshold: int = 2000
    tile_grid: int = 2
    talc_block01_overlap: float = 0.4
    talc_gmm_threshold_high_bias: float = 0.35
    talc_margin_relax: float = 0.0
    talc_refine_mode: str = "dino"


@dataclass
class StorageConfig:
    root: Path
    images_subdir: str = "images"


@dataclass
class ServiceConfig:
    dino: DinoConfig
    segmentation: SegmentationConfig
    storage: StorageConfig
    schema_version: str = "1.0"


def _resolve_path(value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = _ROOT / p
    return p


def load_config(path: Path | str | None = None) -> ServiceConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dino_raw = raw.get("dino", {})
    seg_raw = raw.get("segmentation", {})
    storage_raw = raw.get("storage", {})
    return ServiceConfig(
        dino=DinoConfig(
            repo=_resolve_path(dino_raw.get("repo", "data/models/dinov2")),
            weights=_resolve_path(
                dino_raw.get(
                    "weights",
                    "data/models/checkpoints/dinov2_vits14_reg4_pretrain.pth",
                )
            ),
            num_blocks=int(dino_raw.get("num_blocks", 12)),
            inference_blocks=[int(x) for x in dino_raw.get("inference_blocks", [1, 11])],
            save_activations=bool(dino_raw.get("save_activations", False)),
            device=str(dino_raw.get("device", "")),
        ),
        segmentation=SegmentationConfig(
            mode=str(seg_raw.get("mode", "hybrid")),
            calibration_dir=_resolve_path(
                seg_raw.get("calibration_dir", "data/calib/compiled")
            ),
            rgb_hist_bins=int(seg_raw.get("rgb_hist_bins", 32)),
            min_backproj_score=float(seg_raw.get("min_backproj_score", 1e-6)),
            min_cosine_sim=float(seg_raw.get("min_cosine_sim", 0.3)),
            close_radius=int(seg_raw.get("close_radius", 3)),
            random_state=int(seg_raw.get("random_state", 0)),
            max_samples=int(seg_raw.get("max_samples", 300_000)),
            max_rgb_samples=int(seg_raw.get("max_rgb_samples", 500_000)),
            max_embedding_samples=int(seg_raw.get("max_embedding_samples", 50_000)),
            preprocess=bool(seg_raw.get("preprocess", False)),
            denoise=bool(seg_raw.get("denoise", True)),
            illum_sigma=float(seg_raw.get("illum_sigma", 64.0)),
            region_overlap=float(seg_raw.get("region_overlap", 0.60)),
            block_index=int(seg_raw.get("block_index", 1)),
            fg_dilate_radius=int(seg_raw.get("fg_dilate_radius", 7)),
            talc_refine_fg_dilate_radius=int(
                seg_raw.get("talc_refine_fg_dilate_radius", 10)
            ),
            talc_gmm_fg_buffer_radius=int(
                seg_raw.get("talc_gmm_fg_buffer_radius", 8)
            ),
            talc_gmm_gate_erode=int(seg_raw.get("talc_gmm_gate_erode", 2)),
            talc_black_max=float(seg_raw.get("talc_black_max", 45.0)),
            talc_min_cosine=float(
                seg_raw.get("talc_min_cosine", seg_raw.get("talc_gradient_threshold", 0.3))
            ),
            talc_min_cosine_margin=float(seg_raw.get("talc_min_cosine_margin", 0.0)),
            talc_contour_dilate=int(seg_raw.get("talc_contour_dilate", 2)),
            tile_threshold=int(seg_raw.get("tile_threshold", 2000)),
            tile_grid=int(seg_raw.get("tile_grid", 2)),
            talc_block01_overlap=float(seg_raw.get("talc_block01_overlap", 0.4)),
            talc_gmm_threshold_high_bias=float(
                seg_raw.get("talc_gmm_threshold_high_bias", 0.35)
            ),
            talc_margin_relax=float(seg_raw.get("talc_margin_relax", 0.0)),
            talc_refine_mode=str(seg_raw.get("talc_refine_mode", "dino")),
        ),
        storage=StorageConfig(
            root=_resolve_path(storage_raw.get("root", "data/artifacts")),
            images_subdir=str(storage_raw.get("images_subdir", "images")),
        ),
        schema_version=str(raw.get("schema_version", "1.0")),
    )
