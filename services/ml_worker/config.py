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
    region_map: str = "dino"
    block_index: int = 1
    region_overlap: float = 0.6
    close_radius: int = 3
    random_state: int = 0
    max_samples: int = 300_000
    preprocess: bool = False
    denoise: bool = True
    illum_sigma: float = 64.0


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
            region_map=str(seg_raw.get("region_map", "dino")),
            block_index=int(seg_raw.get("block_index", 1)),
            region_overlap=float(seg_raw.get("region_overlap", 0.6)),
            close_radius=int(seg_raw.get("close_radius", 3)),
            random_state=int(seg_raw.get("random_state", 0)),
            max_samples=int(seg_raw.get("max_samples", 300_000)),
            preprocess=bool(seg_raw.get("preprocess", False)),
            denoise=bool(seg_raw.get("denoise", True)),
            illum_sigma=float(seg_raw.get("illum_sigma", 64.0)),
        ),
        storage=StorageConfig(
            root=_resolve_path(storage_raw.get("root", "data/artifacts")),
            images_subdir=str(storage_raw.get("images_subdir", "images")),
        ),
        schema_version=str(raw.get("schema_version", "1.0")),
    )
