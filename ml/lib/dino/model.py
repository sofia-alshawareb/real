"""DINO model loading and caching."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml.lib.constants import (
    DEFAULT_DINO_REPO,
    DEFAULT_DINO_WEIGHTS,
    FALLBACK_DINO_WEIGHTS,
)
from ml.lib.dino.preprocess import prune_vit_blocks


def resolve_dino_weights(path: str | Path = "") -> str:
    if path:
        p = Path(path)
        if p.exists():
            return str(p)
    if DEFAULT_DINO_WEIGHTS.exists():
        return str(DEFAULT_DINO_WEIGHTS)
    if FALLBACK_DINO_WEIGHTS.exists():
        return str(FALLBACK_DINO_WEIGHTS)
    return ""


def validate_block_indices(num_blocks: int, block_indices: list[int]) -> None:
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}.")
    for idx in block_indices:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(
                f"block index {idx} out of range [0, {num_blocks - 1}]"
            )


class DinoModelCache:
    """Thread-safe singleton cache for loaded DINO models."""

    _lock = threading.Lock()
    _models: dict[tuple[str, str, int, str], nn.Module] = {}

    @classmethod
    def get(
        cls,
        *,
        repo_dir: Path | str,
        weights: str,
        num_blocks: int,
        device: torch.device,
    ) -> nn.Module:
        key = (str(repo_dir), weights, num_blocks, str(device))
        with cls._lock:
            if key not in cls._models:
                cls._models[key] = load_dinov2(
                    Path(repo_dir),
                    weights,
                    num_blocks=num_blocks,
                    device=device,
                )
            return cls._models[key]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._models.clear()


def load_dinov2(
    repo_dir: Path,
    weights: str,
    *,
    num_blocks: int,
    device: torch.device,
) -> nn.Module:
    kwargs: dict[str, Any] = {"pretrained": True}
    if weights:
        kwargs["weights"] = weights
    model = torch.hub.load(
        str(repo_dir), "dinov2_vits14_reg", source="local", **kwargs
    )
    if num_blocks > 0:
        prune_vit_blocks(model, num_blocks)
    return model.to(device).eval()
