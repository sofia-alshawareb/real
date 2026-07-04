"""Image preprocessing for DINO inference."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ml.lib.constants import IMAGENET_MEAN, IMAGENET_STD, PATCH_SIZE


def prune_vit_blocks(model: nn.Module, num_blocks: int) -> nn.Module:
    model.blocks = nn.ModuleList(list(model.blocks[:num_blocks]))
    if hasattr(model, "n_blocks"):
        model.n_blocks = num_blocks
    return model


def pad_rgb_to_patch_multiple(
    rgb: np.ndarray, patch_size: int = PATCH_SIZE
) -> np.ndarray:
    h, w = rgb.shape[:2]
    pad_h = math.ceil(h / patch_size) * patch_size - h
    pad_w = math.ceil(w / patch_size) * patch_size - w
    if pad_h == 0 and pad_w == 0:
        return rgb
    return np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def rgb_to_dino_tensor(rgb: np.ndarray) -> torch.Tensor:
    padded = pad_rgb_to_patch_multiple(rgb)
    arr = padded.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy((arr - mean) / std).permute(2, 0, 1).unsqueeze(0)
