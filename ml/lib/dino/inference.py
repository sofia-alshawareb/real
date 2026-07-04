"""DINO patch-feature extraction and activation maps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.lib.dino.model import DinoModelCache, validate_block_indices
from ml.lib.dino.preprocess import rgb_to_dino_tensor
from ml.lib.types import DinoBlockOutput, DinoInferenceResult


def features_to_activation(
    feats: torch.Tensor,
    target_hw: tuple[int, int],
) -> np.ndarray:
    act = torch.linalg.vector_norm(feats, dim=0)
    lo, hi = float(act.min()), float(act.max())
    if hi > lo:
        act = (act - lo) / (hi - lo)
    else:
        act = torch.zeros_like(act)
    act_up = F.interpolate(
        act.unsqueeze(0).unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return act_up.squeeze().numpy().astype(np.float32)


def upsample_patch_map(patch_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(patch_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return up.squeeze().numpy().astype(np.float32)


@torch.inference_mode()
def extract_multi_block_features(
    rgb: np.ndarray,
    *,
    device: torch.device,
    block_indices: Sequence[int],
    num_blocks: int,
    model: nn.Module | None = None,
    repo_dir: Path | None = None,
    weights: str = "",
) -> DinoInferenceResult:
    """Single forward pass; extract patch features for requested block indices."""
    block_list = sorted(set(int(b) for b in block_indices))
    validate_block_indices(num_blocks, block_list)

    if model is None:
        assert repo_dir is not None
        model = DinoModelCache.get(
            repo_dir=repo_dir,
            weights=weights,
            num_blocks=num_blocks,
            device=device,
        )

    h, w = rgb.shape[:2]
    image = rgb_to_dino_tensor(rgb).to(device)
    outputs = model.get_intermediate_layers(
        image,
        n=block_list,
        reshape=True,
        norm=True,
    )

    blocks: dict[int, DinoBlockOutput] = {}
    for block_idx, tokens in zip(block_list, outputs):
        feats = tokens[0].detach().float().cpu()
        activation = features_to_activation(feats, (h, w))
        blocks[block_idx] = DinoBlockOutput(
            block_index=block_idx,
            features=feats,
            activation=activation,
        )

    del image, outputs
    return DinoInferenceResult(
        blocks=blocks,
        native_width=w,
        native_height=h,
        inference_blocks=block_list,
        meta={
            "num_blocks": num_blocks,
            "inference_blocks": block_list,
        },
    )
