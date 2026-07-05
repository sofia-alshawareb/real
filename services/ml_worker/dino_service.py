"""DINO inference service with GPU lock and model singleton."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ml.lib.constants import COARSE_FINE_DINO_BLOCK, TALC_EMBEDDING_BLOCK
from ml.lib.dino.inference import extract_multi_block_features
from ml.lib.dino.model import DinoModelCache, resolve_dino_weights, validate_block_indices
from ml.lib.types import DinoArtifacts, DinoInferenceResult
from services.ml_worker.config import DinoConfig


class DinoInferenceService:
    def __init__(self, config: DinoConfig) -> None:
        self.config = config
        self._gpu_lock = threading.Lock()
        self._ready = False
        self._device = torch.device(
            config.device
            if config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._weights = resolve_dino_weights(str(config.weights))
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.num_blocks != 12:
            raise ValueError(
                f"dino.num_blocks must be 12 for block extraction, got {self.config.num_blocks}"
            )
        blocks = [int(b) for b in self.config.inference_blocks]
        if COARSE_FINE_DINO_BLOCK not in blocks:
            raise ValueError(
                f"dino.inference_blocks must include block {COARSE_FINE_DINO_BLOCK}, got {blocks}"
            )
        if TALC_EMBEDDING_BLOCK not in blocks:
            raise ValueError(
                f"dino.inference_blocks must include block {TALC_EMBEDDING_BLOCK} for talc embedding, got {blocks}"
            )
        validate_block_indices(self.config.num_blocks, blocks)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._ready

    def warmup(self) -> None:
        """Load model singleton at startup (R12)."""
        with self._gpu_lock:
            DinoModelCache.get(
                repo_dir=self.config.repo,
                weights=self._weights,
                num_blocks=self.config.num_blocks,
                device=self._device,
            )
            self._ready = True

    def run(
        self,
        rgb: np.ndarray,
        *,
        save_activations: bool | None = None,
    ) -> tuple[DinoInferenceResult, DinoArtifacts | None]:
        """Single forward pass; returns in-memory block 1+11 outputs."""
        persist = (
            self.config.save_activations
            if save_activations is None
            else save_activations
        )

        with self._gpu_lock:
            model = DinoModelCache.get(
                repo_dir=self.config.repo,
                weights=self._weights,
                num_blocks=self.config.num_blocks,
                device=self._device,
            )
            result = extract_multi_block_features(
                rgb,
                device=self._device,
                block_indices=self.config.inference_blocks,
                num_blocks=self.config.num_blocks,
                model=model,
                repo_dir=self.config.repo,
                weights=self._weights,
            )

        weights_hash = ""
        if self._weights and Path(self._weights).exists():
            weights_hash = hashlib.sha256(
                Path(self._weights).read_bytes()
            ).hexdigest()[:16]

        result.meta.update(
            {
                "schema_version": "1.0",
                "model_weights_hash": weights_hash,
                "device": str(self._device),
                "input_scale": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        artifacts = DinoArtifacts.from_inference(result) if persist else None
        return result, artifacts
