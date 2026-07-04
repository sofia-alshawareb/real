"""Tests for artifact store."""

from __future__ import annotations

import numpy as np
import pytest

from ml.lib.types import DinoArtifacts, SegmentationResult
from services.ml_worker.storage.filesystem import FilesystemArtifactStore


def test_atomic_segmentation_write(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    result = SegmentationResult(
        labels=labels,
        native_width=2,
        native_height=2,
        mask_width=2,
        mask_height=2,
        metadata={"final_class_counts": {"background": 1}},
    )
    store.save_segmentation("abc123", result)
    assert store.segmentation_ready("abc123")
    meta = store.get_mask_metadata("abc123")
    assert meta["native_width"] == 2
    png = store.get_mask_bytes("abc123")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_dino_artifacts_roundtrip(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    artifacts = DinoArtifacts(
        block01_activation=np.ones((4, 4), dtype=np.float32),
        block11_activation=np.zeros((4, 4), dtype=np.float32),
        block01_features=np.ones((8, 2, 2), dtype=np.float32),
        block11_features=np.zeros((8, 2, 2), dtype=np.float32),
        meta={"model_weights_hash": "deadbeef"},
    )
    store.save_dino("img1", artifacts)
    loaded = store.load_dino("img1")
    assert np.allclose(loaded.block01_activation, artifacts.block01_activation)
