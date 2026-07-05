"""Tests for artifact store."""

from __future__ import annotations

import numpy as np
import pytest
import json

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


def test_block01_features_roundtrip(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    feats = np.arange(32, dtype=np.float32).reshape(8, 2, 2)
    act = np.ones((28, 28), dtype=np.float32)
    dino_dir = tmp_path / "img2" / "dino"
    store.write_block01_features(dino_dir, feats, activation=act, meta={"test": True})
    assert store.dino_features_ready("img2")
    loaded = store.load_block01_features("img2")
    assert np.allclose(loaded, feats)
    loaded_act = store.load_block01_activation("img2")
    assert np.allclose(loaded_act, act)


def test_user_drawn_mask_roundtrip(tmp_path):
    store = FilesystemArtifactStore(tmp_path)
    labels = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    manual_dir = store.save_user_drawn_mask("dev1", labels)
    assert store.user_drawn_mask_ready("dev1")
    assert manual_dir.name == "manual"
    loaded = np.load(manual_dir / "user_drawn_labels.npy")
    assert np.array_equal(loaded, labels)
    assert (manual_dir / "user_drawn_colored.png").exists()
    assert (manual_dir / "user_drawn_grayscale.png").exists()
    meta = json.loads((manual_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["encoding"] == "ui_class_index"

