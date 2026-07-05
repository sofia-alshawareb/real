"""Tests for prepare_calibration script."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.lib.constants import CLASS_COLORS, CLS_COARSE, CLS_TALC
from ml.prepare_calibration import compile_calibration


@pytest.fixture
def mini_calib_tree(tmp_path):
    root = tmp_path / "calib"
    img_dir = root / "img1"
    img_dir.mkdir(parents=True)
    h, w = 64, 64
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    rgb[20:40, 20:40] = 30
    rgb[20:35, 35:40] = 10
    Image.fromarray(rgb).save(img_dir / "normalized.png")

    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[20:40, 20:45] = CLASS_COLORS[CLS_COARSE]
    mask[20:35, 35:40] = CLASS_COLORS[CLS_TALC]
    Image.fromarray(mask).save(img_dir / "user_drawn_colored.png")

    feats = np.random.randn(384, 5, 5).astype(np.float32)
    np.save(img_dir / "block01_features.npy", feats)
    return root


def test_compile_calibration_outputs(mini_calib_tree, tmp_path):
    out = tmp_path / "compiled"
    report = compile_calibration(mini_calib_tree, out, random_state=0)

    assert (out / "summary.json").exists()
    assert (out / "rgb" / "coarse.npy").exists()
    assert (out / "rgb_histograms.npz").exists()
    assert report["counts"]["coarse"] > 0
    assert report["counts"]["talc"] > 0
    assert (out / "prep_report.json").exists()
    prep = json.loads((out / "prep_report.json").read_text(encoding="utf-8"))
    assert "img1" in prep["images"]
    assert (mini_calib_tree / "img1" / "filtered_talc_overlay.png").exists()
    assert (out / "overlays" / "img1_filtered_talc_overlay.png").exists()
