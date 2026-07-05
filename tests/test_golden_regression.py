"""Golden regression: library path matches CLI labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.lib.dino.inference import extract_multi_block_features
from ml.lib.dino.model import resolve_dino_weights
from ml.lib.imaging import load_rgb_from_path
from ml.lib.segmentation.pipeline import segment_image
from ml.lib.types import SegmentConfig


@pytest.mark.slow
def test_golden_labels_match_cli(golden_image_path, tmp_path):
    """Compare ml/lib pipeline to CLI output when golden image is available."""
    rgb = load_rgb_from_path(golden_image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = Path("data/models/dinov2")
    weights = resolve_dino_weights("")
    if not repo.exists():
        pytest.skip("DINO repo not available")

    dino = extract_multi_block_features(
        rgb,
        device=device,
        block_indices=[1, 11],
        num_blocks=12,
        repo_dir=repo,
        weights=weights,
    )
    lib_result = segment_image(
        rgb,
        dino.activation(1),
        SegmentConfig(region_map="dino", block_index=1, random_state=0),
        block11_features=dino.features(11),
    )

    assert "defect_detection" in lib_result.metadata
    assert lib_result.metadata["defect_detection"]["method"] == (
        "block11_min_intensity_patch_similarity"
    )
    pytest.skip(
        "CLI still uses intensity GMM for defects; golden label parity disabled "
        "until CLI is updated for block-11 similarity defects"
    )

    import subprocess
    import sys

    out_dir = tmp_path / "cli_out"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        "ml/intensity_gmm_segment.py",
        "--input",
        str(golden_image_path),
        "--output-dir",
        str(out_dir),
        "--no-log-intermediates",
        "--num-blocks",
        "12",
        "--block-index",
        "1",
        "--region-map",
        "dino",
        "--random-state",
        "0",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    stem = golden_image_path.stem
    npy_candidates = list(out_dir.glob(f"{stem}_dino_block01_*labels*.npy"))
    seg_npy = out_dir / f"{stem}_dino_block01_10_result.npy"
    if not seg_npy.exists():
        seg_from_png = out_dir / f"{stem}_dino_block01_10_result.png"
        assert seg_from_png.exists(), "CLI did not produce expected output"
        pytest.skip("CLI does not save labels.npy; compare via re-run only")

    cli_labels = np.load(seg_npy)
    assert np.array_equal(lib_result.labels, cli_labels)
