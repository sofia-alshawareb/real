"""GMM segmentation: DINO or gradient regions + intensity-based FG/defect.

1. Intensity = raw gray by default (optional ``--preprocess``: flat-field + denoise).
2. Region map for 4-band GMM: DINO block activation (default) or intensity gradient
   (``--region-map intensity_gradient``).
3. 4-Gaussian GMM on the **region map** → adjacent-intersection thresholds
   → 4 region bands (used for promotion only).
4. Morphological closing on each region mask, then merge.
5. 2-Gaussian GMM on **intensity** → seed ``I ≥ t``. Promoted FG mask =
   seed ∪ region connected components with ≥60% overlap.
6. Split foreground: largest CC = foreground object; other FG CCs = partitions.
7. 2-Gaussian GMM on **intensity** outside foreground → seed ``I < t``. Promoted
   defect mask = seed ∪ region CCs with ≥60% overlap (DINO mode), or in gradient
   mode only whole gradient-region CCs with ≥60% overlap to the seed (seed itself
   is not kept).
8. Final 4-class map + morphological closing per class.

Outputs use a stem prefix reflecting the region map
(``{stem}_dino_blockXX_…`` or ``{stem}_region_intensity_gradient_…``).

With ``--log-intermediates``, also ``{stem}_…_intermediates/`` with numbered steps
and optional ``probes/`` subfolder.

Optional probe maps (``--log-probes``, default on when intermediates are logged)
under ``…_intermediates/probes/`` — alternative intensity spaces, texture /
structure metrics, and DINO feature views for future algorithm experiments.
These do **not** change segmentation output.

Usage:
    python ml/intensity_gmm_segment.py

    python ml/intensity_gmm_segment.py \\
        --source "data/raw/task3/Фото руд по сортам. ч1" \\
        --output-dir "outputs/intensity_gmm_segment" \\
        --log-intermediates
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Must be set before NumPy/SciPy/sklearn load BLAS, otherwise parallel GMM
# fits can oversubscribe or deadlock inside OpenBLAS/MKL.
for _env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env_key] = "1"

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import brentq
from scipy.stats import norm
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters import rank, sobel
from skimage.morphology import closing, disk
from skimage.restoration import denoise_bilateral
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
_DATA_RAW_TASK3 = (
    _ROOT / "data" / "raw" / "task3"
    if (_ROOT / "data" / "raw" / "task3").is_dir()
    and any((_ROOT / "data" / "raw" / "task3").iterdir())
    else _ROOT / "task3-data"
)
DEFAULT_INPUT = (
    _DATA_RAW_TASK3
    / "Фото руд по сортам. ч1/Оталькованные руды/2550374-2 10х.JPG"
)
DEFAULT_OUTPUT_DIR = _ROOT / "outputs/intensity_gmm_segment"
DEFAULT_SOURCE = _DATA_RAW_TASK3 / "Фото руд по сортам. ч1"
DEFAULT_DINO_REPO = _ROOT / "data/models/dinov2"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
DEFAULT_DINO_WEIGHTS = _ROOT / "data/models/checkpoints/dinov2_vits14_reg4_pretrain.pth"
FALLBACK_DINO_WEIGHTS = (
    Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth"
)
PATCH_SIZE = 14
DEFAULT_DINO_BLOCK_INDEX = 1
DEFAULT_DINO_NUM_BLOCKS = 2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

N_GAUSSIANS_REGIONS = 4
N_GAUSSIANS_BINARY = 2
DEFAULT_REGION_OVERLAP = 0.60
REGION_MAP_DINO = "dino"
REGION_MAP_INTENSITY_GRADIENT = "intensity_gradient"
VALID_REGION_MAPS = (REGION_MAP_DINO, REGION_MAP_INTENSITY_GRADIENT)

CLS_BACKGROUND = 0
CLS_FOREGROUND = 1
CLS_PARTITIONS = 2
CLS_DEFECT = 3

CLASS_NAMES = {
    CLS_BACKGROUND: "background",
    CLS_FOREGROUND: "foreground_object",
    CLS_PARTITIONS: "partitions",
    CLS_DEFECT: "defect",
}

CLASS_COLORS = {
    CLS_BACKGROUND: (40, 40, 40),
    CLS_FOREGROUND: (220, 50, 50),
    CLS_PARTITIONS: (50, 140, 255),
    CLS_DEFECT: (240, 200, 40),
}

# Region bands from 4-Gaussian GMM on DINO activation (before promotion).
REGION_NAMES = {
    0: "dino_region_0",
    1: "dino_region_1",
    2: "dino_region_2",
    3: "dino_region_3",
}

REGION_COLORS = {
    0: (30, 30, 30),
    1: (70, 130, 180),
    2: (60, 179, 113),
    3: (220, 20, 60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DINO block activation region GMM + intensity-based foreground/defect promotion."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Single input image path (ignored when --source is set).",
    )
    parser.add_argument(
        "--source",
        default="",
        help=(
            "Root folder of images (searched recursively). When set, processes every "
            "image and mirrors the folder structure under --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--log-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save intermediate artifacts under {stem}_dino_blockXX_intermediates/, "
            "including GMM histograms with fitted Gaussians and thresholds "
            "(default: true). Use --no-log-intermediates to disable."
        ),
    )
    parser.add_argument(
        "--log-probes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When intermediates are logged, also save probe maps under "
            "…/probes/ (alternative intensity/texture/DINO views; default: true). "
            "Use --no-log-probes to disable."
        ),
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=256,
        help="Histogram bins for logged GMM plots (default: 256).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip images whose result PNG already exists (default: true).",
    )
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flat-field (+ optional denoise) before GMM (default: false).",
    )
    parser.add_argument(
        "--denoise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bilateral denoise after flat-field when --preprocess is on (default: true).",
    )
    parser.add_argument(
        "--illum-sigma",
        type=float,
        default=64.0,
        help="Gaussian sigma for flat-field illumination estimate (default: 64).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300000,
        help="Max pixels used to fit each GMM (default: 300000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="GMM random seed (default: 0).",
    )
    parser.add_argument(
        "--region-overlap",
        type=float,
        default=DEFAULT_REGION_OVERLAP,
        help=(
            "Min fraction of each DINO-region *connected component* that must lie "
            f"inside the intensity seed to promote that component "
            f"(default: {DEFAULT_REGION_OVERLAP})."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(2, min(8, os.cpu_count() or 4)),
        help="Thread-pool size for parallel steps (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--close-radius",
        type=int,
        default=3,
        help=(
            "Disk radius for morphological closing applied per class to fill small "
            "gaps (default: 3; 0 disables)."
        ),
    )
    parser.add_argument(
        "--region-map",
        choices=VALID_REGION_MAPS,
        default=REGION_MAP_DINO,
        help=(
            "Map used for 4-band region GMM and CC promotion: "
            f"'{REGION_MAP_DINO}' (DINO block activation, default) or "
            f"'{REGION_MAP_INTENSITY_GRADIENT}' (Sobel gradient of intensity)."
        ),
    )
    parser.add_argument(
        "--dino-repo",
        default=str(DEFAULT_DINO_REPO),
        help="Local DINOv2 repo for torch.hub.load(source='local').",
    )
    parser.add_argument(
        "--dino-weights",
        default="",
        help="DINOv2 checkpoint path (default: project checkpoint, then hub cache).",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=DEFAULT_DINO_NUM_BLOCKS,
        help=(
            "Number of ViT blocks to keep in the truncated DINO model "
            f"(default: {DEFAULT_DINO_NUM_BLOCKS})."
        ),
    )
    parser.add_argument(
        "--block-index",
        type=int,
        default=DEFAULT_DINO_BLOCK_INDEX,
        help=(
            "Transformer block index for the activation map "
            f"(default: {DEFAULT_DINO_BLOCK_INDEX}; must be < --num-blocks)."
        ),
    )
    parser.add_argument(
        "--device",
        default="",
        help="Torch device for DINO (default: cuda if available else cpu).",
    )
    return parser.parse_args()


def resolve_dino_weights(path: str) -> str:
    if path:
        return path
    if DEFAULT_DINO_WEIGHTS.exists():
        return str(DEFAULT_DINO_WEIGHTS)
    if FALLBACK_DINO_WEIGHTS.exists():
        return str(FALLBACK_DINO_WEIGHTS)
    return ""


def validate_dino_block_args(num_blocks: int, block_index: int) -> None:
    if num_blocks < 1:
        raise ValueError(f"--num-blocks must be >= 1, got {num_blocks}.")
    if block_index < 0 or block_index >= num_blocks:
        raise ValueError(
            f"--block-index must be in [0, {num_blocks - 1}], got {block_index}."
        )


def dino_block_tag(block_index: int) -> str:
    return f"block{block_index:02d}"


def dino_output_stem(stem: str, block_index: int) -> str:
    """Shared prefix for outputs tied to a DINO block index."""
    return f"{stem}_dino_{dino_block_tag(block_index)}"


def run_output_stem(stem: str, region_map: str, block_index: int) -> str:
    if region_map == REGION_MAP_INTENSITY_GRADIENT:
        return f"{stem}_region_{REGION_MAP_INTENSITY_GRADIENT}"
    return dino_output_stem(stem, block_index)


def needs_dino_inference(args: argparse.Namespace) -> bool:
    if args.region_map == REGION_MAP_DINO:
        return True
    return bool(args.log_intermediates and args.log_probes)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_gray01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def build_intensity(
    gray: np.ndarray,
    *,
    preprocess: bool,
    illum_sigma: float,
    denoise: bool,
) -> np.ndarray:
    if not preprocess:
        return gray.astype(np.float32)
    illumination = ndi.gaussian_filter(gray, sigma=illum_sigma)
    illumination = np.maximum(illumination, 1e-6)
    corrected = gray / illumination
    corrected = corrected / max(float(np.percentile(corrected, 99.5)), 1e-6)
    corrected = np.clip(corrected, 0.0, 1.0).astype(np.float32)
    if not denoise:
        return corrected
    return denoise_bilateral(
        corrected,
        sigma_color=0.05,
        sigma_spatial=2,
        channel_axis=None,
    ).astype(np.float32)


def _pad_rgb_to_patch_multiple(rgb: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    h, w = rgb.shape[:2]
    pad_h = math.ceil(h / patch_size) * patch_size - h
    pad_w = math.ceil(w / patch_size) * patch_size - w
    if pad_h == 0 and pad_w == 0:
        return rgb
    return np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def rgb_to_dino_tensor(rgb: np.ndarray) -> torch.Tensor:
    padded = _pad_rgb_to_patch_multiple(rgb)
    arr = padded.astype(np.float32) / 255.0
    return torch.from_numpy((arr - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1).unsqueeze(0)


def prune_vit_blocks(model: nn.Module, num_blocks: int) -> nn.Module:
    model.blocks = nn.ModuleList(list(model.blocks[:num_blocks]))
    if hasattr(model, "n_blocks"):
        model.n_blocks = num_blocks
    return model


def load_dinov2_truncated(
    repo_dir: Path,
    weights: str,
    device: torch.device,
    num_blocks: int,
) -> nn.Module:
    kwargs: dict = {"pretrained": True}
    if weights:
        kwargs["weights"] = weights
    model = torch.hub.load(str(repo_dir), "dinov2_vits14_reg", source="local", **kwargs)
    prune_vit_blocks(model, num_blocks)
    return model.to(device).eval()


@torch.inference_mode()
def dino_block_patch_features(
    rgb: np.ndarray,
    *,
    device: torch.device,
    block_index: int,
    num_blocks: int,
    model: nn.Module | None = None,
    repo_dir: Path | None = None,
    weights: str = "",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Return normalized DINO patch tokens (C, hp, wp) on CPU and image (h, w)."""
    owns_model = model is None
    if owns_model:
        assert repo_dir is not None
        model = load_dinov2_truncated(repo_dir, weights, device, num_blocks)
    assert model is not None

    h, w = rgb.shape[:2]
    image = rgb_to_dino_tensor(rgb).to(device)
    outputs = model.get_intermediate_layers(
        image,
        n=[block_index],
        reshape=True,
        norm=True,
    )
    feats = outputs[0][0].detach().float().cpu()
    del image, outputs
    if owns_model:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return feats, (h, w)


def features_to_activation(
    feats: torch.Tensor,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Full-resolution min-max L2 activation from patch-token grid."""
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
    """Upsample a (hp, wp) map to full resolution."""
    tensor = torch.from_numpy(patch_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return up.squeeze().numpy().astype(np.float32)


@torch.inference_mode()
def dino_block_activation_map(
    rgb: np.ndarray,
    *,
    device: torch.device,
    block_index: int,
    num_blocks: int,
    model: nn.Module | None = None,
    repo_dir: Path | None = None,
    weights: str = "",
) -> np.ndarray:
    """Full-resolution min-max L2 activation from the chosen DINO block."""
    feats, target_hw = dino_block_patch_features(
        rgb,
        device=device,
        block_index=block_index,
        num_blocks=num_blocks,
        model=model,
        repo_dir=repo_dir,
        weights=weights,
    )
    return features_to_activation(feats, target_hw)


def normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi > lo:
        return ((arr - lo) / (hi - lo)).astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)


def intensity_gradient_map(intensity: np.ndarray) -> np.ndarray:
    """Normalized Sobel gradient magnitude in [0, 1]."""
    return normalize01(sobel(intensity))


def map_stats(arr: np.ndarray) -> dict[str, float]:
    flat = arr.ravel()
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
    }


def compute_intensity_probes(
    intensity: np.ndarray,
    gray: np.ndarray,
    rgb: np.ndarray,
    *,
    illum_sigma: float,
    local_window: int = 15,
    entropy_radius: int = 7,
) -> dict[str, np.ndarray]:
    """Alternative intensity / texture spaces (full resolution, float32 in [0, 1])."""
    probes: dict[str, np.ndarray] = {}

    probes["11_intensity_log"] = normalize01(-np.log(intensity + 1e-6))
    probes["12_intensity_gradient"] = intensity_gradient_map(intensity)
    probes["13_intensity_local_std"] = normalize01(
        ndi.generic_filter(intensity, np.std, size=local_window)
    )
    probes["14_intensity_laplacian"] = normalize01(np.abs(ndi.laplace(intensity)))
    probes["15_intensity_local_entropy"] = normalize01(
        rank.entropy((intensity * 255.0).astype(np.uint8), disk(entropy_radius))
        / np.log2(256)
    )
    probes["16_intensity_flatfield"] = normalize01(
        build_intensity(
            gray,
            preprocess=True,
            illum_sigma=illum_sigma,
            denoise=False,
        )
    )

    st_elems = structure_tensor(intensity, sigma=1.0)
    l1, l2 = structure_tensor_eigenvalues(st_elems)
    probes["17_structure_coherence"] = normalize01((l1 - l2) / (l1 + l2 + 1e-8))

    rgb_f = rgb.astype(np.float32) / 255.0
    mx = rgb_f.max(axis=2)
    mn = rgb_f.min(axis=2)
    saturation = np.where(mx > 1e-6, (mx - mn) / mx, 0.0)
    probes["18_rgb_saturation"] = normalize01(saturation)

    return probes


def compute_dino_probes(
    patch_feats: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    random_state: int,
    max_pca_samples: int = 50000,
) -> dict[str, np.ndarray]:
    """DINO patch-feature views upsampled to full resolution."""
    c, hp, wp = patch_feats.shape
    vectors = patch_feats.reshape(c, -1).T.numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-8)

    probes: dict[str, np.ndarray] = {}

    feat_std = unit.std(axis=1).reshape(hp, wp)
    probes["19_dino_feature_std"] = normalize01(upsample_patch_map(feat_std, target_hw))

    mean_vec = unit.mean(axis=0)
    mean_vec /= max(float(np.linalg.norm(mean_vec)), 1e-8)
    cosine = unit @ mean_vec
    anomaly = (1.0 - cosine).reshape(hp, wp)
    probes["20_dino_cosine_anomaly"] = normalize01(upsample_patch_map(anomaly, target_hw))

    n_patches = vectors.shape[0]
    if n_patches > max_pca_samples:
        rng = np.random.default_rng(random_state)
        fit_idx = rng.choice(n_patches, size=max_pca_samples, replace=False)
        fit_vectors = unit[fit_idx]
    else:
        fit_vectors = unit
    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(fit_vectors)
    projected = pca.transform(unit).reshape(hp, wp, 3)
    pca_rgb = np.zeros(target_hw + (3,), dtype=np.uint8)
    for ch in range(3):
        ch_up = upsample_patch_map(projected[:, :, ch], target_hw)
        pca_rgb[:, :, ch] = (normalize01(ch_up) * 255.0).astype(np.uint8)
    probes["21_dino_pca_rgb"] = pca_rgb

    probes["_dino_pca_explained_variance"] = pca.explained_variance_ratio_.astype(np.float32)
    return probes


def region_probe_stats(
    region_labels: np.ndarray,
    probe_maps: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for region_id in sorted(int(r) for r in np.unique(region_labels)):
        mask = region_labels == region_id
        if not np.any(mask):
            continue
        region_key = REGION_NAMES.get(region_id, f"region_{region_id}")
        stats[region_key] = {}
        for name, arr in probe_maps.items():
            if name.startswith("_") or arr.ndim != 2:
                continue
            stats[region_key][name] = map_stats(arr[mask])
    return stats


def save_probe_maps(
    probe_dir: Path,
    probe_maps: dict[str, np.ndarray],
    *,
    workers: int,
) -> list[str]:
    probe_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def _save_one(name: str, arr: np.ndarray) -> str:
        if name.startswith("_"):
            return ""
        if arr.ndim == 2:
            Image.fromarray(gray01_to_rgb(arr)).save(probe_dir / f"{name}.png")
            np.save(probe_dir / f"{name}.npy", arr.astype(np.float32))
        elif arr.ndim == 3 and arr.shape[2] == 3:
            Image.fromarray(arr.astype(np.uint8)).save(probe_dir / f"{name}.png")
            np.save(probe_dir / f"{name}.npy", arr)
        return name

    scalar_maps = {k: v for k, v in probe_maps.items() if not k.startswith("_")}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        names = list(scalar_maps.keys())
        results = list(pool.map(lambda item: _save_one(item[0], item[1]), scalar_maps.items()))
    saved = [name for name in results if name]
    return saved


def save_probe_overview(
    path: Path,
    intensity: np.ndarray,
    probe_maps: dict[str, np.ndarray],
    max_panels: int = 6,
) -> None:
    """Save a quick-look row: intensity + selected scalar probes."""
    scalar_names = sorted(
        name for name, arr in probe_maps.items() if not name.startswith("_") and arr.ndim == 2
    )[:max_panels]
    panels = [gray01_to_rgb(intensity)] + [gray01_to_rgb(probe_maps[n]) for n in scalar_names]
    stack = stack_horizontal(panels)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stack).save(path)


def compute_and_save_probes(
    log_dir: Path,
    *,
    intensity: np.ndarray,
    gray: np.ndarray,
    rgb: np.ndarray,
    patch_feats: torch.Tensor,
    region_labels: np.ndarray,
    illum_sigma: float,
    random_state: int,
    workers: int,
) -> dict[str, object]:
    """Compute probe maps and persist them under log_dir/probes/."""
    probe_dir = log_dir / "probes"
    target_hw = intensity.shape[:2]

    intensity_probes = compute_intensity_probes(
        intensity,
        gray,
        rgb,
        illum_sigma=illum_sigma,
    )
    dino_probes = compute_dino_probes(
        patch_feats,
        target_hw,
        random_state=random_state,
    )

    pca_var = dino_probes.pop("_dino_pca_explained_variance", None)
    probe_maps: dict[str, np.ndarray] = {}
    probe_maps.update(intensity_probes)
    probe_maps.update(dino_probes)

    saved = save_probe_maps(probe_dir, probe_maps, workers=workers)
    overview_path = probe_dir / "22_probe_overview.png"
    save_probe_overview(overview_path, intensity, probe_maps)

    region_stats = region_probe_stats(region_labels, probe_maps)
    probe_meta: dict[str, object] = {
        "description": (
            "Exploratory probe maps for future segmentation experiments. "
            "Not used by the current pipeline."
        ),
        "probes": {
            "11_intensity_log": "Negative log intensity; emphasizes darker structures.",
            "12_intensity_gradient": "Sobel gradient magnitude; edges and boundaries.",
            "13_intensity_local_std": "Local intensity std dev; texture heterogeneity.",
            "14_intensity_laplacian": "Laplacian magnitude; fine-scale patterns.",
            "15_intensity_local_entropy": "Rank local entropy; complex micro-texture.",
            "16_intensity_flatfield": "Flat-field corrected intensity (reference space).",
            "17_structure_coherence": "Structure-tensor coherence; oriented texture strength.",
            "18_rgb_saturation": "HSV saturation from RGB; color contrast cue.",
            "19_dino_feature_std": "Std dev across DINO embedding dims per patch.",
            "20_dino_cosine_anomaly": "1 - cosine similarity to mean DINO patch vector.",
            "21_dino_pca_rgb": "First 3 PCA components of DINO patches (RGB view).",
            "22_probe_overview": "Intensity + subset of scalar probes (PNG only).",
        },
        "saved": saved + ["22_probe_overview.png"],
        "map_stats": {name: map_stats(arr) for name, arr in probe_maps.items() if arr.ndim == 2},
        "region_stats": region_stats,
    }
    if pca_var is not None:
        probe_meta["dino_pca_explained_variance_ratio"] = pca_var.tolist()

    probe_summary_path = probe_dir / "00_probes_summary.json"
    region_stats_path = probe_dir / "00_region_probe_stats.json"
    probe_summary_path.write_text(json.dumps(probe_meta, indent=2), encoding="utf-8")
    region_stats_path.write_text(json.dumps(region_stats, indent=2), encoding="utf-8")
    return {
        "probe_dir": "probes",
        "probe_summary": probe_summary_path.name,
        "region_probe_stats": region_stats_path.name,
        "probe_count": len(saved),
    }


def activation_to_rgb(activation: np.ndarray) -> np.ndarray:
    gray = (np.clip(activation, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def gray01_to_rgb(gray01: np.ndarray) -> np.ndarray:
    gray = (np.clip(gray01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def binary_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """White = True/1, black = False/0."""
    on = mask.astype(bool)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[on] = (255, 255, 255)
    return rgb


def seed_overlay_rgb(gray01: np.ndarray, seed_mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Intensity as gray with seed pixels tinted."""
    base = gray01_to_rgb(gray01).astype(np.float32)
    on = seed_mask.astype(bool)
    tint = np.array(color, dtype=np.float32)
    base[on] = 0.45 * base[on] + 0.55 * tint
    return np.clip(base, 0, 255).astype(np.uint8)


def fit_gmm(
    value_map: np.ndarray,
    n_components: int,
    max_samples: int,
    random_state: int,
    pixel_mask: np.ndarray | None = None,
) -> GaussianMixture:
    if pixel_mask is not None:
        values = value_map[pixel_mask.astype(bool)].astype(np.float64).ravel()
    else:
        values = value_map.ravel().astype(np.float64)
    if values.size < n_components:
        raise ValueError(f"Need ≥{n_components} pixels for GMM, got {values.size}.")
    if values.size > max_samples:
        rng = np.random.default_rng(random_state)
        values = values[rng.choice(values.size, size=max_samples, replace=False)]
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
        max_iter=300,
    )
    gmm.fit(values.reshape(-1, 1))
    return gmm


def ordered_gmm_params(gmm: GaussianMixture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = gmm.means_.ravel()
    order = np.argsort(means)
    means = means[order]
    weights = gmm.weights_[order]
    variances = np.array([float(gmm.covariances_[i].ravel()[0]) for i in order])
    variances = np.maximum(variances, 1e-12)
    return means, variances, weights


def _component_pdf(x: np.ndarray, weight: float, mean: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    return weight * norm.pdf(x, loc=mean, scale=std)


def adjacent_intersection(
    mean_lo: float,
    var_lo: float,
    weight_lo: float,
    mean_hi: float,
    var_hi: float,
    weight_hi: float,
) -> float:
    lo, hi = float(mean_lo), float(mean_hi)
    if hi - lo < 1e-6:
        return lo
    std_lo = float(np.sqrt(max(var_lo, 1e-12)))
    std_hi = float(np.sqrt(max(var_hi, 1e-12)))

    def diff(x: float) -> float:
        xv = np.atleast_1d(x)
        return float(
            _component_pdf(xv, weight_lo, mean_lo, std_lo)[0]
            - _component_pdf(xv, weight_hi, mean_hi, std_hi)[0]
        )

    xs = np.linspace(lo, hi, 400)
    vals = np.array([diff(float(x)) for x in xs])
    sign_changes = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if len(sign_changes) > 0:
        i = int(sign_changes[0])
        return float(brentq(diff, xs[i], xs[i + 1]))
    return float(xs[int(np.argmin(np.abs(vals)))])


def thresholds_from_adjacent_intersections(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    thresholds = [
        adjacent_intersection(
            means[i],
            variances[i],
            weights[i],
            means[i + 1],
            variances[i + 1],
            weights[i + 1],
        )
        for i in range(len(means) - 1)
    ]
    thresholds = np.asarray(thresholds, dtype=np.float64)
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = min(
                0.999999,
                thresholds[i - 1] + max(1e-4, 0.25 * (means[i + 1] - means[i])),
            )
    return thresholds


def segment_by_thresholds(value_map: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    labels = np.zeros(value_map.shape, dtype=np.int32)
    for i, t in enumerate(thresholds):
        labels[value_map >= t] = i + 1
    return labels


def morphological_close_label_map(
    labels: np.ndarray,
    radius: int,
    class_ids: list[int] | None = None,
) -> np.ndarray:
    """Morphological closing per class to fill small gaps between its components.

    For each class C, compute ``closing(C)`` and adopt foreign pixels that lie
    inside that closed mask and within ``radius`` of the original C. If several
    classes want the same pixel, the nearest original class wins.
    """
    if radius <= 0:
        return labels.astype(np.int32, copy=True)

    labels = labels.astype(np.int32, copy=False)
    ids = class_ids if class_ids is not None else [int(c) for c in np.unique(labels)]
    footprint = disk(int(radius))
    radius = int(radius)

    steal_dist = np.full((len(ids),) + labels.shape, np.inf, dtype=np.float64)
    for i, cls_id in enumerate(ids):
        mask = labels == cls_id
        if not np.any(mask):
            continue
        closed = closing(mask.astype(np.uint8), footprint=footprint).astype(bool)
        dist_to_c = ndi.distance_transform_edt(~mask)
        steal = closed & (labels != cls_id) & (dist_to_c <= radius)
        steal_dist[i] = np.where(steal, dist_to_c, np.inf)

    out = labels.copy()
    any_steal = np.isfinite(steal_dist).any(axis=0)
    if np.any(any_steal):
        best = np.argmin(steal_dist, axis=0)
        for i, cls_id in enumerate(ids):
            out[any_steal & (best == i)] = cls_id
    return out


def two_gmm_threshold(
    intensity: np.ndarray,
    *,
    max_samples: int,
    random_state: int,
    pixel_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    gmm = fit_gmm(
        intensity,
        N_GAUSSIANS_BINARY,
        max_samples,
        random_state,
        pixel_mask=pixel_mask,
    )
    means, variances, weights = ordered_gmm_params(gmm)
    threshold = adjacent_intersection(
        means[0],
        variances[0],
        weights[0],
        means[1],
        variances[1],
        weights[1],
    )
    return threshold, means, variances, weights


def promote_regions_by_overlap(
    region_labels: np.ndarray,
    seed_mask: np.ndarray,
    overlap_threshold: float,
    n_regions: int = N_GAUSSIANS_REGIONS,
) -> np.ndarray:
    """Grow the intensity seed by DINO connected components that overlap it enough.

    Starts from the seed mask (never removes seed pixels). For every connected
    component of every DINO band label, if ``|CC ∩ seed| / |CC|`` ≥
    ``overlap_threshold``, the **whole** component is added (fills in the parts
    of that blob that lie outside the seed).
    """
    # Keep the full intensity seed; only *add* qualifying DINO components.
    final = (seed_mask > 0).astype(np.uint8)
    seed_bool = final.astype(bool)
    seed_weights = seed_bool.astype(np.float64).ravel()

    for region_id in range(n_regions):
        band = region_labels == region_id
        if not np.any(band):
            continue
        labeled, n_comp = ndi.label(band)
        if n_comp == 0:
            continue
        flat = labeled.ravel()
        sizes = np.bincount(flat)
        overlap_counts = np.bincount(
            flat, weights=seed_weights, minlength=sizes.shape[0]
        )
        selected_ids = [
            comp_id
            for comp_id in range(1, n_comp + 1)
            if sizes[comp_id] > 0
            and (overlap_counts[comp_id] / float(sizes[comp_id])) >= overlap_threshold
        ]
        if not selected_ids:
            continue
        keep = np.isin(labeled, np.asarray(selected_ids, dtype=labeled.dtype))
        final[keep] = 1
    return final


def select_regions_by_overlap(
    region_labels: np.ndarray,
    reference_mask: np.ndarray,
    overlap_threshold: float,
    n_regions: int = N_GAUSSIANS_REGIONS,
) -> np.ndarray:
    """Keep only region CCs that overlap a reference mask enough.

    Unlike ``promote_regions_by_overlap``, the reference mask itself is **not**
    kept. For every connected component of every region band, if
    ``|CC ∩ reference| / |CC|`` ≥ ``overlap_threshold``, the **whole** CC is
    selected.
    """
    final = np.zeros(region_labels.shape, dtype=np.uint8)
    ref_bool = reference_mask.astype(bool)
    ref_weights = ref_bool.astype(np.float64).ravel()

    for region_id in range(n_regions):
        band = region_labels == region_id
        if not np.any(band):
            continue
        labeled, n_comp = ndi.label(band)
        if n_comp == 0:
            continue
        flat = labeled.ravel()
        sizes = np.bincount(flat)
        overlap_counts = np.bincount(
            flat, weights=ref_weights, minlength=sizes.shape[0]
        )
        selected_ids = [
            comp_id
            for comp_id in range(1, n_comp + 1)
            if sizes[comp_id] > 0
            and (overlap_counts[comp_id] / float(sizes[comp_id])) >= overlap_threshold
        ]
        if not selected_ids:
            continue
        keep = np.isin(labeled, np.asarray(selected_ids, dtype=labeled.dtype))
        final[keep] = 1
    return final


def split_foreground_object_and_partitions(
    foreground_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest CC → foreground object; remaining FG CCs → partitions."""
    fg = foreground_mask.astype(bool)
    if not np.any(fg):
        z = np.zeros(fg.shape, dtype=np.uint8)
        return z, z

    labeled, n_comp = ndi.label(fg)
    if n_comp == 0:
        z = np.zeros(fg.shape, dtype=np.uint8)
        return z, z

    sizes = ndi.sum(fg, labeled, index=np.arange(1, n_comp + 1))
    largest_id = int(np.argmax(sizes)) + 1
    fg_object = (labeled == largest_id).astype(np.uint8)
    partitions = (fg & (labeled != largest_id)).astype(np.uint8)
    return fg_object, partitions


def build_final_segmentation(
    fg_object: np.ndarray,
    partitions: np.ndarray,
    defect: np.ndarray,
) -> np.ndarray:
    """Priority: foreground object > partitions > defect > background."""
    seg = np.full(fg_object.shape, CLS_BACKGROUND, dtype=np.uint8)
    seg[defect.astype(bool)] = CLS_DEFECT
    seg[partitions.astype(bool)] = CLS_PARTITIONS
    seg[fg_object.astype(bool)] = CLS_FOREGROUND
    return seg


def labels_to_rgb(labels: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for cls_id, color in colors.items():
        rgb[labels == cls_id] = color
    return rgb


def segmentation_to_rgb(seg: np.ndarray) -> np.ndarray:
    return labels_to_rgb(seg, CLASS_COLORS)


def stack_horizontal(panels: list[np.ndarray]) -> np.ndarray:
    heights = [p.shape[0] for p in panels]
    widths = [p.shape[1] for p in panels]
    h = max(heights)
    out = np.zeros((h, sum(widths), 3), dtype=np.uint8)
    x0 = 0
    for panel in panels:
        ph, pw = panel.shape[:2]
        out[:ph, x0 : x0 + pw] = panel
        x0 += pw
    return out


def add_legend_bar(
    rgb: np.ndarray,
    names: dict[int, str],
    colors: dict[int, tuple[int, int, int]],
    order: list[int] | None = None,
) -> np.ndarray:
    """Append a small legend strip under the stack."""
    bar_h = 36
    h, w = rgb.shape[:2]
    bar = np.full((bar_h, w, 3), 30, dtype=np.uint8)
    canvas = np.vstack([rgb, bar])
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    x = 10
    ids = order if order is not None else list(names.keys())
    for cls_id in ids:
        color = colors[cls_id]
        name = names[cls_id]
        draw.rectangle([x, h + 8, x + 18, h + 26], fill=color)
        draw.text((x + 24, h + 8), name, fill=(240, 240, 240))
        x += 18 + 8 + max(70, 8 * len(name)) + 20
    return np.asarray(img)


def _fit_region_gmm(activation: np.ndarray, max_samples: int, random_state: int):
    """4-GMM on DINO block-1 activation (not intensity)."""
    gmm = fit_gmm(activation, N_GAUSSIANS_REGIONS, max_samples, random_state)
    return ordered_gmm_params(gmm)


def _fit_fg_gmm(intensity: np.ndarray, max_samples: int, random_state: int):
    return two_gmm_threshold(
        intensity,
        max_samples=max_samples,
        random_state=random_state,
    )


def discover_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def save_histogram_with_gmm(
    path: Path,
    values_map: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
    *,
    title: str,
    bins: int = 256,
    sample_mask: np.ndarray | None = None,
    value_label: str = "Value",
) -> None:
    if sample_mask is not None:
        values = values_map[sample_mask.astype(bool)].ravel()
    else:
        values = values_map.ravel()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        values,
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        color="0.75",
        edgecolor="none",
        label="Histogram",
    )
    xs = np.linspace(0.0, 1.0, 1000)
    mixture = np.zeros_like(xs)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (mu, var, w) in enumerate(zip(means, variances, weights)):
        pdf = w * norm.pdf(xs, loc=mu, scale=np.sqrt(var))
        mixture += pdf
        ax.plot(xs, pdf, color=colors[i % len(colors)], lw=2, label=f"G{i} μ={mu:.3f}")
    ax.plot(xs, mixture, color="black", lw=2.0, linestyle="--", label="Mixture")
    for i, t in enumerate(np.atleast_1d(thresholds)):
        ax.axvline(
            float(t),
            color="crimson",
            linestyle=":",
            lw=1.8,
            label="Thresholds" if i == 0 else None,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(value_label)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_one(
    input_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    device: torch.device,
    dino_model: nn.Module | None = None,
    dino_weights: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    log = bool(args.log_intermediates)
    stem = input_path.stem
    block_index = int(args.block_index)
    block_tag = dino_block_tag(block_index)
    region_map = str(args.region_map)
    output_stem = run_output_stem(stem, region_map, block_index)
    result_path = out_dir / f"{output_stem}_10_result.png"
    if args.skip_existing and result_path.exists():
        print(f"Skip existing: {result_path}")
        return result_path

    t0 = time.perf_counter()

    rgb_input = load_rgb(input_path)
    gray = load_gray01(input_path)
    intensity = build_intensity(
        gray,
        preprocess=args.preprocess,
        illum_sigma=args.illum_sigma,
        denoise=args.denoise,
    )

    patch_feats: torch.Tensor | None = None
    if region_map == REGION_MAP_INTENSITY_GRADIENT:
        region_map_values = intensity_gradient_map(intensity)
        region_map_name = "intensity gradient"
    else:
        patch_feats, target_hw = dino_block_patch_features(
            rgb_input,
            device=device,
            block_index=block_index,
            num_blocks=int(args.num_blocks),
            model=dino_model,
            repo_dir=Path(args.dino_repo),
            weights=dino_weights,
        )
        region_map_values = features_to_activation(patch_feats, target_hw)
        region_map_name = f"DINO {block_tag}"

    if patch_feats is None and log and args.log_probes:
        patch_feats, _ = dino_block_patch_features(
            rgb_input,
            device=device,
            block_index=block_index,
            num_blocks=int(args.num_blocks),
            model=dino_model,
            repo_dir=Path(args.dino_repo),
            weights=dino_weights,
        )

    # Parallel: 4-GMM on region map + 2-GMM FG on intensity.
    ctx = get_context("fork")
    with ctx.Pool(processes=min(2, workers)) as pool:
        async_gmm4 = pool.apply_async(
            _fit_region_gmm,
            (region_map_values, args.max_samples, args.random_state),
        )
        async_fg = pool.apply_async(
            _fit_fg_gmm,
            (intensity, args.max_samples, args.random_state),
        )
        means, variances, weights = async_gmm4.get()
        fg_t, fg_means, fg_vars, fg_weights = async_fg.get()

    thresholds = thresholds_from_adjacent_intersections(means, variances, weights)
    region_labels_raw = segment_by_thresholds(region_map_values, thresholds)
    region_labels = morphological_close_label_map(
        region_labels_raw,
        radius=args.close_radius,
        class_ids=list(REGION_NAMES.keys()),
    )

    fg_seed = (intensity >= fg_t).astype(np.uint8)
    fg_mask = promote_regions_by_overlap(
        region_labels,
        fg_seed,
        args.region_overlap,
    )
    fg_object, partitions = split_foreground_object_and_partitions(fg_mask)

    non_fg = fg_mask == 0
    defect_t, def_means, def_vars, def_weights = two_gmm_threshold(
        intensity,
        max_samples=args.max_samples,
        random_state=args.random_state,
        pixel_mask=non_fg,
    )
    defect_seed = (intensity < defect_t).astype(np.uint8)
    if region_map == REGION_MAP_INTENSITY_GRADIENT:
        defect_mask = select_regions_by_overlap(
            region_labels,
            defect_seed,
            args.region_overlap,
        )
        defect_match_mode = "gradient_region_select"
    else:
        defect_mask = promote_regions_by_overlap(
            region_labels,
            defect_seed,
            args.region_overlap,
        )
        defect_match_mode = "seed_plus_region_promote"

    seg_raw = build_final_segmentation(fg_object, partitions, defect_mask)
    seg = morphological_close_label_map(
        seg_raw,
        radius=args.close_radius,
        class_ids=list(CLASS_NAMES.keys()),
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_seg_rgb = pool.submit(segmentation_to_rgb, seg)
        fut_region_rgb = pool.submit(labels_to_rgb, region_labels, REGION_COLORS)
        fut_act_rgb = pool.submit(activation_to_rgb, region_map_values)
        seg_rgb = fut_seg_rgb.result()
        region_rgb = fut_region_rgb.result()
        act_rgb = fut_act_rgb.result()

    stack = add_legend_bar(
        stack_horizontal([rgb_input, seg_rgb]),
        CLASS_NAMES,
        CLASS_COLORS,
        order=[CLS_FOREGROUND, CLS_PARTITIONS, CLS_DEFECT, CLS_BACKGROUND],
    )
    region_stack = add_legend_bar(
        stack_horizontal([rgb_input, act_rgb, region_rgb]),
        REGION_NAMES,
        REGION_COLORS,
        order=[0, 1, 2, 3],
    )

    # Pipeline order prefixes: 04 regions (after close), 10 final result.
    regions_path = out_dir / f"{output_stem}_04_regions.png"
    regions_npy_path = out_dir / f"{output_stem}_04_regions.npy"
    with ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
        futs = [
            pool.submit(Image.fromarray(stack).save, result_path),
            pool.submit(Image.fromarray(region_stack).save, regions_path),
            pool.submit(np.save, regions_npy_path, region_labels.astype(np.int32)),
        ]
        for fut in futs:
            fut.result()

    if log:
        log_dir = out_dir / f"{output_stem}_intermediates"
        log_dir.mkdir(parents=True, exist_ok=True)
        int_rgb = gray01_to_rgb(intensity)
        seeds_stack = stack_horizontal(
            [
                int_rgb,
                seed_overlay_rgb(intensity, fg_seed, (220, 50, 50)),
                seed_overlay_rgb(intensity, defect_seed, (240, 200, 40)),
            ]
        )
        # Numbered by calculation order (see module docstring).
        region_map_tag = (
            REGION_MAP_INTENSITY_GRADIENT
            if region_map == REGION_MAP_INTENSITY_GRADIENT
            else f"dino_{block_tag}_activation"
        )
        path_01_intensity = log_dir / "01_intensity.png"
        path_02_region_png = log_dir / f"02_{region_map_tag}.png"
        path_02_region_npy = log_dir / f"02_{region_map_tag}.npy"
        path_03_act_hist = log_dir / "03_region_map_gmm4_hist.png"
        path_03_regions_pre = log_dir / "03_regions_pre_close.png"
        path_05_fg_hist = log_dir / "05_intensity_gmm2_fg_hist.png"
        path_05_fg_seed = log_dir / "05_fg_intensity_seed.png"
        path_05_07_seeds = log_dir / "05_07_intensity_seeds.png"
        path_06_fg_promoted = log_dir / "06_fg_promoted_mask.png"
        path_07_def_hist = log_dir / "07_intensity_gmm2_defect_hist.png"
        path_07_def_seed = log_dir / "07_defect_intensity_seed.png"
        path_08_def_promoted = log_dir / "08_defect_promoted_mask.png"
        path_09_final_pre = log_dir / "09_final_pre_close.png"
        path_00_summary = log_dir / "00_summary.json"

        save_histogram_with_gmm(
            path_03_act_hist,
            region_map_values,
            means,
            variances,
            weights,
            thresholds,
            title=f"03 {region_map_name} 4-GMM (region bands)",
            bins=args.hist_bins,
            value_label="Region map",
        )
        save_histogram_with_gmm(
            path_05_fg_hist,
            intensity,
            fg_means,
            fg_vars,
            fg_weights,
            np.asarray([fg_t]),
            title="05 Intensity 2-GMM → FG seed (I ≥ t)",
            bins=args.hist_bins,
            value_label="Intensity",
        )
        save_histogram_with_gmm(
            path_07_def_hist,
            intensity,
            def_means,
            def_vars,
            def_weights,
            np.asarray([defect_t]),
            title="07 Intensity 2-GMM on non-FG → defect seed (I < t)",
            bins=args.hist_bins,
            sample_mask=non_fg,
            value_label="Intensity",
        )
        meta = {
            "input": str(input_path),
            "preprocess": args.preprocess,
            "region_map": region_map,
            "dino_num_blocks": int(args.num_blocks),
            "dino_block_index": block_index,
            "output_stem": output_stem,
            "artifact_order": [
                "01_intensity",
                f"02_{region_map_tag}",
                "03_region_map_gmm4_hist / 03_regions_pre_close",
                "04_regions (parent dir)",
                "05_intensity_gmm2_fg_hist / 05_fg_intensity_seed",
                "06_fg_promoted_mask",
                "07_intensity_gmm2_defect_hist / 07_defect_intensity_seed",
                "08_defect_promoted_mask",
                "09_final_pre_close",
                "10_result (parent dir)",
            ],
            "region_map_gmm4": {
                "source": region_map,
                "means": means.tolist(),
                "variances": variances.tolist(),
                "weights": weights.tolist(),
                "thresholds": thresholds.tolist(),
            },
            "fg_intensity_gmm2": {
                "means": fg_means.tolist(),
                "variances": fg_vars.tolist(),
                "weights": fg_weights.tolist(),
                "threshold": float(fg_t),
                "seed_rule": "intensity >= threshold",
                "seed_pixels": int(fg_seed.sum()),
            },
            "defect_intensity_gmm2": {
                "means": def_means.tolist(),
                "variances": def_vars.tolist(),
                "weights": def_weights.tolist(),
                "threshold": float(defect_t),
                "seed_rule": "intensity < threshold",
                "seed_pixels": int(defect_seed.sum()),
                "fit_on": "pixels outside promoted foreground mask",
                "match_mode": defect_match_mode,
                "promoted_pixels": int(defect_mask.sum()),
            },
            "region_overlap": args.region_overlap,
            "close_radius": args.close_radius,
            "final_class_counts": {
                CLASS_NAMES[i]: int((seg == i).sum()) for i in CLASS_NAMES
            },
            "dino_region_counts": {
                REGION_NAMES[i]: int((region_labels == i).sum()) for i in REGION_NAMES
            },
            "histograms": [
                path_03_act_hist.name,
                path_05_fg_hist.name,
                path_07_def_hist.name,
            ],
        }
        if args.log_probes:
            assert patch_feats is not None
            probe_info = compute_and_save_probes(
                log_dir,
                intensity=intensity,
                gray=gray,
                rgb=rgb_input,
                patch_feats=patch_feats,
                region_labels=region_labels,
                illum_sigma=args.illum_sigma,
                random_state=args.random_state,
                workers=workers,
            )
            meta["probes"] = probe_info
            meta["artifact_order"].append("probes/ (exploratory maps)")
        path_00_summary.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
            log_futs = [
                pool.submit(Image.fromarray(int_rgb).save, path_01_intensity),
                pool.submit(Image.fromarray(act_rgb).save, path_02_region_png),
                pool.submit(np.save, path_02_region_npy, region_map_values),
                pool.submit(
                    Image.fromarray(
                        labels_to_rgb(region_labels_raw, REGION_COLORS)
                    ).save,
                    path_03_regions_pre,
                ),
                pool.submit(Image.fromarray(seeds_stack).save, path_05_07_seeds),
                pool.submit(
                    Image.fromarray(binary_mask_to_rgb(fg_seed)).save,
                    path_05_fg_seed,
                ),
                pool.submit(
                    Image.fromarray(binary_mask_to_rgb(fg_mask)).save,
                    path_06_fg_promoted,
                ),
                pool.submit(
                    Image.fromarray(binary_mask_to_rgb(defect_seed)).save,
                    path_07_def_seed,
                ),
                pool.submit(
                    Image.fromarray(binary_mask_to_rgb(defect_mask)).save,
                    path_08_def_promoted,
                ),
                pool.submit(
                    Image.fromarray(segmentation_to_rgb(seg_raw)).save,
                    path_09_final_pre,
                ),
            ]
            for fut in log_futs:
                fut.result()
        print(f"Saved intermediates: {log_dir}")
        print(f"  03 histogram: {path_03_act_hist}")
        print(f"  05 histogram: {path_05_fg_hist}")
        print(f"  07 histogram: {path_07_def_hist}")
        if args.log_probes:
            print(f"  probes: {log_dir / 'probes'} ({meta.get('probes', {}).get('probe_count', 0)} maps)")

    elapsed = time.perf_counter() - t0
    counts = {CLASS_NAMES[i]: int((seg == i).sum()) for i in CLASS_NAMES}
    region_counts = {
        REGION_NAMES[i]: int((region_labels == i).sum()) for i in REGION_NAMES
    }
    print(f"Input: {input_path}")
    print(f"Output dir: {out_dir}")
    print(f"Region map: {region_map_name}")
    if region_map == REGION_MAP_DINO:
        print(f"DINO: block {block_index} / {args.num_blocks} blocks")
    print(f"Log intermediates: {log}")
    print(f"FG seed: intensity>={fg_t:.4f} ({int(fg_seed.sum())} px)")
    print(f"Defect seed: intensity<{defect_t:.4f} ({int(defect_seed.sum())} px)")
    print(f"Final classes: {counts}")
    print(f"DINO regions: {region_counts}")
    print(f"Saved: {result_path}")
    print(f"Wall time: {elapsed:.3f} s")
    return result_path


def main() -> None:
    args = parse_args()
    if needs_dino_inference(args):
        validate_dino_block_args(int(args.num_blocks), int(args.block_index))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dino_weights = resolve_dino_weights(args.dino_weights)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.source:
        source_root = Path(args.source)
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source folder not found: {source_root}")
        image_paths = discover_images(source_root)
        if not image_paths:
            raise FileNotFoundError(f"No images under {source_root}")
        print(f"Device: {device}")
        print(f"Region map: {args.region_map}")
        if args.region_map == REGION_MAP_DINO:
            print(f"DINO: block {args.block_index} / {args.num_blocks} blocks")
        print(f"Source: {source_root}")
        print(f"Output: {output_root}")
        print(f"Images: {len(image_paths)}")
        print(f"Log intermediates: {args.log_intermediates}")

        dino_model = None
        if needs_dino_inference(args):
            dino_model = load_dinov2_truncated(
                Path(args.dino_repo),
                dino_weights,
                device,
                int(args.num_blocks),
            )
        failed: list[tuple[str, str]] = []
        try:
            for image_path in image_paths:
                rel = image_path.relative_to(source_root)
                out_dir = output_root / rel.parent
                try:
                    run_one(
                        image_path,
                        out_dir,
                        args,
                        device=device,
                        dino_model=dino_model,
                        dino_weights=dino_weights,
                    )
                except Exception as exc:
                    failed.append((str(image_path), str(exc)))
                    print(f"FAILED {image_path}: {exc}")
        finally:
            if dino_model is not None:
                del dino_model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        print(f"Done. Failed: {len(failed)} / {len(image_paths)}")
        if failed:
            for path, err in failed[:10]:
                print(f"  {path}: {err}")
        return

    run_one(
        Path(args.input),
        output_root,
        args,
        device=device,
        dino_model=None,
        dino_weights=dino_weights,
    )


if __name__ == "__main__":
    main()
