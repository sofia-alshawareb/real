"""DINOv2 per-block activation heat maps.

Runs DINOv2 on an image, extracts patch-token L2-norm activations from every
transformer block (min-max normalized, upsampled to full resolution), and saves
grayscale + colormap heat maps plus optional overlays on the input.

Usage:
    python ml/dino_block_activation_maps.py

    python ml/dino_block_activation_maps.py \\
        --input "task3-data/Фото руд по сортам. ч1/Оталькованные руды/2550374-2 10х.JPG" \\
        --output-dir outputs/dino_block_activation_maps

    python ml/dino_block_activation_maps.py \\
        --source "task3-data/Фото руд по сортам. ч1" \\
        --output-dir outputs/dino_block_activation_maps
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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
DEFAULT_OUTPUT_DIR = _ROOT / "outputs/dino_block_activation_maps"
DEFAULT_SOURCE = _DATA_RAW_TASK3 / "Фото руд по сортам. ч1"
DEFAULT_DINO_REPO = _ROOT / "data/models/dinov2"
DEFAULT_DINO_WEIGHTS = _ROOT / "data/models/checkpoints/dinov2_vits14_reg4_pretrain.pth"
FALLBACK_DINO_WEIGHTS = (
    Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth"
)
PATCH_SIZE = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save DINOv2 per-block activation heat maps for an image."
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
        "--num-blocks",
        type=int,
        default=0,
        help=(
            "Number of ViT blocks to keep (default: 0 = all blocks in the loaded model)."
        ),
    )
    parser.add_argument(
        "--blocks",
        default="",
        help=(
            "Comma-separated block indices to export (default: all kept blocks, "
            "e.g. 0,1,2 or 11 for last block only)."
        ),
    )
    parser.add_argument(
        "--colormap",
        default="inferno",
        help="Matplotlib colormap for heat maps (default: inferno).",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Alpha for heat-map overlay on the input image (default: 0.55).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip saving overlay PNGs (input + heat map).",
    )
    parser.add_argument(
        "--no-npy",
        action="store_true",
        help="Skip saving raw activation .npy files.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip saving the all-blocks overview grid PNG.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip images whose grid PNG already exists (default: true).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(2, min(8, os.cpu_count() or 4)),
        help="Thread-pool size for saving outputs (default: min(8, CPU count)).",
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


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def discover_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


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


def load_dinov2(
    repo_dir: Path,
    weights: str,
    device: torch.device,
    num_blocks: int = 0,
) -> nn.Module:
    kwargs: dict = {"pretrained": True}
    if weights:
        kwargs["weights"] = weights
    model = torch.hub.load(str(repo_dir), "dinov2_vits14_reg", source="local", **kwargs)
    if num_blocks > 0:
        prune_vit_blocks(model, num_blocks)
    return model.to(device).eval()


def _parse_block_indices(raw: str, num_blocks: int) -> list[int]:
    if not raw.strip():
        return list(range(num_blocks))
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    for idx in indices:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"Block index {idx} out of range [0, {num_blocks - 1}]")
    return indices


def _features_to_activation(feats: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    """Patch-token map (C, hp, wp) → min-max L2 norm upsampled to target_hw."""
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


def activation_to_heatmap(
    activation: np.ndarray,
    colormap: str,
) -> np.ndarray:
    cmap = plt.get_cmap(colormap)
    colored = cmap(np.clip(activation, 0.0, 1.0))[:, :, :3]
    return (colored * 255.0).astype(np.uint8)


def overlay_heatmap(
    rgb: np.ndarray,
    activation: np.ndarray,
    colormap: str,
    alpha: float,
) -> np.ndarray:
    heat = activation_to_heatmap(activation, colormap)
    blended = alpha * heat.astype(np.float32) + (1.0 - alpha) * rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


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


@torch.inference_mode()
def dino_all_block_activations(
    rgb: np.ndarray,
    *,
    device: torch.device,
    model: nn.Module,
    block_indices: list[int],
) -> dict[int, np.ndarray]:
    """Return {block_index: activation_map} for the requested blocks."""
    h, w = rgb.shape[:2]
    image = rgb_to_dino_tensor(rgb).to(device)
    outputs = model.get_intermediate_layers(
        image,
        n=block_indices,
        reshape=True,
        norm=True,
    )
    activations: dict[int, np.ndarray] = {}
    for block_idx, tokens in zip(block_indices, outputs):
        feats = tokens[0].detach().float().cpu()
        activations[block_idx] = _features_to_activation(feats, (h, w))
    del image, outputs
    return activations


def save_block_grid(
    path: Path,
    rgb: np.ndarray,
    activations: dict[int, np.ndarray],
    colormap: str,
) -> None:
    """Save input + one heat map per block in a single row."""
    block_indices = sorted(activations)
    panels = [rgb]
    for block_idx in block_indices:
        panels.append(activation_to_heatmap(activations[block_idx], colormap))

    grid = stack_horizontal(panels)
    n_blocks = len(block_indices)
    fig_w = max(8.0, 2.0 + 2.5 * (n_blocks + 1))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    ax.imshow(grid)
    ax.axis("off")
    labels = ["input"] + [f"block {i}" for i in block_indices]
    panel_w = grid.shape[1] / len(labels)
    for i, label in enumerate(labels):
        ax.text(
            (i + 0.5) * panel_w,
            -8,
            label,
            ha="center",
            va="top",
            transform=ax.transData,
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def run_one(
    input_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    device: torch.device,
    dino_model: nn.Module,
    block_indices: list[int],
    dino_weights: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    stem = input_path.stem
    grid_path = out_dir / f"{stem}_all_blocks_grid.png"
    if args.skip_existing and grid_path.exists() and not args.no_grid:
        print(f"Skip existing: {grid_path}")
        return grid_path

    t0 = time.perf_counter()
    rgb = load_rgb(input_path)
    activations = dino_all_block_activations(
        rgb,
        device=device,
        model=dino_model,
        block_indices=block_indices,
    )

    save_overlay = not args.no_overlay
    save_npy = not args.no_npy
    save_grid = not args.no_grid

    block_dir = out_dir / f"{stem}_blocks"
    block_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "input": str(input_path),
        "image_size": [int(rgb.shape[0]), int(rgb.shape[1])],
        "block_indices": block_indices,
        "num_blocks_in_model": int(getattr(dino_model, "n_blocks", len(block_indices))),
        "colormap": args.colormap,
        "overlay_alpha": float(args.overlay_alpha),
        "dino_repo": args.dino_repo,
        "dino_weights": dino_weights,
        "outputs": {},
    }

    def _save_block(block_idx: int, activation: np.ndarray) -> dict[str, str]:
        heat = activation_to_heatmap(activation, args.colormap)
        heat_path = block_dir / f"block_{block_idx:02d}_heatmap.png"
        Image.fromarray(heat).save(heat_path)
        paths: dict[str, str] = {"heatmap": str(heat_path.name)}
        if save_npy:
            npy_path = block_dir / f"block_{block_idx:02d}_activation.npy"
            np.save(npy_path, activation)
            paths["activation_npy"] = str(npy_path.name)
        if save_overlay:
            overlay_path = block_dir / f"block_{block_idx:02d}_overlay.png"
            overlay = overlay_heatmap(
                rgb,
                activation,
                args.colormap,
                args.overlay_alpha,
            )
            Image.fromarray(overlay).save(overlay_path)
            paths["overlay"] = str(overlay_path.name)
        return paths

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            block_idx: pool.submit(_save_block, block_idx, activation)
            for block_idx, activation in activations.items()
        }
        for block_idx, fut in futs.items():
            meta["outputs"][f"block_{block_idx:02d}"] = fut.result()

    if save_grid:
        save_block_grid(grid_path, rgb, activations, args.colormap)
        meta["grid"] = grid_path.name

    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Input: {input_path}")
    print(f"Output dir: {out_dir}")
    print(f"Blocks: {block_indices}")
    print(f"Saved block maps: {block_dir}")
    if save_grid:
        print(f"Saved grid: {grid_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Wall time: {elapsed:.3f} s")
    return grid_path if save_grid else block_dir


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dino_weights = resolve_dino_weights(args.dino_weights)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dino_model = load_dinov2(
        Path(args.dino_repo),
        dino_weights,
        device,
        num_blocks=args.num_blocks,
    )
    num_blocks = int(getattr(dino_model, "n_blocks", len(dino_model.blocks)))
    block_indices = _parse_block_indices(args.blocks, num_blocks)

    print(f"Device: {device}")
    print(f"DINO blocks in model: {num_blocks}")
    print(f"Exporting blocks: {block_indices}")

    try:
        if args.source:
            source_root = Path(args.source)
            if not source_root.is_dir():
                raise FileNotFoundError(f"Source folder not found: {source_root}")
            image_paths = discover_images(source_root)
            if not image_paths:
                raise FileNotFoundError(f"No images under {source_root}")
            print(f"Source: {source_root}")
            print(f"Output: {output_root}")
            print(f"Images: {len(image_paths)}")

            failed: list[tuple[str, str]] = []
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
                        block_indices=block_indices,
                        dino_weights=dino_weights,
                    )
                except Exception as exc:
                    failed.append((str(image_path), str(exc)))
                    print(f"FAILED {image_path}: {exc}")
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
            dino_model=dino_model,
            block_indices=block_indices,
            dino_weights=dino_weights,
        )
    finally:
        del dino_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
