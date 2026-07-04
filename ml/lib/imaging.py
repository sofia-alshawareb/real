"""Image loading and normalization for the ML service."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_id_from_rgb(rgb: np.ndarray) -> str:
    """Stable id from normalized 8-bit RGB pixels (R9)."""
    return content_hash(save_rgb_png(rgb))[:16]


def image_id_from_bytes(data: bytes) -> str:
    """Derive image_id from normalized RGB content, not raw upload bytes."""
    rgb = load_rgb_from_bytes(data)
    return image_id_from_rgb(rgb)


def normalize_to_rgb8(arr: np.ndarray) -> np.ndarray:
    """Convert array to uint8 RGB, scaling 16-bit or float inputs."""
    if arr.dtype == np.uint16:
        arr = (arr.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    elif arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        return arr[:, :, :3]
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    return arr


def load_rgb_from_bytes(data: bytes) -> np.ndarray:
    """Load first page of image bytes as normalized 8-bit RGB (R4)."""
    with Image.open(io.BytesIO(data)) as img:
        img.seek(0)
        if img.mode not in ("RGB", "L", "I", "I;16", "F"):
            img = img.convert("RGB")
        arr = np.asarray(img)
    return normalize_to_rgb8(arr)


def load_rgb_from_path(path: Path) -> np.ndarray:
    return load_rgb_from_bytes(path.read_bytes())


def save_rgb_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
