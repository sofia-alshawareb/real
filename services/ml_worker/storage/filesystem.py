"""Filesystem-backed artifact store with atomic writes."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ml.lib.constants import CLASS_COLORS, CLASS_NAMES, MASK_SCHEMA_VERSION
from ml.lib.types import DinoArtifacts, SegmentationResult
from services.ml_worker.storage.base import ArtifactStore


def _labels_to_paletted_png(labels: np.ndarray) -> bytes:
    from io import BytesIO

    labels = labels.astype(np.uint8)
    img = Image.fromarray(labels, mode="P")
    palette = [0, 0, 0] * 256
    for cls_id, (r, g, b) in CLASS_COLORS.items():
        palette[cls_id * 3 : cls_id * 3 + 3] = [r, g, b]
    img.putpalette(palette)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


UI_CLASS_COLORS = CLASS_COLORS
UI_CLASS_NAMES = CLASS_NAMES


def _ui_labels_to_colored_png(labels: np.ndarray) -> bytes:
    from io import BytesIO

    labels = labels.astype(np.uint8)
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in UI_CLASS_COLORS.items():
        rgb[labels == cls] = color
    buf = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _ui_labels_to_grayscale_png(labels: np.ndarray) -> bytes:
    from io import BytesIO

    labels = labels.astype(np.uint8)
    gray = np.minimum(4, labels) * 64
    buf = BytesIO()
    Image.fromarray(gray.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class FilesystemArtifactStore:
    def __init__(self, root: Path | str, images_subdir: str = "images") -> None:
        self.root = Path(root)
        self.images_subdir = images_subdir
        self.root.mkdir(parents=True, exist_ok=True)

    def _image_dir(self, image_id: str) -> Path:
        return self.root / image_id

    def _image_path(self, image_id: str) -> Path:
        return self._image_dir(image_id) / self.images_subdir / "normalized.png"

    def _seg_dir(self, image_id: str) -> Path:
        return self._image_dir(image_id) / "segmentation"

    def _dino_dir(self, image_id: str) -> Path:
        return self._image_dir(image_id) / "dino"

    def _manual_dir(self, image_id: str) -> Path:
        return self._image_dir(image_id) / "manual"

    def _temp_dir(self, image_id: str) -> Path:
        return self._image_dir(image_id) / ".tmp"

    def save_image(self, image_id: str, rgb: np.ndarray, raw_bytes: bytes) -> None:
        img_dir = self._image_dir(image_id) / self.images_subdir
        img_dir.mkdir(parents=True, exist_ok=True)
        png_buf = self._rgb_to_png(rgb)
        _atomic_write_bytes(img_dir / "normalized.png", png_buf)
        _atomic_write_bytes(img_dir / "upload.bin", raw_bytes)
        meta = {
            "schema_version": MASK_SCHEMA_VERSION,
            "image_id": image_id,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(img_dir / "meta.json", meta)

    @staticmethod
    def _rgb_to_png(rgb: np.ndarray) -> bytes:
        from io import BytesIO

        buf = BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()

    def load_rgb(self, image_id: str) -> np.ndarray:
        path = self._image_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_id}")
        return np.asarray(Image.open(path).convert("RGB"))

    def image_exists(self, image_id: str) -> bool:
        return self._image_path(image_id).exists()

    def save_dino(self, image_id: str, artifacts: DinoArtifacts) -> None:
        self.write_dino(self._dino_dir(image_id), artifacts)

    def save_segmentation(self, image_id: str, result: SegmentationResult) -> None:
        self.write_segmentation(self._seg_dir(image_id), image_id, result)

    def write_segmentation(
        self, dest_dir: Path, image_id: str, result: SegmentationResult
    ) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_npy(dest_dir / "labels.npy", result.labels)
        _atomic_write_bytes(dest_dir / "labels.png", _labels_to_paletted_png(result.labels))
        summary = {
            "schema_version": MASK_SCHEMA_VERSION,
            "image_id": image_id,
            "native_width": result.native_width,
            "native_height": result.native_height,
            "mask_width": result.mask_width,
            "mask_height": result.mask_height,
            "mask_to_native_scale": result.mask_to_native_scale,
            "classes": {str(k): v for k, v in CLASS_NAMES.items()},
            **result.metadata,
        }
        _atomic_write_json(dest_dir / "summary.json", summary)

    def write_block01_features(
        self,
        dest_dir: Path,
        features: np.ndarray,
        *,
        activation: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_npy(dest_dir / "block01_features.npy", features)
        if activation is not None:
            _atomic_write_npy(dest_dir / "block01_activation.npy", activation)
        payload = {
            "schema_version": MASK_SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        }
        _atomic_write_json(dest_dir / "block01_meta.json", payload)

    def load_block01_features(self, image_id: str) -> np.ndarray:
        path = self._dino_dir(image_id) / "block01_features.npy"
        if not path.exists():
            raise FileNotFoundError(f"Block-1 features not ready for {image_id}")
        return np.load(path)

    def load_block01_activation(self, image_id: str) -> np.ndarray:
        path = self._dino_dir(image_id) / "block01_activation.npy"
        if not path.exists():
            raise FileNotFoundError(f"Block-1 activation not ready for {image_id}")
        return np.load(path)

    def write_block11_features(self, dest_dir: Path, features: np.ndarray, meta: dict | None = None) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_npy(dest_dir / "block11_features.npy", features)
        payload = {
            "schema_version": MASK_SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        }
        _atomic_write_json(dest_dir / "block11_meta.json", payload)

    def load_block11_features(self, image_id: str) -> np.ndarray:
        path = self._dino_dir(image_id) / "block11_features.npy"
        if not path.exists():
            raise FileNotFoundError(f"Block-11 features not ready for {image_id}")
        return np.load(path)

    def dino_features_ready(self, image_id: str) -> bool:
        dino_dir = self._dino_dir(image_id)
        return (dino_dir / "block01_features.npy").exists() and (
            dino_dir / "block01_activation.npy"
        ).exists() and (dino_dir / "block11_features.npy").exists()

    def load_labels(self, image_id: str) -> np.ndarray:
        path = self._seg_dir(image_id) / "labels.npy"
        if not path.exists():
            raise FileNotFoundError(f"Segmentation not ready for {image_id}")
        return np.load(path)

    def write_dino(self, dest_dir: Path, artifacts: DinoArtifacts) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_npy(dest_dir / "block01_activation.npy", artifacts.block01_activation)
        _atomic_write_npy(dest_dir / "block11_activation.npy", artifacts.block11_activation)
        _atomic_write_npy(dest_dir / "block01_features.npy", artifacts.block01_features)
        _atomic_write_npy(dest_dir / "block11_features.npy", artifacts.block11_features)
        meta = dict(artifacts.meta)
        meta["schema_version"] = MASK_SCHEMA_VERSION
        meta["saved_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(dest_dir / "meta.json", meta)

    def load_dino(self, image_id: str) -> DinoArtifacts:
        dino_dir = self._dino_dir(image_id)
        meta_path = dino_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return DinoArtifacts(
            block01_activation=np.load(dino_dir / "block01_activation.npy"),
            block11_activation=np.load(dino_dir / "block11_activation.npy"),
            block01_features=np.load(dino_dir / "block01_features.npy"),
            block11_features=np.load(dino_dir / "block11_features.npy"),
            meta=meta,
        )

    def segmentation_ready(self, image_id: str) -> bool:
        return (self._seg_dir(image_id) / "labels.npy").exists()

    def get_mask_bytes(self, image_id: str) -> bytes:
        png_path = self._seg_dir(image_id) / "labels.png"
        if not png_path.exists():
            raise FileNotFoundError(f"Mask not ready for {image_id}")
        return png_path.read_bytes()

    def get_mask_metadata(self, image_id: str) -> dict:
        summary_path = self._seg_dir(image_id) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Mask metadata not ready for {image_id}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def save_user_drawn_mask(self, image_id: str, labels: np.ndarray) -> Path:
        """Persist hand-drawn UI class mask (dev tooling)."""
        labels = np.asarray(labels, dtype=np.uint8)
        if labels.ndim != 2:
            raise ValueError("labels must be 2-D")
        manual_dir = self._manual_dir(image_id)
        manual_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_npy(manual_dir / "user_drawn_labels.npy", labels)
        _atomic_write_bytes(manual_dir / "user_drawn_colored.png", _ui_labels_to_colored_png(labels))
        _atomic_write_bytes(
            manual_dir / "user_drawn_grayscale.png", _ui_labels_to_grayscale_png(labels)
        )
        meta = {
            "schema_version": MASK_SCHEMA_VERSION,
            "image_id": image_id,
            "width": int(labels.shape[1]),
            "height": int(labels.shape[0]),
            "encoding": "ui_class_index",
            "classes": {str(k): v for k, v in UI_CLASS_NAMES.items()},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(manual_dir / "meta.json", meta)
        return manual_dir

    def user_drawn_mask_ready(self, image_id: str) -> bool:
        return (self._manual_dir(image_id) / "user_drawn_labels.npy").exists()

    def cleanup_temp(self, image_id: str) -> None:
        tmp = self._temp_dir(image_id)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    def begin_temp(self, image_id: str) -> Path:
        tmp = self._temp_dir(image_id)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

    def commit_temp(self, image_id: str, temp_dir: Path) -> None:
        """Move temp artifacts into final locations atomically."""
        for name in ("dino", "segmentation"):
            src = temp_dir / name
            if not src.exists():
                continue
            dst = self._image_dir(image_id) / name
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)
        self.cleanup_temp(image_id)


__all__ = ["FilesystemArtifactStore"]
