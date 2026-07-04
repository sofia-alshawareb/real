"""Load, save, and update compiled calibration artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ml.lib.calibration.histograms import build_rgb_histogram
from ml.lib.calibration.types import CalibrationData, ClassCalibrationStats
from ml.lib.constants import (
    CALIB_BACKGROUND_KEY,
    CALIB_CLASS_KEYS,
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_MAX_EMBEDDING_SAMPLES,
    DEFAULT_MAX_RGB_SAMPLES,
    DEFAULT_RGB_HIST_BINS,
)


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


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


def _normalize_embedding_rows(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows.astype(np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.maximum(norms, 1e-8)).astype(np.float32)


def _cap_rows(rows: np.ndarray, max_rows: int, random_state: int) -> np.ndarray:
    if rows.shape[0] <= max_rows:
        return rows
    rng = np.random.default_rng(random_state)
    idx = rng.choice(rows.shape[0], size=max_rows, replace=False)
    return rows[idx]


class CalibrationStore:
    def __init__(self, root: Path | str, *, rgb_hist_bins: int = DEFAULT_RGB_HIST_BINS) -> None:
        self.root = Path(root)
        self.rgb_hist_bins = rgb_hist_bins
        self._data: CalibrationData | None = None

    @property
    def rgb_dir(self) -> Path:
        return self.root / "rgb"

    @property
    def emb_dir(self) -> Path:
        return self.root / "embeddings"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    @property
    def histogram_path(self) -> Path:
        return self.root / "rgb_histograms.npz"

    def exists(self) -> bool:
        return self.summary_path.exists()

    def load(self) -> CalibrationData:
        if not self.exists():
            raise FileNotFoundError(f"Calibration not found at {self.root}")
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        rgb_samples: dict[str, np.ndarray] = {}
        emb_samples: dict[str, np.ndarray] = {}
        stats: dict[str, ClassCalibrationStats] = {}
        for key in CALIB_CLASS_KEYS:
            rgb_path = self.rgb_dir / f"{key}.npy"
            emb_path = self.emb_dir / f"{key}.npy"
            rgb_samples[key] = (
                np.load(rgb_path).astype(np.float32) if rgb_path.exists() else np.zeros((0, 3), np.float32)
            )
            emb_samples[key] = (
                np.load(emb_path).astype(np.float32) if emb_path.exists() else np.zeros((0, 384), np.float32)
            )
            s = summary.get("rgb", {}).get(key, {})
            e = summary.get("embedding", {}).get(key, {})
            stats[key] = ClassCalibrationStats(
                count=int(s.get("count", rgb_samples[key].shape[0])),
                mean_rgb=np.asarray(s["mean"], dtype=np.float32) if s.get("mean") is not None else None,
                std_rgb=np.asarray(s["std"], dtype=np.float32) if s.get("std") is not None else None,
                mean_embedding=(
                    np.asarray(e["mean"], dtype=np.float32) if e.get("mean") is not None else None
                ),
            )
        bg_path = self.emb_dir / f"{CALIB_BACKGROUND_KEY}.npy"
        if bg_path.exists():
            bg_emb = np.load(bg_path).astype(np.float32)
            emb_samples[CALIB_BACKGROUND_KEY] = bg_emb
            e = summary.get("embedding", {}).get(CALIB_BACKGROUND_KEY, {})
            stats[CALIB_BACKGROUND_KEY] = ClassCalibrationStats(
                count=int(e.get("count", bg_emb.shape[0])),
                mean_embedding=(
                    np.asarray(e["mean"], dtype=np.float32) if e.get("mean") is not None else None
                ),
            )
        histograms: dict[str, np.ndarray] = {}
        if self.histogram_path.exists():
            with np.load(self.histogram_path) as npz:
                for key in CALIB_CLASS_KEYS:
                    if key in npz:
                        histograms[key] = npz[key]
        self._data = CalibrationData(
            rgb_samples=rgb_samples,
            embedding_samples=emb_samples,
            rgb_histograms=histograms,
            stats=stats,
            source_images=list(summary.get("source_images", [])),
            schema_version=str(summary.get("schema_version", CALIBRATION_SCHEMA_VERSION)),
            meta=summary,
        )
        return self._data

    def get(self) -> CalibrationData:
        if self._data is None:
            return self.load()
        return self._data

    def reload(self) -> CalibrationData:
        self._data = None
        return self.load()

    def _compute_stats(self, data: CalibrationData) -> dict[str, Any]:
        rgb_summary: dict[str, Any] = {}
        emb_summary: dict[str, Any] = {}
        histograms: dict[str, np.ndarray] = {}
        for key in CALIB_CLASS_KEYS:
            rgb = data.rgb_samples.get(key, np.zeros((0, 3), np.float32))
            emb = _normalize_embedding_rows(data.embedding_samples.get(key, np.zeros((0, 384), np.float32)))
            data.embedding_samples[key] = emb
            if rgb.shape[0] > 0:
                rgb_summary[key] = {
                    "mean": rgb.mean(axis=0).tolist(),
                    "std": rgb.std(axis=0).tolist(),
                    "count": int(rgb.shape[0]),
                }
                histograms[key] = build_rgb_histogram(rgb, bins=self.rgb_hist_bins)
            else:
                rgb_summary[key] = {"mean": None, "std": None, "count": 0}
                histograms[key] = np.zeros((self.rgb_hist_bins,) * 3, dtype=np.float64)
            if emb.shape[0] > 0:
                mean_emb = emb.mean(axis=0)
                norm = float(np.linalg.norm(mean_emb))
                if norm > 1e-8:
                    mean_emb = mean_emb / norm
                emb_summary[key] = {
                    "mean": mean_emb.tolist(),
                    "count": int(emb.shape[0]),
                }
            else:
                emb_summary[key] = {"mean": None, "count": 0}
            data.stats[key] = ClassCalibrationStats(
                count=int(rgb.shape[0]),
                mean_rgb=rgb.mean(axis=0) if rgb.shape[0] else None,
                std_rgb=rgb.std(axis=0) if rgb.shape[0] else None,
                mean_embedding=(
                    _normalize_embedding_rows(emb.mean(axis=0, keepdims=True))[0]
                    if emb.shape[0]
                    else None
                ),
            )
        bg = _normalize_embedding_rows(
            data.embedding_samples.get(CALIB_BACKGROUND_KEY, np.zeros((0, 384), np.float32))
        )
        if CALIB_BACKGROUND_KEY in data.embedding_samples:
            data.embedding_samples[CALIB_BACKGROUND_KEY] = bg
        if bg.shape[0] > 0:
            mean_emb = _normalize_embedding_rows(bg.mean(axis=0, keepdims=True))[0]
            emb_summary[CALIB_BACKGROUND_KEY] = {
                "mean": mean_emb.tolist(),
                "count": int(bg.shape[0]),
            }
            data.stats[CALIB_BACKGROUND_KEY] = ClassCalibrationStats(
                count=int(bg.shape[0]),
                mean_embedding=mean_emb,
            )
        else:
            emb_summary[CALIB_BACKGROUND_KEY] = {"mean": None, "count": 0}
        data.rgb_histograms = histograms
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "source_images": data.source_images,
            "segmentation_modes": ["intensity", "embedding"],
            "rgb_hist_bins": self.rgb_hist_bins,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "rgb": rgb_summary,
            "embedding": emb_summary,
        }

    def save(self, data: CalibrationData) -> None:
        summary = self._compute_stats(data)
        tmp = Path(tempfile.mkdtemp(dir=self.root.parent if self.root.parent.exists() else None))
        try:
            tmp_rgb = tmp / "rgb"
            tmp_emb = tmp / "embeddings"
            tmp_rgb.mkdir(parents=True)
            tmp_emb.mkdir(parents=True)
            for key in CALIB_CLASS_KEYS:
                _atomic_write_npy(tmp_rgb / f"{key}.npy", data.rgb_samples.get(key, np.zeros((0, 3), np.float32)))
                _atomic_write_npy(
                    tmp_emb / f"{key}.npy",
                    data.embedding_samples.get(key, np.zeros((0, 384), np.float32)),
                )
            bg_rows = data.embedding_samples.get(CALIB_BACKGROUND_KEY, np.zeros((0, 384), np.float32))
            if bg_rows.size:
                _atomic_write_npy(tmp_emb / f"{CALIB_BACKGROUND_KEY}.npy", bg_rows)
            hist_payload = {k: data.rgb_histograms[k] for k in CALIB_CLASS_KEYS if k in data.rgb_histograms}
            np.savez_compressed(tmp / "rgb_histograms.npz", **hist_payload)
            _atomic_write_json(tmp / "summary.json", summary)

            self.root.mkdir(parents=True, exist_ok=True)
            for name in ("rgb", "embeddings"):
                dst = self.root / name
                src = tmp / name
                if dst.exists():
                    shutil.rmtree(dst)
                src.rename(dst)
            shutil.move(str(tmp / "rgb_histograms.npz"), str(self.histogram_path))
            shutil.move(str(tmp / "summary.json"), str(self.summary_path))
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        self._data = data

    def append_samples(
        self,
        class_key: str,
        rgb_rows: np.ndarray,
        emb_rows: np.ndarray,
        *,
        max_rgb_samples: int = DEFAULT_MAX_RGB_SAMPLES,
        max_emb_samples: int = DEFAULT_MAX_EMBEDDING_SAMPLES,
        random_state: int = 0,
    ) -> dict[str, Any]:
        if class_key not in CALIB_CLASS_KEYS:
            raise ValueError(f"Unknown class key: {class_key!r}")
        data = self.get()
        prev_rgb = data.rgb_samples.get(class_key, np.zeros((0, 3), np.float32))
        prev_emb = data.embedding_samples.get(class_key, np.zeros((0, 384), np.float32))
        if rgb_rows.size:
            merged_rgb = np.vstack([prev_rgb, rgb_rows.astype(np.float32)])
            data.rgb_samples[class_key] = _cap_rows(merged_rgb, max_rgb_samples, random_state)
        if emb_rows.size:
            merged_emb = np.vstack([prev_emb, _normalize_embedding_rows(emb_rows.astype(np.float32))])
            data.embedding_samples[class_key] = _cap_rows(merged_emb, max_emb_samples, random_state)
        self.save(data)
        return {
            "class": class_key,
            "appended_rgb": int(rgb_rows.shape[0]) if rgb_rows.size else 0,
            "appended_embedding": int(emb_rows.shape[0]) if emb_rows.size else 0,
            "total_rgb": int(data.rgb_samples[class_key].shape[0]),
            "total_embedding": int(data.embedding_samples[class_key].shape[0]),
        }

    def merge_image_samples(
        self,
        rgb_by_class: dict[str, np.ndarray],
        emb_by_class: dict[str, np.ndarray],
        source_id: str,
    ) -> None:
        data = self.get() if self.exists() else CalibrationData()
        if source_id not in data.source_images:
            data.source_images.append(source_id)
        for key in CALIB_CLASS_KEYS:
            chunks_rgb = [data.rgb_samples.get(key, np.zeros((0, 3), np.float32))]
            chunks_emb = [data.embedding_samples.get(key, np.zeros((0, 384), np.float32))]
            if key in rgb_by_class and rgb_by_class[key].size:
                chunks_rgb.append(rgb_by_class[key].astype(np.float32))
            if key in emb_by_class and emb_by_class[key].size:
                chunks_emb.append(_normalize_embedding_rows(emb_by_class[key].astype(np.float32)))
            data.rgb_samples[key] = np.vstack(chunks_rgb) if chunks_rgb[0].size or len(chunks_rgb) > 1 else chunks_rgb[0]
            if len(chunks_emb) > 1 or chunks_emb[0].size:
                data.embedding_samples[key] = np.vstack([c for c in chunks_emb if c.size]) if any(c.size for c in chunks_emb) else np.zeros((0, 384), np.float32)
        self.save(data)

    def initialize_from_merged(
        self,
        merged_rgb: dict[str, np.ndarray],
        merged_emb: dict[str, np.ndarray],
        source_images: list[str],
    ) -> None:
        data = CalibrationData(source_images=list(source_images))
        for key in CALIB_CLASS_KEYS:
            data.rgb_samples[key] = merged_rgb.get(key, np.zeros((0, 3), np.float32)).astype(np.float32)
            data.embedding_samples[key] = _normalize_embedding_rows(
                merged_emb.get(key, np.zeros((0, 384), np.float32)).astype(np.float32)
            )
        bg_rows = merged_emb.get(CALIB_BACKGROUND_KEY, np.zeros((0, 384), np.float32))
        if bg_rows.size:
            data.embedding_samples[CALIB_BACKGROUND_KEY] = _normalize_embedding_rows(
                bg_rows.astype(np.float32)
            )
        self.save(data)

    def summary_counts(self) -> dict[str, int]:
        if not self.exists():
            return {k: 0 for k in CALIB_CLASS_KEYS}
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        return {k: int(summary.get("rgb", {}).get(k, {}).get("count", 0)) for k in CALIB_CLASS_KEYS}
