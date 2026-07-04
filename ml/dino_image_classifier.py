"""DINO-based image-level binary classifier.

Class 0: fragmented / many small regions.
Class 1: uniform / continuous foreground structure.

Data layout (class folders inside each split):

    data/classification/
      train/
        0/   # class-0 images
        1/   # class-1 images
      val/
        0/
        1/

Folder names must be integer labels (0, 1, …). One prediction per full image.

Usage:
    python ml/dino_image_classifier.py train --config configs/dino_image_classifier.yaml
    python ml/dino_image_classifier.py predict --config configs/dino_image_classifier.yaml \\
        --image path/to/image.jpg --checkpoint outputs/dino_image_classifier/default/best.pt

    python ml/dino_image_classifier.py eval-val --config configs/dino_image_classifier.yaml \\
        --checkpoint outputs/dino_image_classifier/frozen_backbone/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = _ROOT / "configs/dino_image_classifier.yaml"
DEFAULT_DINO_REPO = _ROOT / "data/models/dinov2"
DEFAULT_DINO_WEIGHTS = _ROOT / "data/models/checkpoints/dinov2_vits14_reg4_pretrain.pth"
FALLBACK_DINO_WEIGHTS = (
    Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth"
)
PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
_VALID_POOLING = frozenset({"mean", "max", "std"})
_VALID_HEAD_TYPES = frozenset({"structure", "gap_linear", "gap_mlp"})


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def resolve_dino_weights(path: str) -> str:
    if path:
        p = resolve_path(path)
        if p.exists():
            return str(p)
    if DEFAULT_DINO_WEIGHTS.exists():
        return str(DEFAULT_DINO_WEIGHTS)
    if FALLBACK_DINO_WEIGHTS.exists():
        return str(FALLBACK_DINO_WEIGHTS)
    return ""


def load_config(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for item in overrides:
            key, raw = item.split("=", 1)
            _set_nested(cfg, key.strip(), _parse_override_value(raw.strip()))
    _validate_config(cfg)
    return cfg


def _parse_override_value(raw: str) -> Any:
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_override_value(part.strip()) for part in inner.split(",")]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _validate_config(cfg: dict[str, Any]) -> None:
    head_cfg = cfg["head"]
    pooling = head_cfg["pooling"]
    if not pooling:
        raise ValueError("head.pooling must include at least one of: mean, max, std")
    unknown = set(pooling) - _VALID_POOLING
    if unknown:
        raise ValueError(f"Unknown pooling ops: {sorted(unknown)}")
    head_type = head_cfg.get("type", "structure")
    if head_type not in _VALID_HEAD_TYPES:
        raise ValueError(f"Unknown head.type: {head_type}")
    if head_type == "structure" and not head_cfg.get("conv_channels"):
        raise ValueError("head.conv_channels required for structure head")
    if not bool(cfg["model"].get("freeze_dino", True)):
        raise ValueError("Only freeze_dino=true is supported in this project setup")
    if cfg["model"]["num_blocks"] < 1:
        raise ValueError("model.num_blocks must be >= 1")
    if cfg["data"]["image_size"] % PATCH_SIZE != 0:
        raise ValueError(
            f"data.image_size must be divisible by {PATCH_SIZE}, "
            f"got {cfg['data']['image_size']}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class ClassFolderDataset(Dataset):
    """Reads ``split_dir/{class_label}/*`` image files."""

    def __init__(
        self,
        split_dir: Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.transform = transform
        self.samples: list[Sample] = []
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            try:
                label = int(class_dir.name)
            except ValueError as exc:
                raise ValueError(
                    f"Class folder names must be integer labels, got: {class_dir.name}"
                ) from exc
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    self.samples.append(Sample(path=path, label=label))

        if not self.samples:
            raise FileNotFoundError(f"No images found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label, str(sample.path)


def build_transforms(image_size: int, augment: bool) -> transforms.Compose:
    resize = transforms.Resize((image_size, image_size))
    if augment:
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def prune_vit_blocks(model: nn.Module, num_blocks: int) -> nn.Module:
    model.blocks = nn.ModuleList(list(model.blocks[:num_blocks]))
    if hasattr(model, "n_blocks"):
        model.n_blocks = num_blocks
    return model


def load_dinov2(
    repo_dir: Path,
    weights: str,
    num_blocks: int,
    device: torch.device,
) -> nn.Module:
    kwargs: dict[str, Any] = {"pretrained": True}
    if weights:
        kwargs["weights"] = weights
    model = torch.hub.load(str(repo_dir), "dinov2_vits14_reg", source="local", **kwargs)
    prune_vit_blocks(model, num_blocks)
    return model.to(device)


def _pool_feature_map(features: torch.Tensor, pooling: list[str]) -> torch.Tensor:
    stats: list[torch.Tensor] = []
    if "mean" in pooling:
        stats.append(features.mean(dim=(2, 3)))
    if "max" in pooling:
        stats.append(features.amax(dim=(2, 3)))
    if "std" in pooling:
        stats.append(features.std(dim=(2, 3)))
    return torch.cat(stats, dim=1)


class GapLinearHead(nn.Module):
    """Global pool on DINO patch tokens + single linear layer."""

    def __init__(self, embed_dim: int, pooling: list[str]) -> None:
        super().__init__()
        self.pooling = list(pooling)
        self.fc = nn.Linear(embed_dim * len(self.pooling), 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc(_pool_feature_map(tokens, self.pooling)).squeeze(1)


class GapMlpHead(nn.Module):
    """Global pool on DINO patch tokens + small MLP."""

    def __init__(
        self,
        embed_dim: int,
        mlp_hidden: int,
        dropout: float,
        pooling: list[str],
    ) -> None:
        super().__init__()
        self.pooling = list(pooling)
        in_dim = embed_dim * len(self.pooling)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlp(_pool_feature_map(tokens, self.pooling)).squeeze(1)


class StructureHead(nn.Module):
    """Conv stack on patch-token map + configurable global pooling + MLP."""

    def __init__(
        self,
        embed_dim: int,
        conv_channels: list[int],
        kernel_size: int,
        mlp_hidden: int,
        dropout: float,
        pooling: list[str],
    ) -> None:
        super().__init__()
        if not conv_channels:
            raise ValueError("head.conv_channels must not be empty")

        self.pooling = list(pooling)
        layers: list[nn.Module] = []
        in_ch = embed_dim
        for i, out_ch in enumerate(conv_channels):
            if i == 0:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
            else:
                padding = kernel_size // 2
                layers.append(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
                )
            layers.append(nn.GELU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        pool_dim = in_ch * len(self.pooling)
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, C, H, W)
        x = self.conv(tokens)
        return self.mlp(_pool_feature_map(x, self.pooling)).squeeze(1)


def build_head(head_cfg: dict[str, Any], embed_dim: int) -> nn.Module:
    head_type = head_cfg.get("type", "structure")
    pooling = list(head_cfg["pooling"])

    if head_type == "gap_linear":
        return GapLinearHead(embed_dim=embed_dim, pooling=pooling)
    if head_type == "gap_mlp":
        return GapMlpHead(
            embed_dim=embed_dim,
            mlp_hidden=int(head_cfg["mlp_hidden"]),
            dropout=float(head_cfg.get("dropout", 0.2)),
            pooling=pooling,
        )
    if head_type == "structure":
        return StructureHead(
            embed_dim=embed_dim,
            conv_channels=[int(c) for c in head_cfg["conv_channels"]],
            kernel_size=int(head_cfg["kernel_size"]),
            mlp_hidden=int(head_cfg["mlp_hidden"]),
            dropout=float(head_cfg.get("dropout", 0.2)),
            pooling=pooling,
        )
    raise ValueError(f"Unknown head.type: {head_type}")


class DinoImageClassifier(nn.Module):
    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        head_cfg = cfg["head"]

        self.num_blocks = int(model_cfg["num_blocks"])
        self.block_index = self.num_blocks - 1
        self.freeze_dino = bool(model_cfg["freeze_dino"])

        repo = resolve_path(model_cfg["dino_repo"])
        weights = resolve_dino_weights(str(model_cfg.get("dino_weights", "")))
        self.backbone = load_dinov2(repo, weights, self.num_blocks, device)

        if self.freeze_dino:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.head = build_head(head_cfg, embed_dim=int(model_cfg["embed_dim"]))

    def extract_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_dino:
            with torch.no_grad():
                outputs = self.backbone.get_intermediate_layers(
                    images,
                    n=[self.block_index],
                    reshape=True,
                    norm=True,
                )
                return outputs[0]
        outputs = self.backbone.get_intermediate_layers(
            images,
            n=[self.block_index],
            reshape=True,
            norm=True,
        )
        return outputs[0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.extract_tokens(images)
        return self.head(tokens)


@torch.no_grad()
def evaluate(
    model: DinoImageClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += int((preds == labels.long()).sum().item())
        total += images.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def train_epoch(
    model: DinoImageClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    if model.freeze_dino:
        model.backbone.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += int((preds == labels.long()).sum().item())
        total += images.size(0)

    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: DinoImageClassifier,
    cfg: dict[str, Any],
    epoch: int,
    val_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_accuracy": val_acc,
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        path,
    )


def run_train(cfg: dict[str, Any]) -> None:
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]

    device = torch.device(
        train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(int(train_cfg["seed"]))

    data_root = resolve_path(data_cfg["root"])
    train_dir = data_root / data_cfg["train_dir"]
    val_dir = data_root / data_cfg["val_dir"]

    image_size = int(data_cfg["image_size"])
    train_ds = ClassFolderDataset(
        train_dir,
        transform=build_transforms(image_size, augment=bool(data_cfg["augment"])),
    )
    val_ds = ClassFolderDataset(
        val_dir,
        transform=build_transforms(image_size, augment=False),
    )

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DinoImageClassifier(cfg, device).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    run_dir = resolve_path(out_cfg["dir"]) / out_cfg["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    best_acc = -1.0
    epochs = int(train_cfg["epochs"])

    print(f"Device: {device}")
    print(f"Train images: {len(train_ds)} from {train_dir}")
    print(f"Val images: {len(val_ds)} from {val_dir}")
    print(f"Output: {run_dir}")
    print(f"DINO blocks: {cfg['model']['num_blocks']}")
    print(f"Pooling: {cfg['head']['pooling']}")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.perf_counter() - t0

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        save_checkpoint(run_dir / "last.pt", model, cfg, epoch, val_acc)
        if val_acc >= best_acc:
            best_acc = val_acc
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, val_acc)

    writer.close()
    summary = {
        "best_val_accuracy": best_acc,
        "epochs": epochs,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Best val accuracy: {best_acc:.4f}")
    print(f"Checkpoints: {run_dir / 'best.pt'}")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[DinoImageClassifier, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", cfg)
    model = DinoImageClassifier(ckpt_cfg, device).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_cfg


@torch.no_grad()
def run_predict(cfg: dict[str, Any], image_path: Path, checkpoint_path: Path) -> None:
    device = torch.device(
        cfg["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, ckpt_cfg = load_model_from_checkpoint(checkpoint_path, cfg, device)

    image_size = int(ckpt_cfg["data"]["image_size"])
    transform = build_transforms(image_size, augment=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logit = model(tensor).item()
    prob = float(torch.sigmoid(torch.tensor(logit)).item())
    pred = 1 if prob >= 0.5 else 0

    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prediction: {pred} ({'uniform' if pred == 1 else 'fragmented'})")
    print(f"Probability(class=1): {prob:.4f}")


def _error_tag(true_label: int, pred_label: int) -> str:
    if true_label == 0 and pred_label == 1:
        return "FP"
    if true_label == 1 and pred_label == 0:
        return "FN"
    raise ValueError(f"Not a misclassification: true={true_label}, pred={pred_label}")


def _error_filename(
    tag: str,
    true_label: int,
    pred_label: int,
    prob: float,
    source_path: Path,
) -> str:
    return (
        f"{tag}_true{true_label}_pred{pred_label}_prob{prob:.4f}_{source_path.name}"
    )


@torch.no_grad()
def run_eval_val(
    cfg: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
) -> None:
    """Run inference on the validation set; save only misclassified images."""
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    device = torch.device(
        train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_root = resolve_path(data_cfg["root"])
    val_dir = data_root / data_cfg["val_dir"]
    val_ds = ClassFolderDataset(
        val_dir,
        transform=build_transforms(int(data_cfg["image_size"]), augment=False),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model, ckpt_cfg = load_model_from_checkpoint(checkpoint_path, cfg, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    fp_count = 0
    fn_count = 0
    errors: list[dict[str, Any]] = []

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Val images: {len(val_ds)} from {val_dir}")
    print(f"Saving misclassifications to: {output_dir}")

    for images, labels, paths in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        for i in range(images.size(0)):
            true_label = int(labels[i].item())
            pred_label = int(preds[i].item())
            prob = float(probs[i].item())
            src = Path(paths[i])
            total += 1

            if pred_label == true_label:
                correct += 1
                continue

            tag = _error_tag(true_label, pred_label)
            if tag == "FP":
                fp_count += 1
            else:
                fn_count += 1

            dest_name = _error_filename(tag, true_label, pred_label, prob, src)
            dest_path = output_dir / dest_name
            shutil.copy2(src, dest_path)

            errors.append(
                {
                    "tag": tag,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "prob_class_1": prob,
                    "source": str(src),
                    "saved_as": str(dest_path),
                }
            )

    accuracy = correct / total if total else 0.0
    summary = {
        "checkpoint": str(checkpoint_path),
        "val_dir": str(val_dir),
        "output_dir": str(output_dir),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "errors": total - correct,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "misclassified": errors,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Misclassified: {total - correct} (FP={fp_count}, FN={fn_count})")
    print(f"Summary: {output_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINO image-level binary classifier (0=fragmented, 1=uniform)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train on train/val class folders.")
    train_p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"YAML config path (default: {DEFAULT_CONFIG}).",
    )
    train_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value, e.g. --set model.num_blocks=6",
    )

    pred_p = sub.add_parser("predict", help="Predict class for one image.")
    pred_p.add_argument("--config", default=str(DEFAULT_CONFIG))
    pred_p.add_argument("--image", required=True, help="Input image path.")
    pred_p.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt).")
    pred_p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    eval_p = sub.add_parser(
        "eval-val",
        help="Run on validation set; save only misclassified images (FP/FN).",
    )
    eval_p.add_argument("--config", default=str(DEFAULT_CONFIG))
    eval_p.add_argument(
        "--checkpoint",
        default=str(_ROOT / "outputs/dino_image_classifier/frozen_backbone/best.pt"),
        help="Model checkpoint (.pt).",
    )
    eval_p.add_argument(
        "--output-dir",
        default=str(_ROOT / "outputs/dino_image_classifier/frozen_backbone/val_errors"),
        help="Folder for misclassified validation images.",
    )
    eval_p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_path(args.config), overrides=args.set)

    if args.command == "train":
        run_train(cfg)
    elif args.command == "predict":
        run_predict(
            cfg,
            resolve_path(args.image),
            resolve_path(args.checkpoint),
        )
    elif args.command == "eval-val":
        run_eval_val(
            cfg,
            resolve_path(args.checkpoint),
            resolve_path(args.output_dir),
        )


if __name__ == "__main__":
    main()
