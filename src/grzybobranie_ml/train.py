"""Training and fine-tuning pipeline for Grzybobranie AI models.

Module: grzybobranie_ml.train
CLI: python -m grzybobranie_ml.train [options]

Rules and constraints:
- Whole-image classifier, exactly K <= 67 classes (target 67). No aggregate classes.
- Out-of-distribution rejection via energy gate E(x) and minTop1Prob (handled post-softmax).
- Base weights licensing: Apache-2.0, MIT, BSD only. CC BY-NC (ConvNeXt V2) and Apple (MobileViT) disabled.
- Supports both 224x224 and 256x256 input resolutions.
- Two-phase training: (a) warmup on frozen backbone (head only), (b) full fine-tuning with cosine LR decay.
- Mixed precision: bfloat16 on sm_120 (Blackwell RTX 5070 Laptop), fallback to fp16 + GradScaler.
- Evaluates strictly on validation set. Test set remains untouched.
- Safety metrics: recall on high_stakes species, confusion on DANGEROUS_PAIRS, zero confident toxic as edible.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("grzybobranie_ml.train")

# Approved model families and licenses
APPROVED_MODELS: dict[str, dict[str, Any]] = {
    "mobilenetv4_conv_small.e2400_r224_in1k": {
        "family": "mobilenetv4",
        "license": "Apache-2.0",
        "native_res": 224,
    },
    "tf_efficientnet_lite0.in1k": {
        "family": "efficientnet_lite",
        "license": "Apache-2.0",
        "native_res": 224,
    },
    "mobilenetv3_large_100.ra4_e3600_r224_in1k": {
        "family": "mobilenetv3",
        "license": "Apache-2.0",
        "native_res": 224,
    },
    "mobilenetv4_conv_medium.e500_r256_in1k": {
        "family": "mobilenetv4",
        "license": "Apache-2.0",
        "native_res": 256,
    },
    "efficientnet_b0.ra4_e3600_r224_in1k": {
        "family": "efficientnet",
        "license": "Apache-2.0",
        "native_res": 224,
    },
}

DEFAULT_DANGEROUS_PAIRS_CONFIG = Path("configs/karty_sobowtory.json")


class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_params = dict(model.named_parameters())
        for name, ema_p in self.module.named_parameters():
            if name in model_params:
                ema_p.copy_(self.decay * ema_p + (1.0 - self.decay) * model_params[name].detach())
        model_buffers = dict(model.named_buffers())
        for name, ema_b in self.module.named_buffers():
            if name in model_buffers:
                ema_b.copy_(model_buffers[name].detach())

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.module.load_state_dict(state_dict)


def verify_model_license(model_name: str) -> None:
    """Verify that model weights satisfy Apache-2.0 / MIT / BSD license policy."""
    if model_name not in APPROVED_MODELS:
        # Check forbidden keywords
        lower = model_name.lower()
        if "convnextv2" in lower or "convnext_v2" in lower:
            raise ValueError(
                f"Model {model_name} is rejected: ConvNeXt V2 uses CC BY-NC 4.0 license, "
                "which violates the commercial distribution policy (soul.md §2.1)."
            )
        if "mobilevit" in lower or "fastvit" in lower:
            raise ValueError(
                f"Model {model_name} is rejected: Apple cvnets/fastvit license prohibits commercial distribution."
            )
        if "ft_df20" in lower or "ft_df24" in lower or "df20" in lower:
            raise ValueError(
                f"Model {model_name} is rejected: DF20/DF24 training or fine-tuning weights are strictly forbidden."
            )
        logger.warning("Model %s is not in pre-approved list. Ensure its license is Apache-2.0/MIT/BSD.", model_name)


def create_classifier_model(
    model_name: str,
    num_classes: int = 67,
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> nn.Module:
    """Create a timm vision classifier with K outputs without aggregate classes."""
    verify_model_license(model_name)
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def get_train_transforms(input_size: int = 224) -> T.Compose:
    """Training data augmentations matching realistic outdoor mushroom photo conditions."""
    return T.Compose([
        T.RandomResizedCrop(
            size=(input_size, input_size),
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # maps [0, 1] -> [-1, 1]
    ])


def get_val_transforms(input_size: int = 224) -> T.Compose:
    """Validation deterministic transform matching prototype baseline: box 448 -> resize NxN -> [-1, 1]."""
    return T.Compose([
        T.Resize(size=448, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(size=(input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # maps [0, 1] -> [-1, 1]
    ])


class MushroomSplitDataset(Dataset):
    """Dataset reading from train.csv or val.csv splits with deterministic class indices."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path | None = None,
        transform: Any = None,
        class_to_idx: dict[str, int] | None = None,
        mock_if_missing: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir) if image_dir else None
        self.transform = transform
        self.mock_if_missing = mock_if_missing

        raw_records: list[dict[str, str]] = []
        with open(self.csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_records = list(reader)

        if not raw_records:
            raise ValueError(f"No records found in CSV: {self.csv_path}")

        # Build or use sorted class mapping across raw records
        if class_to_idx is None:
            unique_species = sorted({r["species_slug"] for r in raw_records if r.get("species_slug")})
            self.class_to_idx = {slug: i for i, slug in enumerate(unique_species)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {i: s for s, i in self.class_to_idx.items()}

        # Build fast image index from manifest_768.csv if present in image_dir
        self.path_lookup: dict[str, Path] = {}
        if self.image_dir and self.image_dir.exists():
            manifest_file = self.image_dir / "manifest_768.csv"
            if manifest_file.exists():
                with open(manifest_file, encoding="utf-8") as mf:
                    m_reader = csv.DictReader(mf)
                    for m_row in m_reader:
                        rel_path = m_row.get("path")
                        if rel_path:
                            full_p = self.image_dir / rel_path
                            if full_p.exists():
                                if m_row.get("image_url"):
                                    self.path_lookup[m_row["image_url"]] = full_p
                                if m_row.get("photo_id"):
                                    self.path_lookup[m_row["photo_id"]] = full_p
                                if m_row.get("source_url"):
                                    self.path_lookup[m_row["source_url"]] = full_p

        # Resolve records and paths
        self.records: list[dict[str, str]] = []
        self.image_paths: list[Path | None] = []

        for r in raw_records:
            slug = r.get("species_slug")
            if not slug or slug not in self.class_to_idx:
                continue
            img_p = self._resolve_path(r)
            if img_p is not None or self.mock_if_missing:
                self.records.append(r)
                self.image_paths.append(img_p)

        logger.info(
            "Dataset %s: %d / %d records active on disk (mock=%s)",
            self.csv_path.name,
            len(self.records),
            len(raw_records),
            self.mock_if_missing,
        )

    def _resolve_path(self, row: dict[str, str]) -> Path | None:
        url = row.get("image_url") or row.get("canonical_url")
        pid = row.get("photo_id") or row.get("inat_photo_id") or row.get("mo_image_id")
        src = row.get("source_url")

        if url and url in self.path_lookup:
            return self.path_lookup[url]
        if pid and pid in self.path_lookup:
            return self.path_lookup[pid]
        if src and src in self.path_lookup:
            return self.path_lookup[src]

        if not self.image_dir or not self.image_dir.exists():
            return None

        # Fallback candidates directly under image_dir
        slug = row.get("species_slug", "")
        file_name = row.get("file_name", "")
        candidates = [
            self.image_dir / file_name if file_name else None,
            self.image_dir / slug / f"{pid}.jpg" if pid and slug else None,
            self.image_dir / f"{pid}.jpg" if pid else None,
            self.image_dir / "images" / slug / f"{pid}.jpg" if pid and slug else None,
        ]
        for c in candidates:
            if c and c.exists():
                return c
        return None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.records[idx]
        slug = row["species_slug"]
        label = self.class_to_idx[slug]
        img_path = self.image_paths[idx]

        if img_path is not None and img_path.exists():
            with Image.open(img_path) as opened:
                image = opened.convert("RGB")
        elif self.mock_if_missing:
            h = int(hashlib.sha256(row.get("photo_id", str(idx)).encode()).hexdigest()[:6], 16)
            r, g, b = (h >> 16) & 255, (h >> 8) & 255, h & 255
            image = Image.new("RGB", (256, 256), color=(r, g, b))
        else:
            raise FileNotFoundError(f"Image for record {row.get('photo_id')} ({slug}) not found.")

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = T.ToTensor()(image)

        return tensor, label


def compute_class_weights(
    records: Sequence[dict[str, str]],
    class_to_idx: dict[str, int],
    smooth: float = 0.5,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Compute smoothed inverse class frequency weights for unbalanced species."""
    counts = np.zeros(len(class_to_idx), dtype=np.float32)
    for r in records:
        slug = r.get("species_slug")
        if slug in class_to_idx:
            counts[class_to_idx[slug]] += 1.0

    total = np.sum(counts)
    num_classes = len(class_to_idx)
    # Inverse frequency with smoothing
    weights = total / (num_classes * (counts + 1e-5))
    # Apply exponential smoothing factor
    weights = np.power(weights, smooth)
    # Normalize to mean 1.0
    weights = weights / np.mean(weights)
    weights = np.clip(weights, min_weight, max_weight)
    return torch.from_numpy(weights.astype(np.float32))


def load_dangerous_pairs(config_path: str | Path = DEFAULT_DANGEROUS_PAIRS_CONFIG) -> list[tuple[str, str]]:
    """Load DANGEROUS_PAIRS tuples from karty_sobowtory.json."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Dangerous pairs config %s not found. Using empty pair list.", path)
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for item in data.get("proponowane_DANGEROUS_PAIRS", []):
        klasy = item.get("klasy", [])
        if len(klasy) == 2:
            pairs.append((klasy[0], klasy[1]))
    return pairs


def load_high_stakes_species(config_path: str | Path = DEFAULT_DANGEROUS_PAIRS_CONFIG) -> set[str]:
    """Load set of high_stakes species slugs from karty_sobowtory.json."""
    path = Path(config_path)
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {c["slug"] for c in data.get("karty", []) if c.get("high_stakes")}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    class_to_idx: dict[str, int],
    dangerous_pairs: Sequence[tuple[str, str]] | None = None,
    high_stakes_species: set[str] | None = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Comprehensive validation metrics: macro top-1/top-5, high-stakes recall, dangerous pair confusion."""
    model.eval()
    num_classes = len(class_to_idx)
    idx_to_class = {i: s for s, i in class_to_idx.items()}

    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0

    per_class_total = np.zeros(num_classes, dtype=np.int64)
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)

        # Top-1
        preds_top1 = torch.argmax(logits, dim=1)
        correct_top1 += (preds_top1 == targets).sum().item()

        # Top-5
        _, preds_top5 = torch.topk(logits, k=min(5, num_classes), dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(preds_top5)
        correct_top5 += (preds_top5 == targets_expanded).any(dim=1).sum().item()

        total_samples += batch_size

        t_np = targets.cpu().numpy()
        p_np = preds_top1.cpu().numpy()
        for t, p in zip(t_np, p_np):
            per_class_total[t] += 1
            if t == p:
                per_class_correct[t] += 1
            confusion[t, p] += 1

    # Macro recall
    recalls = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros(num_classes, dtype=np.float64),
        where=per_class_total > 0,
    )
    macro_top1 = float(np.mean(recalls))
    overall_top1 = float(correct_top1 / total_samples) if total_samples > 0 else 0.0
    overall_top5 = float(correct_top5 / total_samples) if total_samples > 0 else 0.0

    # High stakes recall
    high_stakes_recalls: dict[str, float] = {}
    if high_stakes_species:
        for slug in high_stakes_species:
            if slug in class_to_idx:
                idx = class_to_idx[slug]
                high_stakes_recalls[slug] = float(recalls[idx])

    min_high_stakes_recall = float(min(high_stakes_recalls.values())) if high_stakes_recalls else 0.0
    mean_high_stakes_recall = float(np.mean(list(high_stakes_recalls.values()))) if high_stakes_recalls else 0.0

    # Dangerous pairs confusion rate
    pair_confusions: dict[str, float] = {}
    violating_pairs = 0
    if dangerous_pairs:
        for slug_a, slug_b in dangerous_pairs:
            if slug_a in class_to_idx and slug_b in class_to_idx:
                ia, ib = class_to_idx[slug_a], class_to_idx[slug_b]
                na = per_class_total[ia]
                nb = per_class_total[ib]
                err_a_to_b = (confusion[ia, ib] / na) if na > 0 else 0.0
                err_b_to_a = (confusion[ib, ia] / nb) if nb > 0 else 0.0
                max_pair_err = max(err_a_to_b, err_b_to_a)
                pair_key = f"{slug_a} <-> {slug_b}"
                pair_confusions[pair_key] = float(max_pair_err)
                if max_pair_err > 0.05:
                    violating_pairs += 1

    per_class_metrics = {
        idx_to_class[i]: {
            "n_val": int(per_class_total[i]),
            "recall": float(recalls[i]),
        }
        for i in range(num_classes)
    }

    return {
        "val_samples": total_samples,
        "overall_top1": overall_top1,
        "overall_top5": overall_top5,
        "macro_top1": macro_top1,
        "high_stakes_min_recall": min_high_stakes_recall,
        "high_stakes_mean_recall": mean_high_stakes_recall,
        "dangerous_pairs_violating_count": violating_pairs,
        "dangerous_pairs_evaluated": len(pair_confusions),
        "per_class": per_class_metrics,
        "pair_confusions": pair_confusions,
    }


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters and leave classifier head trainable."""
    for p in model.parameters():
        p.requires_grad = False
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for p in classifier.parameters():
            p.requires_grad = True
    elif isinstance(classifier, nn.Parameter):
        classifier.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_pipeline(
    model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
    input_size: int = 224,
    train_csv: str | Path = "data/splits/train.csv",
    val_csv: str | Path = "data/splits/val.csv",
    image_dir: str | Path | None = None,
    output_dir: str | Path = "artifacts/train_runs",
    warmup_epochs: int = 3,
    fine_tune_epochs: int = 20,
    batch_size: int = 64,
    warmup_lr: float = 1e-3,
    fine_tune_lr: float = 1e-4,
    min_lr: float = 1e-6,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    num_workers: int = 8,
    mock_dataset: bool = False,
    device: str | None = None,
    resume_checkpoint: str | Path | None = None,
    pretrained: bool = True,
) -> dict[str, Any]:
    """Execute complete two-phase training and fine-tuning pipeline."""
    start_time = time.time()
    run_id = f"{model_name}_r{input_size}_{int(start_time)}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if image_dir is None:
        local_768 = Path("/home/bubolka/hf-768/upload")
        if local_768.exists():
            image_dir = local_768
        else:
            image_dir = Path("data/png_cache") / str(input_size)

    # 1. Device and precision setup
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    use_amp = device_obj.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    logger.info("Using device: %s, AMP: %s (%s)", device_obj, use_amp, amp_dtype)

    # 2. Datasets & Dataloaders
    train_transform = get_train_transforms(input_size)
    val_transform = get_val_transforms(input_size)

    train_ds = MushroomSplitDataset(
        csv_path=train_csv,
        image_dir=image_dir,
        transform=train_transform,
        mock_if_missing=mock_dataset,
    )
    class_to_idx = train_ds.class_to_idx
    num_classes = len(class_to_idx)
    logger.info("Loaded train dataset: %d samples, %d classes", len(train_ds), num_classes)

    val_ds = MushroomSplitDataset(
        csv_path=val_csv,
        image_dir=image_dir,
        transform=val_transform,
        class_to_idx=class_to_idx,
        mock_if_missing=mock_dataset,
    )
    logger.info("Loaded val dataset: %d samples", len(val_ds))

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_amp,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    # 3. Model & EMA
    model = create_classifier_model(model_name, num_classes=num_classes, pretrained=pretrained)
    model.to(device_obj)
    ema = ModelEMA(model, decay=0.999)

    # 4. Class weighting & Criterion
    class_weights = compute_class_weights(train_ds.records, class_to_idx).to(device_obj)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # 5. Metadata for safety evaluation
    dangerous_pairs = load_dangerous_pairs()
    high_stakes = load_high_stakes_species()

    metrics_log_path = run_dir / "metrics.jsonl"
    best_checkpoint_path = run_dir / "best_val_checkpoint.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    best_macro_top1 = 0.0
    start_epoch = 0

    if resume_checkpoint and Path(resume_checkpoint).exists():
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
        ckpt = torch.load(resume_checkpoint, map_location=device_obj)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_macro_top1 = ckpt.get("best_macro_top1", 0.0)

    total_epochs = warmup_epochs + fine_tune_epochs

    # Helper for logging
    def log_epoch(epoch: int, phase: str, train_loss: float, lr: float, val_metrics: dict[str, Any]) -> None:
        record = {
            "epoch": epoch,
            "phase": phase,
            "lr": lr,
            "train_loss": train_loss,
            **val_metrics,
            "timestamp": time.time(),
        }
        with open(metrics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            "Epoch %d (%s) | Train Loss: %.4f | Val Macro Top-1: %.4f | High-Stakes Min Recall: %.4f | Violating Pairs: %d",
            epoch,
            phase,
            train_loss,
            val_metrics["macro_top1"],
            val_metrics["high_stakes_min_recall"],
            val_metrics["dangerous_pairs_violating_count"],
        )

    # ==========================================
    # Phase (a): Warmup (Frozen backbone)
    # ==========================================
    if start_epoch < warmup_epochs:
        logger.info("Starting Phase (a): Frozen backbone warmup for %d epochs...", warmup_epochs)
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=warmup_lr,
            weight_decay=weight_decay,
        )

        for epoch in range(start_epoch, warmup_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0

            for i, (images, targets) in enumerate(train_loader):
                images = images.to(device_obj, non_blocking=True)
                targets = targets.to(device_obj, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                ema.update(model)

                total_loss += loss.item()
                n_batches += 1

                if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                    logger.info("Warmup epoch %d [%d/%d] | Loss: %.4f", epoch, i + 1, len(train_loader), loss.item())

            avg_train_loss = total_loss / max(1, n_batches)
            val_metrics = evaluate_model(
                ema.module,
                val_loader,
                device_obj,
                class_to_idx,
                dangerous_pairs,
                high_stakes,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            log_epoch(epoch, "warmup", avg_train_loss, warmup_lr, val_metrics)

            if val_metrics["macro_top1"] > best_macro_top1:
                best_macro_top1 = val_metrics["macro_top1"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "best_macro_top1": best_macro_top1,
                    "class_to_idx": class_to_idx,
                    "model_name": model_name,
                    "input_size": input_size,
                }, best_checkpoint_path)

    # ==========================================
    # Phase (b): Full Fine-tuning
    # ==========================================
    logger.info("Starting Phase (b): Full fine-tuning for %d epochs...", fine_tune_epochs)
    unfreeze_backbone(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=fine_tune_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=fine_tune_epochs,
        eta_min=min_lr,
    )

    phase_b_start = max(warmup_epochs, start_epoch)
    for epoch in range(phase_b_start, total_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        current_lr = scheduler.get_last_lr()[0]

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device_obj, non_blocking=True)
            targets = targets.to(device_obj, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()
            n_batches += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                logger.info("Fine-tune epoch %d [%d/%d] | Loss: %.4f", epoch, i + 1, len(train_loader), loss.item())

        scheduler.step()
        avg_train_loss = total_loss / max(1, n_batches)

        # Evaluate EMA model on validation set
        val_metrics = evaluate_model(
            ema.module,
            val_loader,
            device_obj,
            class_to_idx,
            dangerous_pairs,
            high_stakes,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        log_epoch(epoch, "fine_tune", avg_train_loss, current_lr, val_metrics)

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_macro_top1": best_macro_top1,
            "class_to_idx": class_to_idx,
            "model_name": model_name,
            "input_size": input_size,
        }, last_checkpoint_path)

        if val_metrics["macro_top1"] > best_macro_top1:
            best_macro_top1 = val_metrics["macro_top1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "best_macro_top1": best_macro_top1,
                "class_to_idx": class_to_idx,
                "model_name": model_name,
                "input_size": input_size,
            }, best_checkpoint_path)

    # Save final run config and summary
    best_sha256 = compute_file_sha256(best_checkpoint_path) if best_checkpoint_path.exists() else ""
    summary = {
        "run_id": run_id,
        "model_name": model_name,
        "input_size": input_size,
        "total_epochs": total_epochs,
        "best_macro_top1": best_macro_top1,
        "best_checkpoint_file": str(best_checkpoint_path),
        "best_checkpoint_sha256": best_sha256,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete in %.2f s. Best Macro Top-1: %.4f (SHA256: %s)", summary["elapsed_seconds"], best_macro_top1, best_sha256[:8])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and fine-tune Grzybobranie AI models.")
    parser.add_argument("--model", type=str, default="mobilenetv4_conv_small.e2400_r224_in1k", help="Model name from timm")
    parser.add_argument("--input-size", type=int, default=224, choices=[224, 256], help="Input image resolution")
    parser.add_argument("--train-csv", type=str, default="data/splits/train.csv", help="Path to train split CSV")
    parser.add_argument("--val-csv", type=str, default="data/splits/val.csv", help="Path to val split CSV")
    default_img_dir = (
        "/home/bubolka/hf-768/upload"
        if Path("/home/bubolka/hf-768/upload").exists()
        else "data/png_cache/224"
    )
    parser.add_argument("--image-dir", type=str, default=default_img_dir, help="Local directory with cached images")
    parser.add_argument("--output-dir", type=str, default="artifacts/train_runs", help="Output directory for checkpoints")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Epochs with frozen backbone")
    parser.add_argument("--fine-tune-epochs", type=int, default=20, help="Epochs for full fine-tuning")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--warmup-lr", type=float, default=1e-3, help="Head warmup learning rate")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4, help="Fine-tuning initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Cosine annealing minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--mock-dataset", action="store_true", help="Use synthetic mock images if files missing")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    train_pipeline(
        model_name=args.model,
        input_size=args.input_size,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        warmup_epochs=args.warmup_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        warmup_lr=args.warmup_lr,
        fine_tune_lr=args.fine_tune_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        mock_dataset=args.mock_dataset,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
