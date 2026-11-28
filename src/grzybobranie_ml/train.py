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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from grzybobranie_ml.license_policy import normalize_license
from grzybobranie_ml.preprocess import baseline_preprocessing, preprocess
from grzybobranie_ml.split import canonical_url

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
    "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k": {
        "family": "mobilenetv4_hybrid",
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
    """Exponential Moving Average of model parameters with warmup decay ramp."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        """Synchronize EMA parameters and buffers directly to current model state."""
        self.updates = 0
        self.module.load_state_dict(copy.deepcopy(model.state_dict()))
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        eff_decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        model_params = dict(model.named_parameters())
        for name, ema_p in self.module.named_parameters():
            if name in model_params:
                ema_p.copy_(eff_decay * ema_p + (1.0 - eff_decay) * model_params[name].detach())
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


class P1ReferencePreprocessTransform:
    """Picklable P1 manifest reference transform (EXIF -> box 448 -> box NxN -> half_up_uint8 -> [-1, 1])."""

    def __init__(self, input_size: int = 256) -> None:
        self.input_size = int(input_size)
        self.spec = baseline_preprocessing(self.input_size)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        out = preprocess(image, self.spec, layout="CHW")["float"]
        return torch.from_numpy(np.ascontiguousarray(out))


def resolve_val_crop_mode(crop_mode: str) -> str:
    """Map a train/unified crop_mode string to a valid get_val_transforms mode."""
    if crop_mode == "legacy_rrc":
        return "legacy_448_center"
    return crop_mode


def get_train_transforms(
    input_size: int = 256,
    crop_mode: str = "none_stretch",
    saturation_jitter: float = 0.05,
) -> T.Compose:
    """Training data augmentations aligned with full-frame none_stretch and conservative color jitter."""
    if crop_mode in {"none_stretch", "p1_box_none_stretch", "bilinear_none_stretch"}:
        resize_op: Any = T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR)
        geom_aug: Any = T.RandomAffine(
            degrees=15,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            interpolation=T.InterpolationMode.BILINEAR,
        )
    elif crop_mode in {"legacy_rrc", "legacy_448_center"}:
        resize_op = T.RandomResizedCrop(
            size=(input_size, input_size),
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15),
            interpolation=T.InterpolationMode.BILINEAR,
        )
        geom_aug = T.RandomRotation(degrees=15)
    else:
        raise ValueError(f"Unsupported train crop_mode: {crop_mode!r}")

    return T.Compose([
        resize_op,
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=saturation_jitter),
        geom_aug,
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # maps [0, 1] -> [-1, 1]
    ])


def get_val_transforms(input_size: int = 256, crop_mode: str = "none_stretch") -> T.Compose:
    """Validation deterministic transform.

    Default ``none_stretch`` matches the P1 manifest contract:
    EXIF -> to_width 448 (box) -> to_size NxN (box) -> half_up_uint8 -> /127.5 - 1.
    """
    if crop_mode in {"none_stretch", "p1_box_none_stretch"}:
        return T.Compose([P1ReferencePreprocessTransform(input_size)])
    if crop_mode == "bilinear_none_stretch":
        return T.Compose([
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    if crop_mode in {"center_256", "center"}:
        return T.Compose([
            T.Resize(size=input_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size=(input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    if crop_mode == "legacy_448_center":
        return T.Compose([
            T.Resize(size=448, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size=(input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    raise ValueError(f"Unsupported val crop_mode: {crop_mode!r}")


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
                oriented = ImageOps.exif_transpose(opened)
                image = (oriented if oriented is not None else opened).convert("RGB")
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


def load_edible_and_toxic_species(
    config_path: str | Path = DEFAULT_DANGEROUS_PAIRS_CONFIG,
) -> tuple[set[str], set[str]]:
    """Load edible and toxic/deadly species slug sets from karty_sobowtory.json."""
    path = Path(config_path)
    if not path.exists():
        return set(), set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    edible: set[str] = set()
    toxic: set[str] = set()
    for card in data.get("karty", []):
        slug = card.get("slug")
        status = str(card.get("status_jadalnosci", "")).lower()
        if not slug:
            continue
        if status == "jadalny":
            edible.add(slug)
        if status in {"trujący", "śmiertelny", "smiertelnie trujacy"} or "toksycz" in status:
            toxic.add(slug)
    return edible, toxic


class SafetyCrossEntropyLoss(nn.Module):
    """Weighted label-smoothed CE with asymmetric penalty for toxic -> edible errors.

    Edible targets receive zero safety penalty so they are never pushed into false alarms.
    Toxic targets incur an extra penalty proportional to `-log(1 - p_danger)` on their
    edible look-alike counterparts from ``DANGEROUS_PAIRS`` (weight 1.0) and optionally
    other edible species (``general_toxic_to_edible_weight``).
    """

    def __init__(
        self,
        class_to_idx: Mapping[str, int],
        dangerous_pairs: Sequence[tuple[str, str]],
        edible_slugs: set[str],
        toxic_slugs: set[str],
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
        safety_lambda: float = 0.5,
        general_toxic_to_edible_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if safety_lambda < 0.0:
            raise ValueError("safety_lambda must be >= 0.0")
        if general_toxic_to_edible_weight < 0.0:
            raise ValueError("general_toxic_to_edible_weight must be >= 0.0")
        self.label_smoothing = float(label_smoothing)
        self.safety_lambda = float(safety_lambda)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights: torch.Tensor | None = None

        num_classes = len(class_to_idx)
        danger_mask = torch.zeros((num_classes, num_classes), dtype=torch.float32)

        edible_indices = [class_to_idx[s] for s in edible_slugs if s in class_to_idx]
        toxic_indices = [class_to_idx[s] for s in toxic_slugs if s in class_to_idx]

        if general_toxic_to_edible_weight > 0.0:
            for t_idx in toxic_indices:
                for e_idx in edible_indices:
                    danger_mask[t_idx, e_idx] = max(
                        danger_mask[t_idx, e_idx].item(),
                        float(general_toxic_to_edible_weight),
                    )

        for a, b in dangerous_pairs:
            if a not in class_to_idx or b not in class_to_idx:
                continue
            if a in toxic_slugs and b in edible_slugs:
                danger_mask[class_to_idx[a], class_to_idx[b]] = 1.0
            if b in toxic_slugs and a in edible_slugs:
                danger_mask[class_to_idx[b], class_to_idx[a]] = 1.0

        self.register_buffer("danger_mask", danger_mask)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        weights = self.class_weights.to(logits_f.device) if self.class_weights is not None else None
        ce = F.cross_entropy(
            logits_f,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )
        if self.safety_lambda <= 0.0:
            return ce

        probs = F.softmax(logits_f, dim=-1)
        target_mask = self.danger_mask[targets]
        weighted_danger = torch.sum(probs * target_mask, dim=-1)
        clamped = torch.clamp(weighted_danger, min=0.0, max=1.0 - 1e-6)
        safety_penalty = -torch.log(1.0 - clamped)
        return ce + self.safety_lambda * safety_penalty.mean()


class EnergyBoundedOELoss(nn.Module):
    """Liu et al. (2020) energy-bounded Outlier Exposure margin loss."""

    def __init__(
        self,
        margin_in: float = -3.5,
        margin_out: float = -1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if margin_in >= margin_out:
            raise ValueError("margin_in must be strictly less than margin_out")
        self.margin_in = float(margin_in)
        self.margin_out = float(margin_out)
        self.temperature = float(temperature)

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        return -self.temperature * torch.logsumexp(logits.float() / self.temperature, dim=-1)

    @staticmethod
    def _smooth_margin_hinge(violation: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
        """Quadratic penalty for small violations <= beta, linear beyond beta to prevent gradient spikes."""
        pos = F.relu(violation)
        return torch.where(pos <= beta, 0.5 * pos.pow(2), beta * (pos - 0.5 * beta))

    def forward(self, in_logits: torch.Tensor, oe_logits: torch.Tensor) -> torch.Tensor:
        e_in = self.compute_energy(in_logits)
        e_out = self.compute_energy(oe_logits)
        loss_in = self._smooth_margin_hinge(e_in - self.margin_in).mean()
        loss_out = self._smooth_margin_hinge(self.margin_out - e_out).mean()
        return loss_in + loss_out


def _is_strict_cc0_license(raw_license: str, license_url: str | None = None) -> bool:
    """Return True strictly for CC0 / Public Domain Zero licenses."""
    norm = normalize_license(raw_license, license_url)
    if norm in {"cc0", "cc0-1.0"}:
        return True
    cleaned = raw_license.strip().lower().replace(" ", "-")
    return cleaned in {"cc0", "cc0-1.0", "cc0_1.0"}


def validate_oe_manifest_disjointness(
    oe_csv: str | Path,
    ood_ids_path: str | Path = "data/ood-v1/ood_ids.txt",
    excluded_split_csvs: Sequence[str | Path] = (
        "data/splits/test.csv",
        "data/splits/val.csv",
    ),
) -> int:
    """Verify an external CC0 Outlier Exposure CSV is 100% disjoint from OOD-v1 and test/val splits."""
    oe_path = Path(oe_csv).resolve()
    repo_ood_dir = Path("data/ood-v1").resolve()
    if oe_path.is_relative_to(repo_ood_dir) or "ood-v1" in oe_path.parts:
        raise ValueError(f"OE manifest {oe_csv} must be external to data/ood-v1/ (soul.md §3 pkt 5).")
    if not oe_path.exists():
        raise FileNotFoundError(f"OE manifest not found: {oe_path}")

    blocked_ids: set[str] = set()
    blocked_urls: set[str] = set()
    blocked_hashes: set[str] = set()

    id_keys = (
        "photo_id",
        "inat_photo_id",
        "observation_id",
        "inat_observation_id",
        "mo_observation_id",
        "mo_image_id",
        "gbif_occurrence_id",
        "commons_file",
    )
    url_keys = ("canonical_url", "image_url", "source_url")
    hash_keys = ("original_sha256", "sha256")

    ood_file = Path(ood_ids_path)
    extra_csvs: list[Path] = []
    if ood_file.exists():
        with open(ood_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "\t" not in line:
                    continue
                kind, val = line.split("\t", 1)
                val = val.strip()
                if kind in url_keys:
                    blocked_urls.add(canonical_url(val))
                else:
                    blocked_ids.add(val)
        for sib_name in ("ood_fungi.csv", "ood_nonfungi.csv"):
            sib = ood_file.parent / sib_name
            if sib.exists():
                extra_csvs.append(sib)

    for split_p in [*excluded_split_csvs, *extra_csvs]:
        sp = Path(split_p)
        if not sp.exists():
            continue
        with open(sp, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for key in id_keys:
                    v = (row.get(key) or "").strip()
                    if v:
                        blocked_ids.add(v)
                for uk in url_keys:
                    u = (row.get(uk) or "").strip()
                    if u:
                        blocked_urls.add(canonical_url(u))
                for hk in hash_keys:
                    hv = (row.get(hk) or "").strip().lower()
                    if hv:
                        blocked_hashes.add(hv)

    count = 0
    with open(oe_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            count += 1
            split_tag = (row.get("proposed_split") or "").strip().lower()
            if split_tag in {"calib", "test", "ood_calib", "ood_test"}:
                raise ValueError(f"OE manifest cannot contain OOD-v1 split row: {split_tag!r}")
            raw_lic = (row.get("normalized_license") or row.get("license") or "").strip()
            lic_url = (row.get("license_url") or "").strip() or None
            if not _is_strict_cc0_license(raw_lic, lic_url):
                raise ValueError(f"OE dataset must be strictly CC0-licensed, got {raw_lic!r}")
            for key in id_keys:
                v = (row.get(key) or "").strip()
                if v and v in blocked_ids:
                    raise ValueError(f"OE overlap detected on {key}={v}")
            for uk in url_keys:
                u = (row.get(uk) or "").strip()
                if u and canonical_url(u) in blocked_urls:
                    raise ValueError(f"OE overlap detected on URL={u}")
            for hk in hash_keys:
                hv = (row.get(hk) or "").strip().lower()
                if hv and hv in blocked_hashes:
                    raise ValueError(f"OE overlap detected on {hk}={hv}")

    if count == 0:
        raise ValueError(f"OE manifest {oe_path} is empty.")
    return count


class OutlierExposureDataset(Dataset):
    """Dataset reading external CC0 Outlier Exposure images for Energy-Bounded OE training."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path | None = None,
        transform: Any = None,
        mock_if_missing: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir) if image_dir else self.csv_path.parent
        self.transform = transform
        self.mock_if_missing = mock_if_missing

        with open(self.csv_path, encoding="utf-8") as f:
            raw_records = list(csv.DictReader(f))
        if not raw_records:
            raise ValueError(f"No records found in OE CSV: {self.csv_path}")

        self.records: list[dict[str, str]] = []
        self.image_paths: list[Path | None] = []
        for r in raw_records:
            resolved = self._resolve_path(r)
            if resolved is not None or self.mock_if_missing:
                self.records.append(r)
                self.image_paths.append(resolved)

        if not self.records:
            raise FileNotFoundError(
                f"No OE images resolved on disk from {self.csv_path} under {self.image_dir}."
            )

    def _resolve_path(self, row: dict[str, str]) -> Path | None:
        for key in ("local_path", "path", "file_name"):
            rel = (row.get(key) or "").strip()
            if rel:
                p = Path(rel)
                if not p.is_absolute():
                    p = self.image_dir / rel
                if p.exists():
                    return p
        pid = (row.get("photo_id") or "").strip()
        if pid:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                p = self.image_dir / f"{pid}{ext}"
                if p.exists():
                    return p
        return None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.records[idx]
        img_path = self.image_paths[idx]
        if img_path is not None and img_path.exists():
            with Image.open(img_path) as opened:
                oriented = ImageOps.exif_transpose(opened)
                image = (oriented if oriented is not None else opened).convert("RGB")
        elif self.mock_if_missing:
            h = int(hashlib.sha256(row.get("photo_id", str(idx)).encode()).hexdigest()[:6], 16)
            image = Image.new("RGB", (256, 256), color=((h >> 16) & 255, (h >> 8) & 255, h & 255))
        else:
            raise FileNotFoundError(f"OE image not found for record {row.get('photo_id', idx)}")

        tensor = self.transform(image) if self.transform is not None else T.ToTensor()(image)
        return tensor, 0


def load_uncovered_species_by_threshold(
    train_csv: str | Path,
    coverage_csv: str | Path = "configs/pokrycie-67.csv",
) -> set[str]:
    """Identify species below minimum training count thresholds (55 for high_stakes, 40 otherwise)."""
    cov_path = Path(coverage_csv)
    thresholds: dict[str, int] = {}
    if cov_path.exists():
        with open(cov_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                slug = (row.get("class") or "").strip()
                if not slug:
                    continue
                is_hs = str(row.get("high_stakes", "")).strip().lower() == "true"
                prog = int(row.get("prog") or (55 if is_hs else 40))
                thresholds[slug] = prog

    counts: dict[str, int] = {}
    with open(train_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slug = (row.get("species_slug") or "").strip()
            if slug:
                counts[slug] = counts.get(slug, 0) + 1

    uncovered: set[str] = set()
    for slug, cnt in counts.items():
        if cnt < thresholds.get(slug, 40):
            uncovered.add(slug)
    return uncovered


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


def _cycle_loader(loader: DataLoader):
    while True:
        yield from loader


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
    crop_mode: str = "none_stretch",
    saturation_jitter: float = 0.05,
    safety_lambda: float = 0.0,
    general_toxic_to_edible_weight: float = 0.25,
    oe_csv: str | Path | None = None,
    oe_image_dir: str | Path | None = None,
    oe_lambda: float = 0.0,
    oe_margin_in: float = -3.5,
    oe_margin_out: float = -1.0,
    oe_temperature: float = 1.0,
    oe_batch_size: int | None = None,
    exclude_uncovered: bool = False,
    coverage_csv: str | Path = "configs/pokrycie-67.csv",
    ood_ids_path: str | Path = "data/ood-v1/ood_ids.txt",
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
    val_crop_mode = resolve_val_crop_mode(crop_mode)
    train_transform = get_train_transforms(
        input_size=input_size,
        crop_mode=crop_mode,
        saturation_jitter=saturation_jitter,
    )
    val_transform = get_val_transforms(input_size=input_size, crop_mode=val_crop_mode)

    excluded_species: set[str] = set()
    if exclude_uncovered:
        excluded_species = load_uncovered_species_by_threshold(train_csv, coverage_csv=coverage_csv)
        logger.info("Excluding uncovered species below threshold: %s", sorted(excluded_species))

    explicit_class_to_idx: dict[str, int] | None = None
    if excluded_species:
        with open(train_csv, encoding="utf-8") as f:
            active_slugs = sorted({
                (r.get("species_slug") or "").strip()
                for r in csv.DictReader(f)
                if (r.get("species_slug") or "").strip()
                and (r.get("species_slug") or "").strip() not in excluded_species
            })
        explicit_class_to_idx = {slug: i for i, slug in enumerate(active_slugs)}

    train_ds = MushroomSplitDataset(
        csv_path=train_csv,
        image_dir=image_dir,
        transform=train_transform,
        class_to_idx=explicit_class_to_idx,
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
        drop_last=len(train_ds) >= batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    oe_criterion: EnergyBoundedOELoss | None = None
    oe_iter: Any = None
    if oe_csv is not None:
        if oe_lambda < 0.0:
            raise ValueError("oe_lambda must be >= 0.0")
        oe_count = validate_oe_manifest_disjointness(
            oe_csv,
            ood_ids_path=ood_ids_path,
            excluded_split_csvs=("data/splits/test.csv", val_csv),
        )
        logger.info("Validated disjoint CC0 OE manifest %s (%d rows)", oe_csv, oe_count)
        if oe_lambda > 0.0:
            oe_ds = OutlierExposureDataset(
                csv_path=oe_csv,
                image_dir=oe_image_dir,
                transform=train_transform,
                mock_if_missing=mock_dataset,
            )
            eff_oe_batch = min(len(oe_ds), oe_batch_size or max(1, batch_size // 2))
            oe_loader_kwargs: dict[str, Any] = {
                "batch_size": eff_oe_batch,
                "num_workers": min(4, num_workers),
                "pin_memory": use_amp,
            }
            if oe_loader_kwargs["num_workers"] > 0:
                oe_loader_kwargs["persistent_workers"] = True
                oe_loader_kwargs["prefetch_factor"] = 2
            oe_loader = DataLoader(
                oe_ds,
                shuffle=True,
                drop_last=len(oe_ds) >= eff_oe_batch,
                **oe_loader_kwargs,
            )
            oe_iter = _cycle_loader(oe_loader)
            oe_criterion = EnergyBoundedOELoss(
                margin_in=oe_margin_in,
                margin_out=oe_margin_out,
                temperature=oe_temperature,
            ).to(device_obj)

    # 3. Model & EMA
    model = create_classifier_model(model_name, num_classes=num_classes, pretrained=pretrained)
    model.to(device_obj)
    ema = ModelEMA(model, decay=0.999)

    # 4. Metadata for safety evaluation & Criterion
    dangerous_pairs = load_dangerous_pairs()
    high_stakes = load_high_stakes_species()
    edible_slugs, toxic_slugs = load_edible_and_toxic_species()

    class_weights = compute_class_weights(train_ds.records, class_to_idx).to(device_obj)
    if safety_lambda > 0.0:
        criterion: nn.Module = SafetyCrossEntropyLoss(
            class_to_idx=class_to_idx,
            dangerous_pairs=dangerous_pairs,
            edible_slugs=edible_slugs,
            toxic_slugs=toxic_slugs,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            safety_lambda=safety_lambda,
            general_toxic_to_edible_weight=general_toxic_to_edible_weight,
        ).to(device_obj)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    metrics_log_path = run_dir / "metrics.jsonl"
    best_checkpoint_path = run_dir / "best_val_checkpoint.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    best_macro_top1 = 0.0
    start_epoch = 0

    if resume_checkpoint and Path(resume_checkpoint).exists():
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
        ckpt = torch.load(resume_checkpoint, map_location=device_obj, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_macro_top1 = ckpt.get("best_macro_top1", 0.0)

    total_epochs = warmup_epochs + fine_tune_epochs

    def _compute_batch_loss(images: torch.Tensor, targets: torch.Tensor, *, apply_oe: bool = True) -> torch.Tensor:
        if apply_oe and oe_criterion is not None and oe_iter is not None:
            oe_images, _ = next(oe_iter)
            oe_images = oe_images.to(device_obj, non_blocking=True)
            # Run ID + OE together in train() mode so BatchNorm never divides OE by tiny stale running_var.
            all_inputs = torch.cat([images, oe_images], dim=0)
            all_outputs = model(all_inputs)
            b_id = images.size(0)
            outputs = all_outputs[:b_id]
            oe_outputs = all_outputs[b_id:]
            return criterion(outputs, targets) + oe_lambda * oe_criterion(outputs, oe_outputs)
        outputs = model(images)
        return criterion(outputs, targets)

    @torch.no_grad()
    def _refresh_ema_bn(max_batches: int = 150) -> None:
        """Align EMA BatchNorm running stats cleanly to EMA weights on pure mushroom batches."""
        if len(train_ds) < 2:
            return
        bn_layers: list[tuple[nn.modules.batchnorm._BatchNorm, float | None]] = []
        for mod in ema.module.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm) and mod.track_running_stats:
                bn_layers.append((mod, mod.momentum))
                mod.reset_running_stats()
                mod.momentum = None
        if not bn_layers:
            return
        ema.module.train()
        try:
            for b_idx, (id_imgs, _) in enumerate(train_loader):
                if b_idx >= max_batches:
                    break
                if id_imgs.size(0) < 2:
                    continue
                id_imgs = id_imgs.to(device_obj, non_blocking=True)
                with torch.amp.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=use_amp):
                    ema.module(id_imgs)
        finally:
            ema.module.eval()
            for mod, orig_mom in bn_layers:
                mod.momentum = orig_mom

    def _checkpoint_payload(epoch: int, best_score: float, *, include_opt: Any = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "best_macro_top1": best_score,
            "class_to_idx": class_to_idx,
            "model_name": model_name,
            "input_size": input_size,
            "crop_mode": crop_mode,
            "val_crop_mode": val_crop_mode,
            "saturation_jitter": saturation_jitter,
            "safety_lambda": safety_lambda,
            "oe_lambda": oe_lambda,
            "excluded_species": sorted(excluded_species),
        }
        if include_opt is not None:
            payload["optimizer_state_dict"] = include_opt.state_dict()
        return payload

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
            # Keep frozen backbone BatchNorm running stats intact during head-only warmup.
            model.eval()
            classifier = model.get_classifier()
            if isinstance(classifier, nn.Module):
                classifier.train()
            total_loss = 0.0
            n_batches = 0

            for i, (images, targets) in enumerate(train_loader):
                images = images.to(device_obj, non_blocking=True)
                targets = targets.to(device_obj, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=use_amp):
                    # Keep Phase (a) frozen-backbone head warmup purely on in-distribution mushrooms;
                    # apply Energy-Bounded OE strictly during Phase (b) full fine-tuning.
                    loss = _compute_batch_loss(images, targets, apply_oe=False)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                torch.save(_checkpoint_payload(epoch, best_macro_top1), best_checkpoint_path)

    # ==========================================
    # Phase (b): Full Fine-tuning
    # ==========================================
    logger.info("Starting Phase (b): Full fine-tuning for %d epochs...", fine_tune_epochs)
    unfreeze_backbone(model)
    if start_epoch <= warmup_epochs:
        ema.reset(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=fine_tune_lr,
        weight_decay=weight_decay,
    )
    if fine_tune_epochs > 1:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=1,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=fine_tune_epochs - 1,
            eta_min=min_lr,
        )
        scheduler: Any = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[1],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, fine_tune_epochs),
            eta_min=min_lr,
        )

    phase_b_start = max(warmup_epochs, start_epoch)
    for epoch in range(phase_b_start, total_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        current_lr = scheduler.get_last_lr()[0]

        use_oe_this_epoch = (epoch > phase_b_start) or (fine_tune_epochs <= 1)
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device_obj, non_blocking=True)
            targets = targets.to(device_obj, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=use_amp):
                loss = _compute_batch_loss(images, targets, apply_oe=use_oe_this_epoch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()
            n_batches += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                logger.info("Fine-tune epoch %d [%d/%d] | Loss: %.4f", epoch, i + 1, len(train_loader), loss.item())

        scheduler.step()
        avg_train_loss = total_loss / max(1, n_batches)

        # Re-align EMA BatchNorm stats on pure mushroom photos before validation
        _refresh_ema_bn(max_batches=150)

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
        torch.save(_checkpoint_payload(epoch, best_macro_top1, include_opt=optimizer), last_checkpoint_path)

        if val_metrics["macro_top1"] > best_macro_top1:
            best_macro_top1 = val_metrics["macro_top1"]
            torch.save(_checkpoint_payload(epoch, best_macro_top1), best_checkpoint_path)

    # Save final run config and summary
    best_sha256 = compute_file_sha256(best_checkpoint_path) if best_checkpoint_path.exists() else ""
    summary = {
        "run_id": run_id,
        "model_name": model_name,
        "input_size": input_size,
        "num_classes": num_classes,
        "crop_mode": crop_mode,
        "val_crop_mode": val_crop_mode,
        "saturation_jitter": saturation_jitter,
        "safety_lambda": safety_lambda,
        "oe_lambda": oe_lambda,
        "oe_csv": str(oe_csv) if oe_csv is not None else None,
        "excluded_species": sorted(excluded_species),
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
    parser.add_argument("--crop-mode", type=str, default="none_stretch", help="Preprocessing crop mode")
    parser.add_argument("--saturation-jitter", type=float, default=0.05, help="ColorJitter saturation range")
    parser.add_argument("--safety-lambda", type=float, default=0.0, help="Asymmetric SafetyCrossEntropyLoss weight")
    parser.add_argument(
        "--general-toxic-edible-weight",
        type=float,
        default=0.25,
        help="Weight for non-twin toxic -> edible penalty in SafetyCrossEntropyLoss",
    )
    parser.add_argument("--oe-csv", type=str, default=None, help="External CC0 Outlier Exposure CSV manifest")
    parser.add_argument("--oe-image-dir", type=str, default=None, help="Root directory for OE images")
    parser.add_argument("--oe-lambda", type=float, default=0.0, help="Energy-Bounded OE loss weight")
    parser.add_argument("--oe-margin-in", type=float, default=-3.5, help="Energy margin for in-distribution samples")
    parser.add_argument("--oe-margin-out", type=float, default=-1.0, help="Energy margin for OE samples")
    parser.add_argument("--oe-temperature", type=float, default=1.0, help="Temperature for EnergyBoundedOELoss")
    parser.add_argument("--oe-batch-size", type=int, default=None, help="OE mini-batch size")
    parser.add_argument("--exclude-uncovered", action="store_true", help="Exclude species below coverage thresholds from head")
    parser.add_argument("--coverage-csv", type=str, default="configs/pokrycie-67.csv", help="Coverage threshold table")
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
        crop_mode=args.crop_mode,
        saturation_jitter=args.saturation_jitter,
        safety_lambda=args.safety_lambda,
        general_toxic_to_edible_weight=args.general_toxic_edible_weight,
        oe_csv=args.oe_csv,
        oe_image_dir=args.oe_image_dir,
        oe_lambda=args.oe_lambda,
        oe_margin_in=args.oe_margin_in,
        oe_margin_out=args.oe_margin_out,
        oe_temperature=args.oe_temperature,
        oe_batch_size=args.oe_batch_size,
        exclude_uncovered=args.exclude_uncovered,
        coverage_csv=args.coverage_csv,
    )


if __name__ == "__main__":
    main()
