"""Stage P4: Temperature scaling calibration and OOD energy gate module.

Module: grzybobranie_ml.calibrate_ood
CLI: python -m grzybobranie_ml.calibrate_ood [options]

Functions:
- Excludes uncovered species (such as Amanita verna) from the active model head.
- Optimizes temperature T via NLL on validation set (Guo et al. 2017).
- Computes ECE (15 bins) and MCE before and after calibration with bootstrap 95% CI.
- Computes ECE specifically for lethal high-stakes species.
- Evaluates Liu et al. (2020) energy gate E(x) = -T * logsumexp(z / T).
- Evaluates AUROC and FPR@95TPR on OOD calibration, test, and sub-group splits.
- Assesses safety metrics: dangerous pairs confusion and high-stakes recall with gating.
- Generates reliability diagram PNG and JSON artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize as opt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset

from grzybobranie_ml.train import (
    MushroomSplitDataset,
    create_classifier_model,
    get_val_transforms,
    load_dangerous_pairs,
    load_high_stakes_species,
)

logger = logging.getLogger("grzybobranie_ml.calibrate_ood")

DEFAULT_IMAGE_DIR = Path("/home/bubolka/hf-768/upload")
DEFAULT_CHECKPOINT = Path(
    "artifacts/train_runs/mobilenetv4_conv_medium.e500_r256_in1k_r256_1795514748/best_val_checkpoint.pt"
)
DEFAULT_VAL_CSV = Path("data/splits/val.csv")
DEFAULT_OOD_FUNGI_CSV = Path("data/ood-v1/ood_fungi.csv")
DEFAULT_OOD_NONFUNGI_CSV = Path("data/ood-v1/ood_nonfungi.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/calibration")
DEFAULT_KARTY_PATH = Path("configs/karty_sobowtory.json")


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class OODManifestDataset(Dataset):
    """Dataset reading OOD samples mapped via manifest_768.csv."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform: Any = None,
        path_lookup: dict[str, Path] | None = None,
        split_filter: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if path_lookup is not None:
            self.path_lookup = path_lookup
        else:
            self.path_lookup = {}
            manifest_file = self.image_dir / "manifest_768.csv"
            if manifest_file.exists():
                with open(manifest_file, encoding="utf-8") as mf:
                    for r in csv.DictReader(mf):
                        rel = r.get("path")
                        if rel:
                            full_p = self.image_dir / rel
                            if r.get("photo_id"):
                                self.path_lookup[r["photo_id"]] = full_p
                            if r.get("image_url"):
                                self.path_lookup[r["image_url"]] = full_p
                            if r.get("source_url"):
                                self.path_lookup[r["source_url"]] = full_p

        self.records: list[dict[str, str]] = []
        self.image_paths: list[Path] = []

        with open(self.csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if split_filter and row.get("proposed_split") != split_filter:
                    continue
                pid = row.get("photo_id")
                url = row.get("image_url")
                src = row.get("source_url")
                p = self.path_lookup.get(url) or self.path_lookup.get(pid) or self.path_lookup.get(src)
                if p and p.exists():
                    self.records.append(row)
                    self.image_paths.append(p)
                else:
                    raise FileNotFoundError(f"Image for {self.csv_path.name} not found: {pid} / {url}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        p = self.image_paths[idx]
        with Image.open(p) as img:
            oriented = ImageOps.exif_transpose(img)
            image = (oriented if oriented is not None else img).convert("RGB")
        tensor = self.transform(image) if self.transform else T.ToTensor()(image)
        return tensor, idx


def fit_temperature_scaling(
    logits: np.ndarray,
    targets: np.ndarray,
    initial_t: float = 1.0,
) -> float:
    """Fit temperature parameter T > 0 on logits and targets by minimizing NLL (Guo et al. 2017)."""
    if len(logits) == 0:
        return 1.0

    val_z = torch.from_numpy(logits.astype(np.float64))
    val_y = torch.from_numpy(targets.astype(np.int64))
    temp_param = nn.Parameter(torch.tensor([float(initial_t)], dtype=torch.float64))

    optimizer = torch.optim.LBFGS([temp_param], lr=0.01, max_iter=200, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(val_z / temp_param, val_y)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        t_val = float(temp_param.item())
        if np.isfinite(t_val) and 0.05 < t_val < 20.0:
            return t_val
    except (RuntimeError, ValueError) as e:
        logger.warning("L-BFGS temperature optimization failed: %s. Using bounded scalar minimization.", e)

    # Fallback to bounded scalar minimization
    def nll_scalar(t: float) -> float:
        scaled = logits / t
        max_s = np.max(scaled, axis=1, keepdims=True)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
        log_probs = scaled - lse
        target_log_probs = log_probs[np.arange(len(targets)), targets]
        return float(-np.mean(target_log_probs))

    res = opt.minimize_scalar(nll_scalar, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def compute_nll(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> float:
    """Compute Negative Log-Likelihood given logits, targets, and temperature."""
    if len(logits) == 0:
        return 0.0
    scaled = logits / temperature
    max_s = np.max(scaled, axis=1, keepdims=True)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
    log_probs = scaled - lse
    target_log_probs = log_probs[np.arange(len(targets)), targets]
    return float(-np.mean(target_log_probs))


def compute_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable softmax(logits / T) along axis 1."""
    if len(logits) == 0:
        return np.empty((0, logits.shape[1] if logits.ndim > 1 else 0), dtype=np.float64)
    scaled = logits / temperature
    max_s = np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled - max_s)
    return exp / np.sum(exp, axis=1, keepdims=True)


def compute_ece_mce(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float = 1.0,
    n_bins: int = 15,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) with M equal-width bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    if len(targets) == 0:
        empty_bins = [
            {
                "bin": i,
                "low": float(bin_boundaries[i]),
                "high": float(bin_boundaries[i + 1]),
                "count": 0,
                "acc": 0.0,
                "conf": 0.0,
                "diff": 0.0,
            }
            for i in range(n_bins)
        ]
        return 0.0, 0.0, empty_bins

    probs = compute_softmax(logits, temperature)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    corrects = (preds == targets).astype(np.float64)

    ece = 0.0
    mce = 0.0
    bin_stats: list[dict[str, Any]] = []

    n_samples = len(targets)
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs >= low) & (confs <= high) if i == 0 else (confs > low) & (confs <= high)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(corrects[mask]))
            bin_conf = float(np.mean(confs[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (count / n_samples) * diff
            mce = max(mce, diff)
            bin_stats.append({
                "bin": i,
                "low": float(low),
                "high": float(high),
                "count": count,
                "acc": bin_acc,
                "conf": bin_conf,
                "diff": diff,
            })
        else:
            bin_stats.append({
                "bin": i,
                "low": float(low),
                "high": float(high),
                "count": 0,
                "acc": 0.0,
                "conf": 0.0,
                "diff": 0.0,
            })

    return float(ece), float(mce), bin_stats


def bootstrap_ece_ci(
    logits: np.ndarray,
    targets: np.ndarray,
    temperature: float = 1.0,
    n_bins: int = 15,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for ECE."""
    n = len(targets)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        ece, _, _ = compute_ece_mce(logits, targets, temperature, n_bins)
        return float(ece), float(ece)

    probs = compute_softmax(logits, temperature)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    corrects = (preds == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    rng = np.random.default_rng(seed)

    boot_eces = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sub_confs = confs[idx]
        sub_corrects = corrects[idx]
        ece = 0.0
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (sub_confs >= low) & (sub_confs <= high) if i == 0 else (sub_confs > low) & (sub_confs <= high)
            count = np.sum(mask)
            if count > 0:
                bin_acc = np.mean(sub_corrects[mask])
                bin_conf = np.mean(sub_confs[mask])
                ece += (count / n) * abs(bin_acc - bin_conf)
        boot_eces[b] = ece

    low_ci, high_ci = np.percentile(boot_eces, [2.5, 97.5])
    return float(low_ci), float(high_ci)


def compute_energy_vector(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Compute Liu et al. (2020) energy E(x) = -T * logsumexp(z / T) for a batch of logits."""
    if len(logits) == 0:
        return np.empty(0, dtype=np.float64)
    scaled = logits / temperature
    max_s = np.max(scaled, axis=1, keepdims=True)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
    return float(-temperature) * lse.squeeze(axis=1)


def compute_auroc_and_fpr(
    id_energy: np.ndarray,
    ood_energy: np.ndarray,
    target_id_tpr: float = 0.95,
) -> tuple[float, float, float]:
    """Compute AUROC and FPR at target ID TPR (standard: 95%).

    Returns:
        (auroc, fpr_at_target_tpr, energy_threshold)
    """
    if len(id_energy) == 0 or len(ood_energy) == 0:
        return 0.5, 1.0, 0.0

    # Lower energy indicates in-distribution. So score = -energy is higher for ID.
    u = stats.mannwhitneyu(-id_energy, -ood_energy, alternative="greater").statistic
    auroc = float(u / (len(id_energy) * len(ood_energy)))

    # Threshold for target_id_tpr with method="higher" to guarantee >= target_id_tpr keep rate on ID
    threshold = float(np.percentile(id_energy, target_id_tpr * 100.0, method="higher"))
    # FPR on OOD is the fraction of OOD samples with energy <= threshold (falsely accepted as ID)
    fpr = float(np.mean(ood_energy <= threshold))
    return auroc, fpr, threshold


def render_reliability_diagram(
    raw_bin_stats: list[dict[str, Any]],
    cal_bin_stats: list[dict[str, Any]],
    raw_ece: float,
    cal_ece: float,
    raw_mce: float,
    cal_mce: float,
    temperature: float,
    save_path: str | Path,
) -> None:
    """Render a reliability diagram comparison before vs after temperature scaling using PIL."""
    width, height = 1200, 700
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Margins and plot dimensions
    margin_top = 80
    plot_width = 460
    plot_height = 460

    x_left1 = 100
    y_top = margin_top
    x_left2 = 680

    def draw_single_diagram(
        x0: int,
        y0: int,
        bin_stats: list[dict[str, Any]],
        title: str,
        ece: float,
        mce: float,
        accent_color: tuple[int, int, int],
    ) -> None:
        # Background box
        draw.rectangle([x0, y0, x0 + plot_width, y0 + plot_height], fill=(250, 250, 250), outline=(200, 200, 200), width=1)

        # Diagonal perfect calibration line
        draw.line([x0, y0 + plot_height, x0 + plot_width, y0], fill=(180, 180, 180), width=2)

        # Grid lines (5 ticks: 0.2, 0.4, 0.6, 0.8)
        for tick in [0.2, 0.4, 0.6, 0.8]:
            px = int(x0 + tick * plot_width)
            py = int(y0 + plot_height - tick * plot_height)
            draw.line([px, y0, px, y0 + plot_height], fill=(230, 230, 230), width=1)
            draw.line([x0, py, x0 + plot_width, py], fill=(230, 230, 230), width=1)

        # Plot bins
        n_bins = len(bin_stats)
        bin_w = plot_width / n_bins
        for i, b in enumerate(bin_stats):
            count = b["count"]
            if count == 0:
                continue
            acc = b["acc"]
            conf = b["conf"]

            bx = x0 + i * bin_w
            bw = max(2, int(bin_w - 2))

            # Accuracy bar
            bar_h = int(acc * plot_height)
            by = y0 + plot_height - bar_h
            draw.rectangle([bx, by, bx + bw, y0 + plot_height], fill=accent_color, outline=(40, 40, 40))

            # Confidence marker
            cy = int(y0 + plot_height - conf * plot_height)
            draw.line([bx, cy, bx + bw, cy], fill=(220, 50, 50), width=3)

        # Axis labels & annotations
        draw.text((x0 + 10, y0 - 35), title, fill=(20, 20, 20))
        draw.text((x0 + 10, y0 + 15), f"ECE: {ece * 100:.2f}% | MCE: {mce * 100:.2f}%", fill=(30, 30, 30))
        draw.text((x0 + plot_width // 2 - 40, y0 + plot_height + 15), "Confidence", fill=(50, 50, 50))
        draw.text((x0 - 55, y0 + plot_height // 2), "Accuracy", fill=(50, 50, 50))

    # Draw Uncalibrated
    draw_single_diagram(
        x_left1,
        y_top,
        raw_bin_stats,
        "Uncalibrated (T = 1.000)",
        raw_ece,
        raw_mce,
        (130, 170, 225),
    )

    # Draw Calibrated
    draw_single_diagram(
        x_left2,
        y_top,
        cal_bin_stats,
        f"Calibrated (T = {temperature:.4f})",
        cal_ece,
        cal_mce,
        (80, 180, 120),
    )

    # Header title
    draw.text((width // 2 - 180, 25), "Reliability Diagram (15 bins) — Validation Set", fill=(10, 10, 10))

    # Legend at bottom
    legend_y = y_top + plot_height + 45
    draw.rectangle([width // 2 - 180, legend_y, width // 2 - 160, legend_y + 12], fill=(130, 170, 225), outline=(0, 0, 0))
    draw.text((width // 2 - 150, legend_y - 2), "Accuracy bar", fill=(40, 40, 40))
    draw.line([width // 2 + 20, legend_y + 6, width // 2 + 50, legend_y + 6], fill=(220, 50, 50), width=3)
    draw.text((width // 2 + 60, legend_y - 2), "Mean confidence marker", fill=(40, 40, 40))

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_p)
    logger.info("Saved reliability diagram to %s", out_p)


def evaluate_checkpoint_calibration(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    val_csv: str | Path = DEFAULT_VAL_CSV,
    ood_fungi_csv: str | Path = DEFAULT_OOD_FUNGI_CSV,
    ood_nonfungi_csv: str | Path = DEFAULT_OOD_NONFUNGI_CSV,
    karty_path: str | Path = DEFAULT_KARTY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_top1_prob: float = 0.50,
    energy_threshold_override: float | None = -2.2082,
    batch_size: int = 128,
    num_workers: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cached_logits_path: str | Path | None = None,
    force_extract: bool = False,
    crop_mode: str = "none_stretch",
    calib_only: bool = False,
) -> dict[str, Any]:
    """Execute Stage P4: temperature calibration and OOD energy gate."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(checkpoint_path)

    device_obj = torch.device(device)
    logger.info("Running P4 calibration and OOD gate on %s (device: %s)...", ckpt_path, device_obj)

    # Check if cached logits exist to save time
    cache_file = Path(cached_logits_path) if cached_logits_path else output_path / "extracted_logits.pt"
    use_cache = False

    if cache_file.exists() and not force_extract:
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        cached_ckpt = cached.get("checkpoint_path")
        cached_crop_mode = cached.get("crop_mode", "legacy_448_center")
        if cached_ckpt and Path(cached_ckpt).resolve() != ckpt_path.resolve():
            logger.warning(
                "Cached logits in %s were generated for %s, but target is %s. Cache invalidated; extracting fresh logits.",
                cache_file,
                cached_ckpt,
                ckpt_path,
            )
            use_cache = False
        elif cached_crop_mode != crop_mode:
            logger.warning(
                "Cached logits in %s used crop_mode=%r, but target is %r. Cache invalidated; extracting fresh logits.",
                cache_file,
                cached_crop_mode,
                crop_mode,
            )
            use_cache = False
        else:
            use_cache = True

    if use_cache:
        logger.info("Loading cached logits from %s", cache_file)
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        class_to_idx_66 = cached["class_to_idx_66"]
        idx_to_class_66 = cached["idx_to_class_66"]
        active_classes = cached["active_classes"]
        val_logits = cached["val_logits"].double().numpy()
        val_targets = np.array(cached["val_targets"], dtype=np.int64)
        val_records = cached["val_records"]
        verna_logits = cached["verna_logits"].double().numpy()
        verna_records = cached["verna_records"]
        ood_fungi_logits = cached["ood_fungi_logits"].double().numpy()
        ood_fungi_records = cached["ood_fungi_records"]
        ood_nonfungi_logits = cached["ood_nonfungi_logits"].double().numpy()
        ood_nonfungi_records = cached["ood_nonfungi_records"]
    else:
        logger.info("Extracting logits from model on disk...")
        ckpt = torch.load(ckpt_path, map_location=device_obj, weights_only=False)
        model_name = ckpt["model_name"]
        input_size = ckpt["input_size"]
        class_to_idx_67 = ckpt["class_to_idx"]
        sorted_classes = sorted(class_to_idx_67.keys())

        # Amanita verna exclusion: remove index from active classes
        verna_idx = class_to_idx_67.get("amanita_verna", -1)
        active_classes = [c for c in sorted_classes if c != "amanita_verna"]
        class_to_idx_66 = {c: i for i, c in enumerate(active_classes)}
        idx_to_class_66 = {i: c for i, c in enumerate(active_classes)}

        model = create_classifier_model(model_name, num_classes=len(class_to_idx_67), pretrained=False).to(device_obj)
        model.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
        model.eval()

        transform = get_val_transforms(input_size, crop_mode=crop_mode)
        val_ds = MushroomSplitDataset(val_csv, image_dir=image_dir, transform=transform, class_to_idx=class_to_idx_66)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        verna_ds = MushroomSplitDataset(val_csv, image_dir=image_dir, transform=transform, class_to_idx={"amanita_verna": 0})
        verna_loader = DataLoader(verna_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        shared_lookup = val_ds.path_lookup
        ood_split_filter = "calib" if calib_only else None
        ood_f_ds = OODManifestDataset(
            ood_fungi_csv,
            image_dir=image_dir,
            transform=transform,
            path_lookup=shared_lookup,
            split_filter=ood_split_filter,
        )
        ood_f_loader = DataLoader(ood_f_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        ood_nf_ds = OODManifestDataset(
            ood_nonfungi_csv,
            image_dir=image_dir,
            transform=transform,
            path_lookup=shared_lookup,
            split_filter=ood_split_filter,
        )
        ood_nf_loader = DataLoader(ood_nf_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        def run_extraction(loader: DataLoader) -> tuple[np.ndarray, list[int]]:
            logits_list = []
            targets_list = []
            with torch.no_grad():
                for imgs, tgts in loader:
                    imgs = imgs.to(device_obj, non_blocking=True)
                    with torch.amp.autocast(device_type=device_obj.type, dtype=torch.bfloat16):
                        out67 = model(imgs)
                    if verna_idx >= 0:
                        out66 = torch.cat([out67[:, :verna_idx], out67[:, verna_idx + 1:]], dim=1)
                    else:
                        out66 = out67
                    logits_list.append(out66.cpu().double().numpy())
                    targets_list.extend(tgts.tolist())
            return np.concatenate(logits_list, axis=0), targets_list

        val_logits, val_targets_list = run_extraction(val_loader)
        val_targets = np.array(val_targets_list, dtype=np.int64)
        val_records = val_ds.records

        verna_logits, _ = run_extraction(verna_loader)
        verna_records = verna_ds.records

        ood_fungi_logits, _ = run_extraction(ood_f_loader)
        ood_fungi_records = ood_f_ds.records

        ood_nonfungi_logits, _ = run_extraction(ood_nf_loader)
        ood_nonfungi_records = ood_nf_ds.records

        # Save cache
        torch.save(
            {
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": compute_file_sha256(ckpt_path) if ckpt_path.exists() else "",
                "crop_mode": crop_mode,
                "calib_only": calib_only,
                "class_to_idx_66": class_to_idx_66,
                "idx_to_class_66": idx_to_class_66,
                "active_classes": active_classes,
                "val_logits": torch.from_numpy(val_logits),
                "val_targets": val_targets_list,
                "val_records": val_records,
                "verna_logits": torch.from_numpy(verna_logits),
                "verna_records": verna_records,
                "ood_fungi_logits": torch.from_numpy(ood_fungi_logits),
                "ood_fungi_records": ood_fungi_records,
                "ood_nonfungi_logits": torch.from_numpy(ood_nonfungi_logits),
                "ood_nonfungi_records": ood_nonfungi_records,
            },
            cache_file,
        )

    # 1. Temperature Fitting
    t_opt = fit_temperature_scaling(val_logits, val_targets)
    nll_raw = compute_nll(val_logits, val_targets, 1.0)
    nll_cal = compute_nll(val_logits, val_targets, t_opt)

    # 2. ECE and MCE computation
    ece_raw, mce_raw, raw_bins = compute_ece_mce(val_logits, val_targets, 1.0, n_bins=15)
    ece_cal, mce_cal, cal_bins = compute_ece_mce(val_logits, val_targets, t_opt, n_bins=15)

    # Bootstrap 95% CI
    ci_raw_low, ci_raw_high = bootstrap_ece_ci(val_logits, val_targets, 1.0, n_bins=15)
    ci_cal_low, ci_cal_high = bootstrap_ece_ci(val_logits, val_targets, t_opt, n_bins=15)

    # ECE on lethal classes
    karty_p = Path(karty_path)
    edible_slugs: set[str] = set()
    toxic_or_deadly_slugs: set[str] = set()
    lethal_slugs: list[str] = []

    if karty_p.exists():
        with open(karty_p, encoding="utf-8") as f:
            karty_data = json.load(f)
            for c in karty_data.get("karty", []):
                st = str(c.get("status_jadalnosci", "")).lower()
                slug = c.get("slug")
                if not slug:
                    continue
                if st == "jadalny":
                    edible_slugs.add(slug)
                if st in ["trujący", "śmiertelny", "smiertelnie trujacy"] or "toksycz" in st:
                    toxic_or_deadly_slugs.add(slug)
                if "śmiertel" in st or "smiertel" in st:
                    lethal_slugs.append(slug)
    else:
        lethal_slugs = [
            "amanita_phalloides",
            "amanita_virosa",
            "cortinarius_orellanus",
            "galerina_marginata",
            "gyromitra_esculenta",
            "inocybe_erubescens",
            "lepiota_brunneoincarnata",
        ]

    lethal_indices = [class_to_idx_66[s] for s in lethal_slugs if s in class_to_idx_66]
    mask_lethal = np.isin(val_targets, lethal_indices)
    ece_lethal_raw, mce_lethal_raw, _ = compute_ece_mce(val_logits[mask_lethal], val_targets[mask_lethal], 1.0, n_bins=15)
    ece_lethal_cal, mce_lethal_cal, _ = compute_ece_mce(val_logits[mask_lethal], val_targets[mask_lethal], t_opt, n_bins=15)

    # Render reliability diagram
    diagram_path = output_path / "reliability_diagram.png"
    render_reliability_diagram(
        raw_bins,
        cal_bins,
        ece_raw,
        ece_cal,
        mce_raw,
        mce_cal,
        t_opt,
        diagram_path,
    )

    # 3. Energy Gate & OOD metrics
    e_val = compute_energy_vector(val_logits, t_opt)
    p_val = np.max(compute_softmax(val_logits, t_opt), axis=1)

    e_fungi = compute_energy_vector(ood_fungi_logits, t_opt)
    e_verna = compute_energy_vector(verna_logits, t_opt)

    is_calib_f = np.array([r.get("proposed_split") == "calib" for r in ood_fungi_records])
    is_calib_nf = np.array([r.get("proposed_split") == "calib" for r in ood_nonfungi_records])

    ood_splits: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {
        "ood_calib": (
            np.concatenate([ood_fungi_logits[is_calib_f], ood_nonfungi_logits[is_calib_nf]]),
            [r for r in ood_fungi_records if r.get("proposed_split") == "calib"]
            + [r for r in ood_nonfungi_records if r.get("proposed_split") == "calib"],
        ),
        "ood_fungi": (ood_fungi_logits, ood_fungi_records),
        "ood_nonfungi": (ood_nonfungi_logits, ood_nonfungi_records),
    }
    if not calib_only:
        ood_splits["ood_test"] = (
            np.concatenate([ood_fungi_logits[~is_calib_f], ood_nonfungi_logits[~is_calib_nf]]),
            [r for r in ood_fungi_records if r.get("proposed_split") == "test"]
            + [r for r in ood_nonfungi_records if r.get("proposed_split") == "test"],
        )
        ood_splits["ood_total"] = (
            np.concatenate([ood_fungi_logits, ood_nonfungi_logits]),
            ood_fungi_records + ood_nonfungi_records,
        )

    # Determine energy threshold (Recommendation B frozen: -2.2082, or compute at 95% keep rate if override is None)
    _auroc_calib, _fpr95_calib, calib_threshold = compute_auroc_and_fpr(
        e_val, compute_energy_vector(ood_splits["ood_calib"][0], t_opt), target_id_tpr=0.95
    )
    energy_threshold = energy_threshold_override if energy_threshold_override is not None else calib_threshold

    ood_metrics_report: dict[str, Any] = {}
    for split_name, (split_logits, split_records) in ood_splits.items():
        split_e = compute_energy_vector(split_logits, t_opt)
        split_probs = compute_softmax(split_logits, t_opt)
        split_p = np.max(split_probs, axis=1)
        split_preds = np.argmax(split_logits, axis=1)

        auroc, fpr95, _ = compute_auroc_and_fpr(e_val, split_e, target_id_tpr=0.95)

        is_ood = split_e > energy_threshold
        is_unclear = (~is_ood) & (split_p < min_top1_prob)
        is_ok = (~is_ood) & (~is_unclear)

        conf_edible_mask = [is_ok[i] and (idx_to_class_66[split_preds[i]] in edible_slugs) for i in range(len(split_e))]
        conf_edible_cnt = int(np.sum(conf_edible_mask))

        ood_metrics_report[split_name] = {
            "n_samples": len(split_e),
            "auroc": float(auroc),
            "fpr_at_95tpr": float(fpr95),
            "mean_energy": float(np.mean(split_e)),
            "ood_rejection_rate": float(np.mean(is_ood)),
            "unclear_rejection_rate": float(np.mean(is_unclear)),
            "total_rejection_rate": float(np.mean(~is_ok)),
            "acceptance_rate": float(np.mean(is_ok)),
            "confident_edible_count": conf_edible_cnt,
            "confident_edible_rate": float(conf_edible_cnt / len(split_e)),
        }

    # Val keep / rejection rates
    val_preds = np.argmax(val_logits, axis=1)
    val_is_ood = e_val > energy_threshold
    val_is_unclear = (~val_is_ood) & (p_val < min_top1_prob)
    val_is_ok = (~val_is_ood) & (~val_is_unclear)
    val_gating = {
        "n_samples": len(e_val),
        "energy_threshold": float(energy_threshold),
        "val_energy_rejection_rate": float(np.mean(val_is_ood)),
        "val_unclear_rejection_rate": float(np.mean(val_is_unclear)),
        "val_total_rejection_rate": float(np.mean(~val_is_ok)),
        "val_keep_rate": float(np.mean(val_is_ok)),
    }

    # Safety: confident toxic as edible on val
    val_conf_toxic_as_edible = sum(
        val_is_ok[i]
        and (idx_to_class_66[val_targets[i]] in toxic_or_deadly_slugs)
        and (idx_to_class_66[val_preds[i]] in edible_slugs)
        for i in range(len(val_targets))
    )

    # 4. Out-of-model Priority 1 look-alikes
    p1_slugs = [
        "clitocybe_rivulosa",
        "lepiota_subincarnata",
        "lepiota_castanea",
        "lepiota_clypeolaria",
        "tricholoma_pardinum",
        "tricholoma_virgatum",
    ]
    p1_report: dict[str, Any] = {}
    for s in p1_slugs:
        mask = np.array([r.get("class_slug") == s for r in ood_fungi_records])
        n = int(np.sum(mask))
        if n > 0:
            sub_e = e_fungi[mask]
            sub_probs = compute_softmax(ood_fungi_logits[mask], t_opt)
            sub_p = np.max(sub_probs, axis=1)
            sub_preds = np.argmax(ood_fungi_logits[mask], axis=1)

            is_ood = sub_e > energy_threshold
            is_unclear = (~is_ood) & (sub_p < min_top1_prob)
            is_ok = (~is_ood) & (~is_unclear)
            conf_edible = sum(is_ok[i] and (idx_to_class_66[sub_preds[i]] in edible_slugs) for i in range(n))

            p1_report[s] = {
                "n_samples": n,
                "mean_energy": float(np.mean(sub_e)),
                "ood_rejected_count": int(np.sum(is_ood)),
                "unclear_rejected_count": int(np.sum(is_unclear)),
                "total_rejected_count": int(np.sum(~is_ok)),
                "total_rejection_rate": float(np.mean(~is_ok)),
                "accepted_count": int(np.sum(is_ok)),
                "confident_edible_count": int(conf_edible),
            }

    # 5. Amanita verna report (uncovered species)
    p_verna = np.max(compute_softmax(verna_logits, t_opt), axis=1)
    preds_verna = np.argmax(verna_logits, axis=1)
    verna_is_ood = e_verna > energy_threshold
    verna_is_unclear = (~verna_is_ood) & (p_verna < min_top1_prob)
    verna_is_ok = (~verna_is_ood) & (~verna_is_unclear)
    verna_conf_edible = sum(verna_is_ok[i] and (idx_to_class_66[preds_verna[i]] in edible_slugs) for i in range(len(e_verna)))

    verna_report = {
        "n_samples": len(e_verna),
        "mean_energy": float(np.mean(e_verna)),
        "ood_rejected_count": int(np.sum(verna_is_ood)),
        "unclear_rejected_count": int(np.sum(verna_is_unclear)),
        "total_rejected_count": int(np.sum(~verna_is_ok)),
        "accepted_count": int(np.sum(verna_is_ok)),
        "confident_edible_count": int(verna_conf_edible),
        "per_sample_predictions": [
            {
                "idx": i + 1,
                "energy": float(e_verna[i]),
                "top1_prob": float(p_verna[i]),
                "predicted_class": idx_to_class_66[preds_verna[i]],
                "predicted_status": "out_of_distribution" if verna_is_ood[i] else ("unclear" if verna_is_unclear[i] else "ok"),
            }
            for i in range(len(e_verna))
        ],
    }

    # 6. Dangerous pairs evaluation
    all_pairs_config = load_dangerous_pairs(karty_path)
    active_pairs = [
        (a, b) for a, b in all_pairs_config
        if a in class_to_idx_66 and b in class_to_idx_66
    ]

    pairs_report: dict[str, Any] = {}
    problematic_pairs: dict[str, Any] = {}
    violating_raw_count = 0
    violating_energy_count = 0
    violating_combined_count = 0

    for a, b in active_pairs:
        ia, ib = class_to_idx_66[a], class_to_idx_66[b]
        mask_a = val_targets == ia
        mask_b = val_targets == ib
        na, nb = int(np.sum(mask_a)), int(np.sum(mask_b))

        # Raw (no gating)
        c_ab_raw = int(np.sum(val_preds[mask_a] == ib))
        c_ba_raw = int(np.sum(val_preds[mask_b] == ia))
        err_ab_raw = float(c_ab_raw / na) if na > 0 else 0.0
        err_ba_raw = float(c_ba_raw / nb) if nb > 0 else 0.0
        max_raw = max(err_ab_raw, err_ba_raw)
        if max_raw > 0.05:
            violating_raw_count += 1

        # Energy gate only
        ok_e = e_val <= energy_threshold
        mask_a_e = mask_a & ok_e
        mask_b_e = mask_b & ok_e
        na_e, nb_e = int(np.sum(mask_a_e)), int(np.sum(mask_b_e))
        c_ab_e = int(np.sum(val_preds[mask_a_e] == ib))
        c_ba_e = int(np.sum(val_preds[mask_b_e] == ia))
        err_ab_e = float(c_ab_e / na_e) if na_e > 0 else 0.0
        err_ba_e = float(c_ba_e / nb_e) if nb_e > 0 else 0.0
        max_e = max(err_ab_e, err_ba_e)
        if max_e > 0.05:
            violating_energy_count += 1

        # Combined gate (E <= th and p >= min_top1_prob)
        ok_c = ok_e & (p_val >= min_top1_prob)
        mask_a_c = mask_a & ok_c
        mask_b_c = mask_b & ok_c
        na_c, nb_c = int(np.sum(mask_a_c)), int(np.sum(mask_b_c))
        c_ab_c = int(np.sum(val_preds[mask_a_c] == ib))
        c_ba_c = int(np.sum(val_preds[mask_b_c] == ia))
        err_ab_c = float(c_ab_c / na_c) if na_c > 0 else 0.0
        err_ba_c = float(c_ba_c / nb_c) if nb_c > 0 else 0.0
        max_c = max(err_ab_c, err_ba_c)
        if max_c > 0.05:
            violating_combined_count += 1

        pair_data = {
            "raw": {
                "max_err": max_raw,
                "cab": c_ab_raw,
                "na": na,
                "eab": err_ab_raw,
                "cba": c_ba_raw,
                "nb": nb,
                "eba": err_ba_raw,
            },
            "energy_gated": {
                "max_err": max_e,
                "cab": c_ab_e,
                "na": na_e,
                "eab": err_ab_e,
                "cba": c_ba_e,
                "nb": nb_e,
                "eba": err_ba_e,
            },
            "combined_gated": {
                "max_err": max_c,
                "cab": c_ab_c,
                "na": na_c,
                "eab": err_ab_c,
                "cba": c_ba_c,
                "nb": nb_c,
                "eba": err_ba_c,
            },
        }
        pair_key = f"{a} <-> {b}"
        pairs_report[pair_key] = pair_data
        if max_raw > 0.05:
            problematic_pairs[pair_key] = pair_data

    # 7. High-stakes species recall
    high_stakes_slugs = sorted([s for s in load_high_stakes_species(karty_path) if s in class_to_idx_66])
    hs_report: dict[str, Any] = {}
    for s in high_stakes_slugs:
        idx = class_to_idx_66[s]
        mask = val_targets == idx
        n = int(np.sum(mask))

        correct_raw = int(np.sum(val_preds[mask] == idx))
        rec_raw = float(correct_raw / n) if n > 0 else 0.0

        # Energy gate
        ok_e = e_val <= energy_threshold
        n_e = int(np.sum(mask & ok_e))
        correct_e = int(np.sum(val_preds[mask & ok_e] == idx))
        rec_e_effective = float(correct_e / n) if n > 0 else 0.0
        rec_e_conditional = float(correct_e / n_e) if n_e > 0 else 0.0

        # Combined gate
        ok_c = ok_e & (p_val >= min_top1_prob)
        n_c = int(np.sum(mask & ok_c))
        correct_c = int(np.sum(val_preds[mask & ok_c] == idx))
        rec_c_effective = float(correct_c / n) if n > 0 else 0.0
        rec_c_conditional = float(correct_c / n_c) if n_c > 0 else 0.0

        hs_report[s] = {
            "n_val": n,
            "raw_recall": rec_raw,
            "energy_gated": {
                "n_accepted": n_e,
                "n_correct": correct_e,
                "effective_recall": rec_e_effective,
                "conditional_recall": rec_e_conditional,
            },
            "combined_gated": {
                "n_accepted": n_c,
                "n_correct": correct_c,
                "effective_recall": rec_c_effective,
                "conditional_recall": rec_c_conditional,
            },
        }

    # Compile complete calibration summary
    results = {
        "checkpoint": str(checkpoint_path),
        "temperature": float(t_opt),
        "calibration": {
            "method": "temperature",
            "temperature": float(t_opt),
            "fitSplit": "val",
            "eceBins": 15,
            "raw": {
                "nll": float(nll_raw),
                "ece": float(ece_raw),
                "mce": float(mce_raw),
                "ci95": [float(ci_raw_low), float(ci_raw_high)],
                "lethal_ece": float(ece_lethal_raw),
                "lethal_mce": float(mce_lethal_raw),
            },
            "calibrated": {
                "nll": float(nll_cal),
                "ece": float(ece_cal),
                "mce": float(mce_cal),
                "ci95": [float(ci_cal_low), float(ci_cal_high)],
                "lethal_ece": float(ece_lethal_cal),
                "lethal_mce": float(mce_lethal_cal),
            },
            "displayBands": {
                "high": 0.8,
                "medium": 0.5,
            },
        },
        "ood": {
            "method": "energy",
            "energyTemperature": float(t_opt),
            "energyThreshold": float(energy_threshold),
            "minTop1Prob": float(min_top1_prob),
            "val_gating": val_gating,
            "splits": ood_metrics_report,
            "priority1_look_alikes": p1_report,
            "uncovered_amanita_verna": verna_report,
        },
        "safety": {
            "val_confident_toxic_as_edible_count": int(val_conf_toxic_as_edible),
            "val_confident_toxic_as_edible_rate": float(val_conf_toxic_as_edible / len(val_targets)),
            "uncovered_verna_confident_edible_count": int(verna_conf_edible),
            "ood_confident_edible_count": ood_metrics_report["ood_calib"]["confident_edible_count"]
            if calib_only
            else ood_metrics_report["ood_total"]["confident_edible_count"],
        },
        "dangerous_pairs_summary": {
            "total_evaluated": len(active_pairs),
            "violating_raw_count": violating_raw_count,
            "violating_energy_gated_count": violating_energy_count,
            "violating_combined_gated_count": violating_combined_count,
            "problematic_pairs_list": list(problematic_pairs.keys()),
        },
        "problematic_dangerous_pairs": problematic_pairs,
        "dangerous_pairs": pairs_report,
        "high_stakes": hs_report,
    }

    results_file = output_path / "calibration_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved calibration and OOD results to %s", results_file)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 Calibration and OOD Energy Gate Evaluation.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR), help="Image directory")
    parser.add_argument("--val-csv", type=str, default=str(DEFAULT_VAL_CSV), help="Validation split CSV")
    parser.add_argument("--ood-fungi-csv", type=str, default=str(DEFAULT_OOD_FUNGI_CSV), help="OOD fungi CSV")
    parser.add_argument("--ood-nonfungi-csv", type=str, default=str(DEFAULT_OOD_NONFUNGI_CSV), help="OOD nonfungi CSV")
    parser.add_argument("--karty-path", type=str, default=str(DEFAULT_KARTY_PATH), help="Path to karty_sobowtory.json")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--min-top1-prob", type=float, default=0.50, help="Minimum top-1 probability threshold")
    parser.add_argument("--energy-threshold", type=str, default="-2.2082", help="Override energy threshold theta, or 'auto' for 95%% val TPR")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--cached-logits", type=str, default="artifacts/calibration/extracted_logits.pt", help="Path to cached logits")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of logits from checkpoint")
    parser.add_argument("--crop-mode", type=str, default="none_stretch", help="Validation preprocessing crop mode")
    parser.add_argument("--calib-only", action="store_true", help="Restrict OOD extraction and evaluation strictly to proposed_split=='calib'")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    e_override = None if args.energy_threshold.strip().lower() in {"auto", "none"} else float(args.energy_threshold)

    evaluate_checkpoint_calibration(
        checkpoint_path=args.checkpoint,
        image_dir=args.image_dir,
        val_csv=args.val_csv,
        ood_fungi_csv=args.ood_fungi_csv,
        ood_nonfungi_csv=args.ood_nonfungi_csv,
        karty_path=args.karty_path,
        output_dir=args.output_dir,
        min_top1_prob=args.min_top1_prob,
        energy_threshold_override=e_override,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        cached_logits_path=args.cached_logits,
        force_extract=args.force_extract,
        crop_mode=args.crop_mode,
        calib_only=args.calib_only,
    )


if __name__ == "__main__":
    main()
