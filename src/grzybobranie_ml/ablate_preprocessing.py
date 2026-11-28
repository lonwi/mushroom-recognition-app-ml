"""Diagnostic preprocessing ablation & asymmetric veto evaluation on val.csv and ood_calib.

Module: grzybobranie_ml.ablate_preprocessing
CLI: python -m grzybobranie_ml.ablate_preprocessing [options]

Evaluates frozen checkpoint `best_val_checkpoint.pt` (and optionally frozen FP16 TFLite)
strictly on `data/splits/val.csv` and `ood_calib` (proposed_split == "calib") across:
1. legacy_448_center: Resize(448) -> CenterCrop(256) (~24.5% of frame)
2. p1_box_none_stretch: Reference P1 chain (EXIF -> box 448 -> box 256 -> half_up_uint8 -> [-1, 1])
3. bilinear_none_stretch: Direct BILINEAR Resize((256, 256))
4. center_256: Proportional Resize(256) -> CenterCrop(256)

Also evaluates P5 asymmetric twin-pair veto (twin_pair_veto_threshold=0.10, twin_pair_min_ratio=5.0)
vs legacy high-stakes veto (0.20) on val.csv observations.

STRICT IMMUTABILITY: Never reads data/splits/test.csv or ood_test rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from grzybobranie_ml.calibrate_ood import (
    compute_auroc_and_fpr,
    compute_ece_mce,
    compute_energy_vector,
    compute_file_sha256,
    compute_nll,
    compute_softmax,
    fit_temperature_scaling,
)
from grzybobranie_ml.export_tflite import TFLiteRunner
from grzybobranie_ml.multi_photo import (
    fuse_observation_frames,
    load_species_edibility_sets,
)
from grzybobranie_ml.train import (
    MushroomSplitDataset,
    create_classifier_model,
    get_val_transforms,
    load_dangerous_pairs,
    load_high_stakes_species,
)

logger = logging.getLogger("grzybobranie_ml.ablate_preprocessing")

DEFAULT_CHECKPOINT = Path(
    "artifacts/train_runs/mobilenetv4_conv_medium.e500_r256_in1k_r256_1795514748/best_val_checkpoint.pt"
)
DEFAULT_TFLITE = Path("artifacts/models/mobilenetv4_medium_r256_fp16.tflite")
DEFAULT_VAL_CSV = Path("data/splits/val.csv")
DEFAULT_IMAGE_DIR = Path("/home/bubolka/hf-768/upload")
DEFAULT_OOD_FUNGI_CSV = Path("data/ood-v1/ood_fungi.csv")
DEFAULT_OOD_NONFUNGI_CSV = Path("data/ood-v1/ood_nonfungi.csv")
DEFAULT_KARTY_PATH = Path("configs/karty_sobowtory.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/ablation")

DEFAULT_CROP_MODES = (
    "legacy_448_center",
    "p1_box_none_stretch",
    "bilinear_none_stretch",
    "center_256",
)

FROZEN_TEMPERATURE_SINGLE = 0.7848
FROZEN_TEMPERATURE_MULTI = 0.5146
FROZEN_ENERGY_THRESHOLD = -2.2082
FROZEN_MIN_TOP1_PROB = 0.50
FROZEN_DISAGREEMENT_THRESHOLD = 0.70
FROZEN_HIGH_STAKES_VETO_THRESHOLD = 0.20
ASYMMETRIC_TWIN_VETO_THRESHOLD = 0.10
ASYMMETRIC_TWIN_MIN_RATIO = 5.0


def _assert_not_test_split(csv_path: str | Path) -> None:
    resolved = Path(csv_path).resolve()
    if resolved.name == "test.csv":
        raise ValueError(
            f"Refusing to evaluate test split {csv_path!r} during diagnostic ablation."
        )


class CalibOnlyOODDataset(Dataset):
    """Dataset reading ONLY `proposed_split == 'calib'` OOD samples (never `test`)."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform: Any = None,
        path_lookup: dict[str, Path] | None = None,
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
                if row.get("proposed_split") != "calib":
                    continue
                pid = row.get("photo_id")
                url = row.get("image_url")
                src = row.get("source_url")
                p = self.path_lookup.get(url) or self.path_lookup.get(pid) or self.path_lookup.get(src)
                if p and p.exists():
                    self.records.append(row)
                    self.image_paths.append(p)
                else:
                    raise FileNotFoundError(
                        f"OOD calib image not found for {self.csv_path.name}: {pid} / {url}"
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        p = self.image_paths[idx]
        with Image.open(p) as opened:
            oriented = ImageOps.exif_transpose(opened)
            image = (oriented if oriented is not None else opened).convert("RGB")
        tensor = self.transform(image) if self.transform else image
        return tensor, idx


def extract_pytorch_logits_66(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    verna_idx: int,
) -> np.ndarray:
    """Extract float64 logits [N, 66] from PyTorch model with Amanita verna sliced out."""
    all_logits = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out67 = model(images)
            else:
                out67 = model(images)
            if verna_idx >= 0:
                out66 = torch.cat([out67[:, :verna_idx], out67[:, verna_idx + 1 :]], dim=1)
            else:
                out66 = out67
            all_logits.append(out66.cpu().double().numpy())
    return np.concatenate(all_logits, axis=0)


def extract_tflite_logits_66(
    runner: TFLiteRunner,
    loader: DataLoader,
) -> np.ndarray:
    """Extract float64 logits [N, 66] from TFLite FP16 runner."""
    all_logits = []
    for images, _ in loader:
        nhwc = images.permute(0, 2, 3, 1).numpy().astype(np.float32)
        all_logits.append(runner.predict_batch(nhwc).astype(np.float64))
    return np.concatenate(all_logits, axis=0)


def evaluate_variant_metrics(
    *,
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    val_records: list[dict[str, Any]],
    verna_logits: np.ndarray,
    ood_calib_fungi_logits: np.ndarray,
    ood_calib_nonfungi_logits: np.ndarray,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    edible_slugs: set[str],
    toxic_slugs: set[str],
    high_stakes_slugs: set[str],
    dangerous_pairs: Sequence[tuple[str, str]],
    temperature_single: float = FROZEN_TEMPERATURE_SINGLE,
    temperature_multi: float = FROZEN_TEMPERATURE_MULTI,
    energy_threshold: float = FROZEN_ENERGY_THRESHOLD,
    min_top1_prob: float = FROZEN_MIN_TOP1_PROB,
    tflite_val_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute single-frame, OOD calib, and P5 veto metrics for one preprocessing variant."""
    num_classes = len(class_to_idx)
    preds = np.argmax(val_logits, axis=1)
    overall_top1_raw = float(np.mean(preds == val_targets))

    top5_idx = np.argsort(-val_logits, axis=1)[:, : min(5, num_classes)]
    overall_top5_raw = float(
        np.mean([val_targets[i] in top5_idx[i] for i in range(len(val_targets))])
    )

    per_class_recalls = []
    for c in range(num_classes):
        mask = val_targets == c
        n_c = int(np.sum(mask))
        if n_c > 0:
            per_class_recalls.append(float(np.mean(preds[mask] == c)))
    macro_top1_raw = float(np.mean(per_class_recalls)) if per_class_recalls else 0.0

    # Calibration metrics at frozen T and variant-refitted T
    t_refit = fit_temperature_scaling(val_logits, val_targets, initial_t=1.0)
    nll_frozen = compute_nll(val_logits, val_targets, temperature_single)
    nll_refit = compute_nll(val_logits, val_targets, t_refit)
    ece_frozen, mce_frozen, _ = compute_ece_mce(val_logits, val_targets, temperature_single, n_bins=15)
    ece_refit, mce_refit, _ = compute_ece_mce(val_logits, val_targets, t_refit, n_bins=15)

    # Single-frame OOD gate at frozen thresholds
    val_e = compute_energy_vector(val_logits, temperature_single)
    val_probs = compute_softmax(val_logits, temperature_single)
    val_p = np.max(val_probs, axis=1)
    val_is_ood = val_e > energy_threshold
    val_is_unclear = (~val_is_ood) & (val_p < min_top1_prob)
    val_is_ok = (~val_is_ood) & (~val_is_unclear)

    keep_rate = float(np.mean(val_is_ok))
    cond_acc = float(np.mean(preds[val_is_ok] == val_targets[val_is_ok])) if np.any(val_is_ok) else 0.0

    toxic_true_mask = np.array([idx_to_class[t] in toxic_slugs for t in val_targets])
    edible_pred_mask = np.array([idx_to_class[p] in edible_slugs for p in preds])
    conf_toxic_as_edible_raw = int(np.sum(toxic_true_mask & edible_pred_mask))
    conf_toxic_as_edible_acc = int(np.sum(toxic_true_mask & edible_pred_mask & val_is_ok))

    # High-stakes recalls (n >= 8)
    hs_species_metrics: dict[str, Any] = {}
    hs_cond_recalls: list[float] = []
    hs_raw_recalls: list[float] = []
    hs_failing: list[str] = []
    for s in sorted(high_stakes_slugs):
        if s not in class_to_idx:
            continue
        idx = class_to_idx[s]
        mask = val_targets == idx
        n_s = int(np.sum(mask))
        if n_s == 0:
            continue
        mask_acc = mask & val_is_ok
        n_acc = int(np.sum(mask_acc))
        raw_rec = float(np.mean(preds[mask] == idx))
        cond_rec = float(np.mean(preds[mask_acc] == idx)) if n_acc > 0 else 0.0
        hs_raw_recalls.append(raw_rec)
        if n_s >= 8:
            hs_cond_recalls.append(cond_rec)
            if cond_rec < 0.85:
                hs_failing.append(s)
        hs_species_metrics[s] = {
            "n": n_s,
            "n_accepted": n_acc,
            "raw_recall": raw_rec,
            "conditional_recall": cond_rec,
        }

    # Dangerous pairs (42 active pairs)
    active_pairs = [(a, b) for a, b in dangerous_pairs if a in class_to_idx and b in class_to_idx]
    pair_details: dict[str, Any] = {}
    raw_violating: list[str] = []
    cond_violating: list[str] = []
    for a, b in active_pairs:
        ia, ib = class_to_idx[a], class_to_idx[b]
        ma, mb = val_targets == ia, val_targets == ib
        na, nb = int(np.sum(ma)), int(np.sum(mb))
        eab_raw = float(np.sum(preds[ma] == ib) / na) if na > 0 else 0.0
        eba_raw = float(np.sum(preds[mb] == ia) / nb) if nb > 0 else 0.0
        max_raw = max(eab_raw, eba_raw)
        if max_raw > 0.05:
            raw_violating.append(f"{a} <-> {b}")

        ma_acc, mb_acc = ma & val_is_ok, mb & val_is_ok
        na_acc, nb_acc = int(np.sum(ma_acc)), int(np.sum(mb_acc))
        eab_cond = float(np.sum(preds[ma_acc] == ib) / na_acc) if na_acc > 0 else 0.0
        eba_cond = float(np.sum(preds[mb_acc] == ia) / nb_acc) if nb_acc > 0 else 0.0
        max_cond = max(eab_cond, eba_cond)
        pair_key = f"{a} <-> {b}"
        if max_cond > 0.05:
            cond_violating.append(pair_key)

        pair_details[pair_key] = {
            "n_a": na,
            "n_b": nb,
            "n_a_accepted": na_acc,
            "n_b_accepted": nb_acc,
            "raw_max_err": max_raw,
            "cond_err_a_to_b": eab_cond,
            "cond_err_b_to_a": eba_cond,
            "cond_max_err": max_cond,
        }

    # OOD calib metrics (strictly calib partition)
    def _eval_ood_block(ood_logits: np.ndarray, temp: float, e_th: float) -> dict[str, Any]:
        e_o = compute_energy_vector(ood_logits, temp)
        p_o = compute_softmax(ood_logits, temp)
        p_top1 = np.max(p_o, axis=1)
        pred_o = np.argmax(ood_logits, axis=1)
        is_ood = e_o > e_th
        is_unclear = (~is_ood) & (p_top1 < min_top1_prob)
        is_ok = (~is_ood) & (~is_unclear)
        is_edible = np.array([idx_to_class[pr] in edible_slugs for pr in pred_o])
        ce_cnt = int(np.sum(is_ok & is_edible))
        n = len(ood_logits)
        return {
            "n_samples": n,
            "mean_energy": float(np.mean(e_o)) if n > 0 else 0.0,
            "energy_rejection_rate": float(np.mean(is_ood)) if n > 0 else 0.0,
            "unclear_rejection_rate": float(np.mean(is_unclear)) if n > 0 else 0.0,
            "acceptance_rate": float(np.mean(is_ok)) if n > 0 else 0.0,
            "confident_edible_count": ce_cnt,
            "confident_edible_rate": float(ce_cnt / n) if n > 0 else 0.0,
        }

    ood_calib_all = np.concatenate([ood_calib_fungi_logits, ood_calib_nonfungi_logits], axis=0)
    auroc_frozen, fpr95_frozen, th95_frozen = compute_auroc_and_fpr(
        val_e, compute_energy_vector(ood_calib_all, temperature_single), target_id_tpr=0.95
    )
    th85_frozen = float(np.percentile(val_e, 85.0, method="higher"))

    val_e_refit = compute_energy_vector(val_logits, t_refit)
    auroc_refit, fpr95_refit, _ = compute_auroc_and_fpr(
        val_e_refit, compute_energy_vector(ood_calib_all, t_refit), target_id_tpr=0.95
    )
    th85_refit = float(np.percentile(val_e_refit, 85.0, method="higher"))

    ood_calib_frozen = {
        "auroc": float(auroc_frozen),
        "fpr_at_95tpr": float(fpr95_frozen),
        "theta_at_95tpr": float(th95_frozen),
        "theta_at_85tpr": float(th85_frozen),
        "total": _eval_ood_block(ood_calib_all, temperature_single, energy_threshold),
        "fungi": _eval_ood_block(ood_calib_fungi_logits, temperature_single, energy_threshold),
        "nonfungi": _eval_ood_block(ood_calib_nonfungi_logits, temperature_single, energy_threshold),
        "at_85tpr_threshold": _eval_ood_block(ood_calib_all, temperature_single, th85_frozen),
    }
    ood_calib_refit = {
        "temperature_refit": float(t_refit),
        "auroc": float(auroc_refit),
        "fpr_at_95tpr": float(fpr95_refit),
        "theta_at_85tpr": float(th85_refit),
        "at_85tpr_threshold": _eval_ood_block(ood_calib_all, t_refit, th85_refit),
    }

    # Uncovered Amanita verna (val)
    verna_eval = _eval_ood_block(verna_logits, temperature_single, energy_threshold)

    # P5 observation-level evaluation: legacy veto (0.20) vs asymmetric twin-pair veto (0.10 + 5x)
    obs_map: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(val_records):
        oid = (
            r.get("observation_id")
            or r.get("inat_observation_id")
            or r.get("mo_observation_id")
            or r.get("source_url")
            or f"frame_{i}"
        )
        obs_map[oid].append(i)

    def _run_p5_mode(use_asymmetric: bool) -> dict[str, Any]:
        ok_obs = 0
        correct_ok = 0
        veto_obs = 0
        toxic_as_edible_obs = 0
        multi_total = 0
        multi_ok = 0
        multi_correct = 0
        multi_veto = 0
        multi_toxic_edible = 0
        obs_preds: dict[str, int] = {}
        obs_truth: dict[str, int] = {}

        for oid, idxs in obs_map.items():
            truth_idx = int(val_targets[idxs[0]])
            truth_slug = idx_to_class[truth_idx]
            res = fuse_observation_frames(
                logits=val_logits[idxs],
                idx_to_class=idx_to_class,
                class_to_idx=class_to_idx,
                edible_slugs=edible_slugs,
                high_stakes_slugs=high_stakes_slugs,
                temperature_frame=temperature_single,
                temperature_multi=temperature_multi,
                energy_threshold_frame=energy_threshold,
                energy_threshold_fused=energy_threshold,
                min_top1_prob=min_top1_prob,
                disagreement_prob_threshold=FROZEN_DISAGREEMENT_THRESHOLD,
                high_stakes_veto_threshold=FROZEN_HIGH_STAKES_VETO_THRESHOLD,
                veto_check_all_frames=True,
                dangerous_pairs=active_pairs if use_asymmetric else None,
                twin_pair_veto_threshold=ASYMMETRIC_TWIN_VETO_THRESHOLD,
                twin_pair_min_ratio=ASYMMETRIC_TWIN_MIN_RATIO,
            )
            is_multi = len(idxs) > 1
            if is_multi:
                multi_total += 1
            if res.status == "HIGH_STAKES_VETO":
                veto_obs += 1
                if is_multi:
                    multi_veto += 1
            if res.status == "OK" and res.predicted_idx is not None:
                ok_obs += 1
                obs_preds[oid] = res.predicted_idx
                obs_truth[oid] = truth_idx
                if res.predicted_idx == truth_idx:
                    correct_ok += 1
                if truth_slug in toxic_slugs and res.confident_edible:
                    toxic_as_edible_obs += 1
                if is_multi:
                    multi_ok += 1
                    if res.predicted_idx == truth_idx:
                        multi_correct += 1
                    if truth_slug in toxic_slugs and res.confident_edible:
                        multi_toxic_edible += 1

        pair_viol = 0
        max_pair_err = 0.0
        for a, b in active_pairs:
            ia, ib = class_to_idx[a], class_to_idx[b]
            a_ok = [oid for oid, t in obs_truth.items() if t == ia]
            b_ok = [oid for oid, t in obs_truth.items() if t == ib]
            eab = sum(1 for oid in a_ok if obs_preds[oid] == ib) / len(a_ok) if a_ok else 0.0
            eba = sum(1 for oid in b_ok if obs_preds[oid] == ia) / len(b_ok) if b_ok else 0.0
            m_err = max(eab, eba)
            max_pair_err = max(max_pair_err, m_err)
            if m_err > 0.05:
                pair_viol += 1

        return {
            "total_observations": len(obs_map),
            "accepted_ok_observations": ok_obs,
            "conditional_accuracy": float(correct_ok / ok_obs) if ok_obs > 0 else 0.0,
            "high_stakes_veto_count": veto_obs,
            "confident_toxic_as_edible_count": toxic_as_edible_obs,
            "dangerous_pairs_violating_count": pair_viol,
            "dangerous_pairs_max_err": float(max_pair_err),
            "multi_photo_subset": {
                "total_observations": multi_total,
                "accepted_ok_observations": multi_ok,
                "conditional_accuracy": float(multi_correct / multi_ok) if multi_ok > 0 else 0.0,
                "high_stakes_veto_count": multi_veto,
                "confident_toxic_as_edible_count": multi_toxic_edible,
            },
        }

    p5_legacy_veto = _run_p5_mode(use_asymmetric=False)
    p5_asymmetric_veto = _run_p5_mode(use_asymmetric=True)

    tflite_parity: dict[str, Any] | None = None
    if tflite_val_logits is not None:
        tf_preds = np.argmax(tflite_val_logits, axis=1)
        hs_indices = {class_to_idx[s] for s in high_stakes_slugs if s in class_to_idx}
        hs_mask = np.isin(val_targets, list(hs_indices))
        diff = np.abs(tflite_val_logits - val_logits)
        tflite_parity = {
            "top1_agreement_overall": float(np.mean(tf_preds == preds)),
            "top1_agreement_high_stakes": float(np.mean(tf_preds[hs_mask] == preds[hs_mask])) if np.any(hs_mask) else 1.0,
            "top1_accuracy_raw": float(np.mean(tf_preds == val_targets)),
            "logit_mae": float(np.mean(diff)),
            "logit_max_error": float(np.max(diff)),
        }

    worst_pairs = dict(
        sorted(pair_details.items(), key=lambda kv: kv[1]["cond_max_err"], reverse=True)[:12]
    )

    return {
        "val_samples": len(val_targets),
        "overall_top1_raw": overall_top1_raw,
        "overall_top5_raw": overall_top5_raw,
        "macro_top1_raw": macro_top1_raw,
        "calibration": {
            "temperature_frozen": float(temperature_single),
            "nll_frozen": float(nll_frozen),
            "ece_frozen": float(ece_frozen),
            "mce_frozen": float(mce_frozen),
            "temperature_refit": float(t_refit),
            "nll_refit": float(nll_refit),
            "ece_refit": float(ece_refit),
            "mce_refit": float(mce_refit),
        },
        "gated_single_frame_frozen": {
            "keep_rate": keep_rate,
            "energy_rejection_rate": float(np.mean(val_is_ood)),
            "unclear_rejection_rate": float(np.mean(val_is_unclear)),
            "conditional_accuracy": cond_acc,
            "confident_toxic_as_edible_accepted": conf_toxic_as_edible_acc,
            "confident_toxic_as_edible_raw": conf_toxic_as_edible_raw,
        },
        "high_stakes_recall": {
            "total_evaluated": len(hs_species_metrics),
            "raw_min_recall": float(min(hs_raw_recalls)) if hs_raw_recalls else 0.0,
            "raw_mean_recall": float(np.mean(hs_raw_recalls)) if hs_raw_recalls else 0.0,
            "cond_min_recall_n_ge_8": float(min(hs_cond_recalls)) if hs_cond_recalls else 0.0,
            "cond_mean_recall_n_ge_8": float(np.mean(hs_cond_recalls)) if hs_cond_recalls else 0.0,
            "failing_species_count": len(hs_failing),
            "failing_species": hs_failing,
        },
        "dangerous_pairs": {
            "active_pairs_count": len(active_pairs),
            "raw_violating_count": len(raw_violating),
            "raw_max_err": float(max((v["raw_max_err"] for v in pair_details.values()), default=0.0)),
            "cond_violating_count": len(cond_violating),
            "cond_failing_pairs": cond_violating,
            "cond_max_err": float(max((v["cond_max_err"] for v in pair_details.values()), default=0.0)),
            "worst_pairs": worst_pairs,
            "all_pairs": pair_details,
        },
        "ood_calib_frozen": ood_calib_frozen,
        "ood_calib_refit": ood_calib_refit,
        "amanita_verna_val": verna_eval,
        "p5_veto_comparison": {
            "legacy_veto_020": p5_legacy_veto,
            "asymmetric_twin_veto_010_5x": p5_asymmetric_veto,
        },
        "tflite_fp16_parity": tflite_parity,
    }


def render_ablation_markdown(report: dict[str, Any]) -> str:
    """Render comparative Markdown report for Step 1 (CenterCrop vs none_stretch + P5 veto)."""
    lines = [
        "# Raport Diagnostyczny Krok 1: Ablacja Preprocessingu (`CenterCrop` vs `none_stretch`) oraz Asymetryczne Weto P5",
        "",
        f"**Data raportu:** `{report['timestamp']}`",
        f"**Checkpoint PyTorch:** `{report['checkpoint']}` (`sha256: {report['checkpoint_sha256'][:16]}...`)",
        f"**Model TFLite FP16:** `{report['tflite_model']}` (`sha256: {report['tflite_sha256'][:16]}...`)",
        "**Zbiory użyte w analizie:** wyłącznie `data/splits/val.csv` oraz `ood_calib` (`proposed_split == \"calib\"`). Zbiory `test.csv` oraz `ood_test` pozostały w 100% nienaruszone.",
        "",
        "## 1. Porównanie wariantów preprocessingu na `val.csv` i `ood_calib` (zamrożony checkpoint `6992f5f1`)",
        "",
        "| Wariant | Top-1 Raw | Macro Top-1 | Keep Rate | Cond. Acc | Toxic→Edible (acc) | HS Min Recall | Failing Pairs (>5%) | `ood_calib` CE (frozen θ) | `ood_calib` CE (@85% TPR θ) | `ood_calib` AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, data in report["variants"].items():
        g = data["gated_single_frame_frozen"]
        hs = data["high_stakes_recall"]
        dp = data["dangerous_pairs"]
        ood_f = data["ood_calib_frozen"]
        ood_85 = ood_f["at_85tpr_threshold"]
        lines.append(
            f"| `{mode}` | {data['overall_top1_raw']*100:.2f}% | {data['macro_top1_raw']*100:.2f}% | "
            f"{g['keep_rate']*100:.2f}% | {g['conditional_accuracy']*100:.2f}% | "
            f"{g['confident_toxic_as_edible_accepted']} | {hs['cond_min_recall_n_ge_8']*100:.2f}% ({hs['failing_species_count']} fail) | "
            f"{dp['cond_violating_count']}/{dp['active_pairs_count']} (max {dp['cond_max_err']*100:.2f}%) | "
            f"{ood_f['total']['confident_edible_rate']*100:.2f}% ({ood_f['total']['confident_edible_count']}/{ood_f['total']['n_samples']}) | "
            f"{ood_85['confident_edible_rate']*100:.2f}% ({ood_85['confident_edible_count']}/{ood_85['n_samples']}, θ={ood_f['theta_at_85tpr']:.4f}) | "
            f"{ood_f['auroc']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Weryfikacja TFLite FP16 vs PyTorch na `val.csv`",
        "",
        "| Wariant | Top-1 Agreement (All) | Top-1 Agreement (High-Stakes) | Logit MAE | Max Logit Err |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode, data in report["variants"].items():
        tp = data.get("tflite_fp16_parity")
        if tp:
            lines.append(
                f"| `{mode}` | {tp['top1_agreement_overall']*100:.2f}% | {tp['top1_agreement_high_stakes']*100:.2f}% | "
                f"{tp['logit_mae']:.4f} | {tp['logit_max_error']:.4f} |"
            )

    lines.extend([
        "",
        "## 3. Wpływ asymetrycznego weta sobowtórów P5 (`p_t < 0.10` oraz `p_e >= 5 * p_t`) na `val.csv`",
        "",
        "| Wariant preprocessingu | Tryb weta P5 | OK Obs. | Cond. Acc (Obs) | Veto Count | Toxic→Edible (Obs) | Multi Cond. Acc | Multi Toxic→Edible | Failing Pairs (>5%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode, data in report["variants"].items():
        for label, key in [("Legacy Veto (0.20)", "legacy_veto_020"), ("Asymmetric Twin Veto (0.10 + 5x)", "asymmetric_twin_veto_010_5x")]:
            p5 = data["p5_veto_comparison"][key]
            mp = p5["multi_photo_subset"]
            lines.append(
                f"| `{mode}` | {label} | {p5['accepted_ok_observations']}/{p5['total_observations']} | "
                f"{p5['conditional_accuracy']*100:.2f}% | {p5['high_stakes_veto_count']} | "
                f"{p5['confident_toxic_as_edible_count']} | {mp['conditional_accuracy']*100:.2f}% ({mp['accepted_ok_observations']}/{mp['total_observations']}) | "
                f"{mp['confident_toxic_as_edible_count']} | {p5['dangerous_pairs_violating_count']} (max {p5['dangerous_pairs_max_err']*100:.2f}%) |"
            )

    lines.extend([
        "",
        "## 4. Najtrudniejsze pary `DANGEROUS_PAIRS` (`legacy_448_center` vs `p1_box_none_stretch`)",
        "",
        "| Para niebezpieczna | `legacy_448_center` max err | `p1_box_none_stretch` max err | Delta (pp) |",
        "|---|---:|---:|---:|",
    ])
    if "legacy_448_center" in report["variants"] and "p1_box_none_stretch" in report["variants"]:
        leg_dp = report["variants"]["legacy_448_center"]["dangerous_pairs"]
        p1_dp = report["variants"]["p1_box_none_stretch"]["dangerous_pairs"]
        leg_pairs = leg_dp.get("all_pairs") or leg_dp["worst_pairs"]
        p1_pairs = p1_dp.get("all_pairs") or p1_dp["worst_pairs"]
        all_keys = sorted(set(leg_pairs) | set(p1_pairs))
        # Sort by max error across both variants so regressions in either variant are never hidden
        all_keys.sort(
            key=lambda k: max(
                leg_pairs.get(k, {}).get("cond_max_err", 0.0),
                p1_pairs.get(k, {}).get("cond_max_err", 0.0),
            ),
            reverse=True,
        )
        failing_union = [
            k
            for k in all_keys
            if leg_pairs.get(k, {}).get("cond_max_err", 0.0) > 0.05
            or p1_pairs.get(k, {}).get("cond_max_err", 0.0) > 0.05
        ]
        display_keys = failing_union if len(failing_union) >= 10 else all_keys[: max(10, len(failing_union))]
        for k in display_keys:
            e_leg = leg_pairs.get(k, {}).get("cond_max_err")
            e_p1 = p1_pairs.get(k, {}).get("cond_max_err")
            leg_str = f"{e_leg*100:.2f}%" if e_leg is not None else "n/a"
            p1_str = f"{e_p1*100:.2f}%" if e_p1 is not None else "n/a"
            delta_str = f"{(e_p1 - e_leg)*100:+.2f} pp" if (e_leg is not None and e_p1 is not None) else "n/a"
            lines.append(f"| `{k}` | {leg_str} | {p1_str} | {delta_str} |")

    lines.append("")
    return "\n".join(lines)


def run_ablation(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    tflite_path: str | Path = DEFAULT_TFLITE,
    val_csv: str | Path = DEFAULT_VAL_CSV,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    ood_fungi_csv: str | Path = DEFAULT_OOD_FUNGI_CSV,
    ood_nonfungi_csv: str | Path = DEFAULT_OOD_NONFUNGI_CSV,
    karty_path: str | Path = DEFAULT_KARTY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    crop_modes: Sequence[str] = DEFAULT_CROP_MODES,
    tflite_modes: Sequence[str] = ("legacy_448_center", "p1_box_none_stretch"),
    batch_size: int = 128,
    num_workers: int = 8,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Run full preprocessing ablation strictly on val.csv and ood_calib."""
    _assert_not_test_split(val_csv)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    ckpt_p = Path(checkpoint_path)
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    input_size = int(ckpt["input_size"])
    class_to_idx_67 = ckpt["class_to_idx"]
    sorted_classes = sorted(class_to_idx_67.keys())
    verna_idx = class_to_idx_67.get("amanita_verna", -1)
    active_classes = [c for c in sorted_classes if c != "amanita_verna"]
    class_to_idx_66 = {c: i for i, c in enumerate(active_classes)}
    idx_to_class_66 = {i: c for i, c in enumerate(active_classes)}

    model = create_classifier_model(model_name, num_classes=len(class_to_idx_67), pretrained=False).to(device)
    model.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
    model.eval()

    edible_slugs, toxic_slugs, _ = load_species_edibility_sets(karty_path)
    dangerous_pairs = load_dangerous_pairs(karty_path)
    high_stakes_slugs = load_high_stakes_species(karty_path)

    tflite_runner: TFLiteRunner | None = None
    tflite_p = Path(tflite_path)
    if tflite_p.exists():
        tflite_runner = TFLiteRunner(tflite_p, num_threads=max(1, num_workers))

    variants_report: dict[str, Any] = {}
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    for mode in crop_modes:
        logger.info("Running ablation for crop_mode=%s...", mode)
        transform = get_val_transforms(input_size=input_size, crop_mode=mode)

        val_ds = MushroomSplitDataset(
            csv_path=val_csv,
            image_dir=image_dir,
            transform=transform,
            class_to_idx=class_to_idx_66,
            mock_if_missing=False,
        )
        val_loader = DataLoader(val_ds, **loader_kwargs)

        verna_ds = MushroomSplitDataset(
            csv_path=val_csv,
            image_dir=image_dir,
            transform=transform,
            class_to_idx={"amanita_verna": 0},
            mock_if_missing=False,
        )
        verna_loader = DataLoader(verna_ds, batch_size=64, shuffle=False, num_workers=min(4, num_workers))

        shared_lookup = val_ds.path_lookup
        ood_f_ds = CalibOnlyOODDataset(ood_fungi_csv, image_dir=image_dir, transform=transform, path_lookup=shared_lookup)
        ood_f_loader = DataLoader(ood_f_ds, **loader_kwargs)

        ood_nf_ds = CalibOnlyOODDataset(ood_nonfungi_csv, image_dir=image_dir, transform=transform, path_lookup=shared_lookup)
        ood_nf_loader = DataLoader(ood_nf_ds, **loader_kwargs)

        val_logits = extract_pytorch_logits_66(model, val_loader, device, verna_idx)
        val_targets = np.array([class_to_idx_66[r["species_slug"]] for r in val_ds.records], dtype=np.int64)
        verna_logits = extract_pytorch_logits_66(model, verna_loader, device, verna_idx)
        ood_f_logits = extract_pytorch_logits_66(model, ood_f_loader, device, verna_idx)
        ood_nf_logits = extract_pytorch_logits_66(model, ood_nf_loader, device, verna_idx)

        tf_val_logits: np.ndarray | None = None
        if tflite_runner is not None and mode in tflite_modes:
            tf_val_logits = extract_tflite_logits_66(tflite_runner, val_loader)

        if mode == "p1_box_none_stretch":
            cache_payload: dict[str, Any] = {
                "crop_mode": mode,
                "checkpoint_path": str(ckpt_p),
                "checkpoint_sha256": compute_file_sha256(ckpt_p),
                "tflite_path": str(tflite_p) if tflite_p.exists() else "",
                "tflite_sha256": compute_file_sha256(tflite_p) if tflite_p.exists() else "",
                "class_to_idx_66": class_to_idx_66,
                "idx_to_class_66": idx_to_class_66,
                "active_classes": active_classes,
                "val_logits": torch.from_numpy(val_logits.astype(np.float32)),
                "val_targets": val_targets.tolist(),
                "val_records": val_ds.records,
                "verna_logits": torch.from_numpy(verna_logits.astype(np.float32)),
                "verna_records": verna_ds.records,
                "ood_calib_fungi_logits": torch.from_numpy(ood_f_logits.astype(np.float32)),
                "ood_calib_fungi_records": ood_f_ds.records,
                "ood_calib_nonfungi_logits": torch.from_numpy(ood_nf_logits.astype(np.float32)),
                "ood_calib_nonfungi_records": ood_nf_ds.records,
            }
            if tf_val_logits is not None:
                cache_payload["tflite_val_logits"] = torch.from_numpy(tf_val_logits.astype(np.float32))
            torch.save(cache_payload, out_dir / "p1_box_none_stretch_val_cache.pt")

        variants_report[mode] = evaluate_variant_metrics(
            val_logits=val_logits,
            val_targets=val_targets,
            val_records=val_ds.records,
            verna_logits=verna_logits,
            ood_calib_fungi_logits=ood_f_logits,
            ood_calib_nonfungi_logits=ood_nf_logits,
            class_to_idx=class_to_idx_66,
            idx_to_class=idx_to_class_66,
            edible_slugs=edible_slugs,
            toxic_slugs=toxic_slugs,
            high_stakes_slugs=high_stakes_slugs,
            dangerous_pairs=dangerous_pairs,
            tflite_val_logits=tf_val_logits,
        )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": compute_file_sha256(ckpt_p),
        "tflite_model": str(tflite_path),
        "tflite_sha256": compute_file_sha256(tflite_p) if tflite_p.exists() else "",
        "val_csv": str(val_csv),
        "val_csv_sha256": compute_file_sha256(val_csv),
        "ood_fungi_csv_sha256": compute_file_sha256(ood_fungi_csv),
        "ood_nonfungi_csv_sha256": compute_file_sha256(ood_nonfungi_csv),
        "frozen_thresholds": {
            "temperature_single": FROZEN_TEMPERATURE_SINGLE,
            "temperature_multi": FROZEN_TEMPERATURE_MULTI,
            "energy_threshold": FROZEN_ENERGY_THRESHOLD,
            "min_top1_prob": FROZEN_MIN_TOP1_PROB,
            "twin_pair_veto_threshold": ASYMMETRIC_TWIN_VETO_THRESHOLD,
            "twin_pair_min_ratio": ASYMMETRIC_TWIN_MIN_RATIO,
        },
        "variants": variants_report,
    }

    json_path = out_dir / "preprocessing_ablation_val.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = out_dir / "preprocessing_ablation_report.md"
    md_path.write_text(render_ablation_markdown(report), encoding="utf-8")
    logger.info("Saved preprocessing ablation report to %s and %s", json_path, md_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preprocessing ablation and asymmetric veto comparison on val.csv & ood_calib."
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tflite", type=str, default=str(DEFAULT_TFLITE))
    parser.add_argument("--val-csv", type=str, default=str(DEFAULT_VAL_CSV))
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--ood-fungi-csv", type=str, default=str(DEFAULT_OOD_FUNGI_CSV))
    parser.add_argument("--ood-nonfungi-csv", type=str, default=str(DEFAULT_OOD_NONFUNGI_CSV))
    parser.add_argument("--karty-path", type=str, default=str(DEFAULT_KARTY_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_ablation(
        checkpoint_path=args.checkpoint,
        tflite_path=args.tflite,
        val_csv=args.val_csv,
        image_dir=args.image_dir,
        ood_fungi_csv=args.ood_fungi_csv,
        ood_nonfungi_csv=args.ood_nonfungi_csv,
        karty_path=args.karty_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
