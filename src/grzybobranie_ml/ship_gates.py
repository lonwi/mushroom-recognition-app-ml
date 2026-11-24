"""Final Ship Gates Audit module (Stage P6 / P8).

Performs the single, frozen, definitive audit across:
- In-distribution test set (data/splits/test.csv)
- Out-of-distribution test set (data/ood-v1 test split: 2,230 images)
- Uncovered species test set (Amanita verna test samples: 11 images)
- Multi-photo fusion observations (1,968 multi-photo observations)
- LiteRT TFLite FP16 vs PyTorch agreement

Evaluates strict fail-closed safety criteria:
1. confident_toxic_as_edible == 0 (single frame and multi-photo)
2. per-pair confusion <= 0.05 across all active DANGEROUS_PAIRS
3. high_stakes_recall_min >= 0.85 (for n >= 8) with Wilson 95% lower bounds
4. tfliteAgreement >= 0.99 (overall and high-stakes)
5. ood_test open-set confident edible <= 0.02 (Wilson upper bound <= 0.05)
6. multi-photo fusion safety and veto validation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from grzybobranie_ml.calibrate_ood import (
    compute_ece_mce,
    compute_energy_vector,
    compute_softmax,
)
from grzybobranie_ml.export_tflite import (
    TFLiteRunner,
    compute_file_sha256,
    load_and_prepare_model,
)
from grzybobranie_ml.multi_photo import (
    MultiPhotoResult,
    fuse_observation_frames,
    load_species_edibility_sets,
)
from grzybobranie_ml.train import (
    MushroomSplitDataset,
    get_val_transforms,
    load_dangerous_pairs,
    load_high_stakes_species,
)

logger = logging.getLogger("grzybobranie_ml.ship_gates")

DEFAULT_CHECKPOINT = Path(
    "artifacts/train_runs/mobilenetv4_conv_medium.e500_r256_in1k_r256_1795514748/best_val_checkpoint.pt"
)
DEFAULT_TFLITE = Path("artifacts/models/mobilenetv4_medium_r256_fp16.tflite")
DEFAULT_TEST_CSV = Path("data/splits/test.csv")
DEFAULT_IMAGE_DIR = Path("/home/bubolka/hf-768/upload")
DEFAULT_OOD_FUNGI_CSV = Path("data/ood-v1/ood_fungi.csv")
DEFAULT_OOD_NONFUNGI_CSV = Path("data/ood-v1/ood_nonfungi.csv")
DEFAULT_KARTY_PATH = Path("configs/karty_sobowtory.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/ship_gates")

# Frozen Recommendation B parameters
FROZEN_TEMPERATURE_SINGLE = 0.7848
FROZEN_TEMPERATURE_MULTI = 0.5146
FROZEN_ENERGY_THRESHOLD = -2.2082
FROZEN_MIN_TOP1_PROB = 0.50
FROZEN_DISAGREEMENT_THRESHOLD = 0.70
FROZEN_HIGH_STAKES_VETO_THRESHOLD = 0.20


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    spread = z * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    lower = 0.0 if k == 0 else max(0.0, float((center - spread) / denom))
    upper = 1.0 if k == n else min(1.0, float((center + spread) / denom))
    return lower, upper


def extract_pytorch_test_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract raw logits [N, K] from PyTorch model."""
    logger.info("Extracting PyTorch logits on test dataset (%d batches)...", len(loader))
    all_logits = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = model(images)
            else:
                out = model(images)
            all_logits.append(out.cpu().float().numpy())
    return np.concatenate(all_logits, axis=0)


def extract_tflite_test_logits(
    runner: TFLiteRunner,
    loader: DataLoader,
) -> np.ndarray:
    """Extract raw logits [N, K] from TFLite runner."""
    logger.info("Extracting TFLite FP16 logits on test dataset (%d batches)...", len(loader))
    all_logits = []
    for images, _ in loader:
        nhwc = images.permute(0, 2, 3, 1).numpy().astype(np.float32)
        batch_logits = runner.predict_batch(nhwc)
        all_logits.append(batch_logits)
    return np.concatenate(all_logits, axis=0)


def evaluate_single_frame_audit(
    py_logits: np.ndarray,
    tflite_logits: np.ndarray,
    targets: np.ndarray,
    records: list[dict[str, Any]],
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    edible_slugs: set[str],
    toxic_slugs: set[str],
    high_stakes_slugs: set[str],
    dangerous_pairs: list[tuple[str, str]],
    temperature: float = FROZEN_TEMPERATURE_SINGLE,
    energy_threshold: float = FROZEN_ENERGY_THRESHOLD,
    min_top1_prob: float = FROZEN_MIN_TOP1_PROB,
) -> dict[str, Any]:
    """Execute deep single-frame Ship Gates evaluation on test set."""
    n_total = len(targets)
    assert len(py_logits) == n_total
    assert len(tflite_logits) == n_total

    # PyTorch gating
    py_e = compute_energy_vector(py_logits, temperature)
    py_probs = compute_softmax(py_logits, temperature)
    py_p = np.max(py_probs, axis=1)
    py_preds = np.argmax(py_logits, axis=1)

    py_is_ood = py_e > energy_threshold
    py_is_unclear = (~py_is_ood) & (py_p < min_top1_prob)
    py_is_accepted = (~py_is_ood) & (~py_is_unclear)

    # TFLite gating
    tf_e = compute_energy_vector(tflite_logits, temperature)
    tf_probs = compute_softmax(tflite_logits, temperature)
    tf_p = np.max(tf_probs, axis=1)
    tf_preds = np.argmax(tflite_logits, axis=1)

    tf_is_ood = tf_e > energy_threshold
    tf_is_unclear = (~tf_is_ood) & (tf_p < min_top1_prob)
    tf_is_accepted = (~tf_is_ood) & (~tf_is_unclear)

    # 1. TFLite vs PyTorch Agreement (Gate 4)
    top1_agreement_all = float(np.mean(tf_preds == py_preds))

    high_stakes_indices = {class_to_idx[s] for s in high_stakes_slugs if s in class_to_idx}
    hs_mask = np.isin(targets, list(high_stakes_indices))
    n_hs = int(np.sum(hs_mask))
    top1_agreement_hs = float(np.mean(tf_preds[hs_mask] == py_preds[hs_mask])) if n_hs > 0 else 1.0

    diff = np.abs(tflite_logits - py_logits)
    logit_mae = float(np.mean(diff))
    max_logit_err = float(np.max(diff))

    # 2. Confident Toxic as Edible (Gate 1)
    toxic_true_mask = np.array([idx_to_class[t] in toxic_slugs for t in targets])
    edible_pred_mask = np.array([idx_to_class[p] in edible_slugs for p in tf_preds])

    toxic_as_edible_raw = int(np.sum(toxic_true_mask & edible_pred_mask))
    toxic_as_edible_accepted = int(np.sum(toxic_true_mask & edible_pred_mask & tf_is_accepted))

    # Also for PyTorch
    py_edible_pred_mask = np.array([idx_to_class[p] in edible_slugs for p in py_preds])
    py_toxic_as_edible_accepted = int(np.sum(toxic_true_mask & py_edible_pred_mask & py_is_accepted))

    # 3. DANGEROUS_PAIRS Confusion (Gate 2)
    active_pairs = [
        (a, b) for a, b in dangerous_pairs
        if a in class_to_idx and b in class_to_idx
    ]
    pair_results: dict[str, Any] = {}
    failing_pairs: list[str] = []

    for a, b in active_pairs:
        ia, ib = class_to_idx[a], class_to_idx[b]
        mask_a = targets == ia
        mask_b = targets == ib
        na, nb = int(np.sum(mask_a)), int(np.sum(mask_b))

        # Raw confusion
        c_ab_raw = int(np.sum(tf_preds[mask_a] == ib)) if na > 0 else 0
        c_ba_raw = int(np.sum(tf_preds[mask_b] == ia)) if nb > 0 else 0
        rate_ab_raw = float(c_ab_raw / na) if na > 0 else 0.0
        rate_ba_raw = float(c_ba_raw / nb) if nb > 0 else 0.0

        # Accepted confusion (combined gate)
        mask_a_acc = mask_a & tf_is_accepted
        mask_b_acc = mask_b & tf_is_accepted
        na_acc = int(np.sum(mask_a_acc))
        nb_acc = int(np.sum(mask_b_acc))

        c_ab_acc = int(np.sum(tf_preds[mask_a_acc] == ib)) if na_acc > 0 else 0
        c_ba_acc = int(np.sum(tf_preds[mask_b_acc] == ia)) if nb_acc > 0 else 0

        # Conditional confusion on accepted
        cond_rate_ab = float(c_ab_acc / na_acc) if na_acc > 0 else 0.0
        cond_rate_ba = float(c_ba_acc / nb_acc) if nb_acc > 0 else 0.0
        # Effective confusion on total true class count
        eff_rate_ab = float(c_ab_acc / na) if na > 0 else 0.0
        eff_rate_ba = float(c_ba_acc / nb) if nb > 0 else 0.0

        max_confusion = max(cond_rate_ab, cond_rate_ba)
        pair_key = f"{a} <-> {b}"
        pair_pass = bool(max_confusion <= 0.05)
        if not pair_pass:
            failing_pairs.append(pair_key)

        pair_results[pair_key] = {
            "species_a": a,
            "species_b": b,
            "n_a": na,
            "n_b": nb,
            "n_a_accepted": na_acc,
            "n_b_accepted": nb_acc,
            "raw_confusion_a_to_b": rate_ab_raw,
            "raw_confusion_b_to_a": rate_ba_raw,
            "confusion_a_to_b_count": c_ab_acc,
            "confusion_b_to_a_count": c_ba_acc,
            "conditional_confusion_a_to_b": cond_rate_ab,
            "conditional_confusion_b_to_a": cond_rate_ba,
            "effective_confusion_a_to_b": eff_rate_ab,
            "effective_confusion_b_to_a": eff_rate_ba,
            "max_confusion": max_confusion,
            "passed": pair_pass,
        }

    # 4. High-Stakes Species Recall (Gate 3)
    high_stakes_results: dict[str, Any] = {}
    failing_hs_species: list[str] = []
    min_hs_recall = 1.0

    for s in sorted(high_stakes_slugs):
        if s not in class_to_idx:
            continue
        idx = class_to_idx[s]
        mask = targets == idx
        n = int(np.sum(mask))
        if n == 0:
            continue

        mask_acc = mask & tf_is_accepted
        n_acc = int(np.sum(mask_acc))

        raw_tp = int(np.sum(tf_preds[mask] == idx))
        acc_tp = int(np.sum(tf_preds[mask_acc] == idx))

        raw_recall = float(raw_tp / n)
        eff_recall = float(acc_tp / n)
        cond_recall = float(acc_tp / n_acc) if n_acc > 0 else 0.0

        w_low_raw, w_high_raw = compute_wilson_ci(raw_tp, n)
        w_low_eff, w_high_eff = compute_wilson_ci(acc_tp, n)
        w_low_cond, w_high_cond = compute_wilson_ci(acc_tp, n_acc) if n_acc > 0 else (0.0, 0.0)

        # Ship gate threshold applies when n >= 8
        species_pass = True
        if n >= 8:
            if cond_recall < 0.85:
                species_pass = False
                failing_hs_species.append(s)
            min_hs_recall = min(min_hs_recall, cond_recall)

        high_stakes_results[s] = {
            "n": n,
            "n_accepted": n_acc,
            "raw_tp": raw_tp,
            "accepted_tp": acc_tp,
            "raw_recall": raw_recall,
            "effective_recall": eff_recall,
            "conditional_recall": cond_recall,
            "wilson_ci_raw": [w_low_raw, w_high_raw],
            "wilson_ci_effective": [w_low_eff, w_high_eff],
            "wilson_ci_conditional": [w_low_cond, w_high_cond],
            "wilson_lower_bound_95": w_low_cond,
            "passed": species_pass,
        }

    # 5. Overall General Metrics (Accuracy, ECE)
    raw_acc_tf = float(np.mean(tf_preds == targets))
    raw_acc_py = float(np.mean(py_preds == targets))
    cond_acc_tf = float(np.mean(tf_preds[tf_is_accepted] == targets[tf_is_accepted]))
    cond_acc_py = float(np.mean(py_preds[py_is_accepted] == targets[py_is_accepted]))

    ece_tf, mce_tf, _ = compute_ece_mce(tflite_logits, targets, temperature=temperature, n_bins=15)
    ece_py, mce_py, _ = compute_ece_mce(py_logits, targets, temperature=temperature, n_bins=15)

    return {
        "dataset_size": n_total,
        "keep_rate_tflite": float(np.mean(tf_is_accepted)),
        "keep_rate_pytorch": float(np.mean(py_is_accepted)),
        "energy_rejection_rate_tflite": float(np.mean(tf_is_ood)),
        "unclear_rejection_rate_tflite": float(np.mean(tf_is_unclear)),
        "accuracy_raw_tflite": raw_acc_tf,
        "accuracy_raw_pytorch": raw_acc_py,
        "accuracy_conditional_tflite": cond_acc_tf,
        "accuracy_conditional_pytorch": cond_acc_py,
        "ece_tflite": float(ece_tf),
        "ece_pytorch": float(ece_py),
        "mce_tflite": float(mce_tf),
        "mce_pytorch": float(mce_py),
        "tflite_agreement_overall": top1_agreement_all,
        "tflite_agreement_high_stakes": top1_agreement_hs,
        "tflite_logit_mae": logit_mae,
        "tflite_max_logit_error": max_logit_err,
        "confident_toxic_as_edible_accepted": toxic_as_edible_accepted,
        "confident_toxic_as_edible_raw": toxic_as_edible_raw,
        "pytorch_confident_toxic_as_edible_accepted": py_toxic_as_edible_accepted,
        "dangerous_pairs": {
            "total_active_pairs": len(active_pairs),
            "failing_pairs_count": len(failing_pairs),
            "failing_pairs": failing_pairs,
            "pairs": pair_results,
        },
        "high_stakes_recall": {
            "total_evaluated": len(high_stakes_results),
            "min_conditional_recall": min_hs_recall,
            "failing_species_count": len(failing_hs_species),
            "failing_species": failing_hs_species,
            "species": high_stakes_results,
        },
    }


def evaluate_ood_test_audit(
    ood_fungi_logits: np.ndarray,
    ood_fungi_records: list[dict[str, Any]],
    ood_nonfungi_logits: np.ndarray,
    ood_nonfungi_records: list[dict[str, Any]],
    idx_to_class: dict[int, str],
    edible_slugs: set[str],
    temperature: float = FROZEN_TEMPERATURE_SINGLE,
    energy_threshold: float = FROZEN_ENERGY_THRESHOLD,
    min_top1_prob: float = FROZEN_MIN_TOP1_PROB,
) -> dict[str, Any]:
    """Evaluate OOD test split (strictly proposed_split == 'test')."""
    f_test_mask = (
        np.array([r.get("proposed_split") == "test" for r in ood_fungi_records])
        if len(ood_fungi_records) > 0
        else np.array([], dtype=bool)
    )
    nf_test_mask = (
        np.array([r.get("proposed_split") == "test" for r in ood_nonfungi_records])
        if len(ood_nonfungi_records) > 0
        else np.array([], dtype=bool)
    )

    f_logits = (
        ood_fungi_logits[f_test_mask]
        if np.sum(f_test_mask) > 0
        else np.zeros((0, ood_fungi_logits.shape[1] if ood_fungi_logits.ndim == 2 else 66), dtype=np.float32)
    )
    nf_logits = (
        ood_nonfungi_logits[nf_test_mask]
        if np.sum(nf_test_mask) > 0
        else np.zeros((0, ood_nonfungi_logits.shape[1] if ood_nonfungi_logits.ndim == 2 else 66), dtype=np.float32)
    )

    ood_logits = (
        np.concatenate([f_logits, nf_logits], axis=0)
        if (len(f_logits) > 0 or len(nf_logits) > 0)
        else np.zeros((0, 66), dtype=np.float32)
    )
    n_ood = len(ood_logits)

    e = compute_energy_vector(ood_logits, temperature)
    probs = compute_softmax(ood_logits, temperature)
    p = np.max(probs, axis=1)
    preds = np.argmax(ood_logits, axis=1)

    is_ood = e > energy_threshold
    is_unclear = (~is_ood) & (p < min_top1_prob)
    is_accepted = (~is_ood) & (~is_unclear)

    edible_pred_mask = np.array([idx_to_class[pr] in edible_slugs for pr in preds])
    conf_edible_mask = is_accepted & edible_pred_mask
    conf_edible_count = int(np.sum(conf_edible_mask))
    conf_edible_rate = float(conf_edible_count / n_ood)

    wilson_low, wilson_high = compute_wilson_ci(conf_edible_count, n_ood)

    return {
        "n_samples": n_ood,
        "n_fungi_test": len(f_logits),
        "n_nonfungi_test": len(nf_logits),
        "energy_rejection_count": int(np.sum(is_ood)),
        "energy_rejection_rate": float(np.mean(is_ood)),
        "unclear_rejection_count": int(np.sum(is_unclear)),
        "unclear_rejection_rate": float(np.mean(is_unclear)),
        "accepted_count": int(np.sum(is_accepted)),
        "accepted_rate": float(np.mean(is_accepted)),
        "confident_edible_count": conf_edible_count,
        "confident_edible_rate": conf_edible_rate,
        "confident_edible_rate_pct": round(conf_edible_rate * 100.0, 2),
        "wilson_ci_95": [wilson_low, wilson_high],
        "wilson_upper_bound_95": wilson_high,
        "passed": bool(conf_edible_rate <= 0.02 and wilson_high <= 0.05),
    }


def evaluate_uncovered_species_verna(
    verna_logits: np.ndarray,
    idx_to_class: dict[int, str],
    edible_slugs: set[str],
    temperature: float = FROZEN_TEMPERATURE_SINGLE,
    energy_threshold: float = FROZEN_ENERGY_THRESHOLD,
    min_top1_prob: float = FROZEN_MIN_TOP1_PROB,
) -> dict[str, Any]:
    """Audit uncovered deadly species Amanita verna on test samples."""
    n = len(verna_logits)
    e = compute_energy_vector(verna_logits, temperature)
    probs = compute_softmax(verna_logits, temperature)
    p = np.max(probs, axis=1)
    preds = np.argmax(verna_logits, axis=1)

    is_ood = e > energy_threshold
    is_unclear = (~is_ood) & (p < min_top1_prob)
    is_accepted = (~is_ood) & (~is_unclear)

    edible_pred_mask = np.array([idx_to_class[pr] in edible_slugs for pr in preds])
    conf_edible_count = int(np.sum(is_accepted & edible_pred_mask))

    return {
        "n_samples": n,
        "energy_rejected": int(np.sum(is_ood)),
        "unclear_rejected": int(np.sum(is_unclear)),
        "accepted": int(np.sum(is_accepted)),
        "confident_toxic_as_edible": conf_edible_count,
        "passed": bool(conf_edible_count == 0),
    }


def evaluate_multi_photo_audit(
    records: list[dict[str, Any]],
    logits: np.ndarray,
    class_to_idx: dict[str, int],
    idx_to_class: dict[int, str],
    edible_slugs: set[str],
    toxic_slugs: set[str],
    high_stakes_slugs: set[str],
    temperature_frame: float = FROZEN_TEMPERATURE_SINGLE,
    temperature_multi: float = FROZEN_TEMPERATURE_MULTI,
    energy_threshold_frame: float = FROZEN_ENERGY_THRESHOLD,
    min_top1_prob: float = FROZEN_MIN_TOP1_PROB,
    disagreement_threshold: float = FROZEN_DISAGREEMENT_THRESHOLD,
    veto_threshold: float = FROZEN_HIGH_STAKES_VETO_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate multi-frame observation prediction fusion on test.csv."""
    obs_map: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        oid = (
            r.get("observation_id")
            or r.get("inat_observation_id")
            or r.get("mo_observation_id")
            or r.get("source_url")
            or f"frame_{i}"
        )
        obs_map[oid].append(i)

    multi_obs = {oid: idxs for oid, idxs in obs_map.items() if len(idxs) > 1}
    single_obs = {oid: idxs for oid, idxs in obs_map.items() if len(idxs) == 1}

    logger.info(
        "Multi-photo test evaluation: %d total observations (%d multi-photo with %d frames, %d single-photo)",
        len(obs_map),
        len(multi_obs),
        sum(len(v) for v in multi_obs.values()),
        len(single_obs),
    )

    fused_results: list[dict[str, Any]] = []
    correct_count = 0
    accepted_count = 0
    fused_toxic_as_edible = 0
    conflict_detected_count = 0
    veto_triggered_count = 0
    all_ood_count = 0

    for oid, frame_indices in multi_obs.items():
        sub_logits = logits[frame_indices]
        true_slugs = {records[idx]["species_slug"] for idx in frame_indices}
        ground_truth = true_slugs.pop() if len(true_slugs) == 1 else records[frame_indices[0]]["species_slug"]
        is_true_toxic = ground_truth in toxic_slugs

        res: MultiPhotoResult = fuse_observation_frames(
            logits=sub_logits,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            edible_slugs=edible_slugs,
            high_stakes_slugs=high_stakes_slugs,
            temperature_frame=temperature_frame,
            temperature_multi=temperature_multi,
            energy_threshold_frame=energy_threshold_frame,
            energy_threshold_fused=energy_threshold_frame,
            min_top1_prob=min_top1_prob,
            disagreement_prob_threshold=disagreement_threshold,
            high_stakes_veto_threshold=veto_threshold,
            veto_check_all_frames=True,
        )

        if res.status == "CONFLICT_DETECTED":
            conflict_detected_count += 1
        elif res.status == "HIGH_STAKES_VETO":
            veto_triggered_count += 1
        elif res.status == "ALL_FRAMES_OOD":
            all_ood_count += 1

        is_accepted = res.status == "OK"
        if is_accepted:
            accepted_count += 1
            if res.predicted_slug == ground_truth:
                correct_count += 1
            if is_true_toxic and (res.predicted_slug in edible_slugs):
                fused_toxic_as_edible += 1

        fused_results.append({
            "observation_id": oid,
            "num_frames": len(frame_indices),
            "ground_truth": ground_truth,
            "status": res.status,
            "predicted_slug": res.predicted_slug,
            "confidence": res.confidence,
        })

    n_multi = len(multi_obs)
    cond_acc = float(correct_count / accepted_count) if accepted_count > 0 else 0.0
    eff_acc = float(correct_count / n_multi) if n_multi > 0 else 0.0

    return {
        "n_multi_observations": n_multi,
        "n_multi_frames_total": sum(len(v) for v in multi_obs.values()),
        "accepted_count": accepted_count,
        "keep_rate": float(accepted_count / n_multi) if n_multi > 0 else 0.0,
        "correct_count": correct_count,
        "conditional_accuracy": cond_acc,
        "effective_accuracy": eff_acc,
        "conflict_detected_count": conflict_detected_count,
        "high_stakes_veto_count": veto_triggered_count,
        "all_frames_ood_count": all_ood_count,
        "fused_toxic_as_edible_count": fused_toxic_as_edible,
        "passed": bool(fused_toxic_as_edible == 0 and cond_acc >= 0.85),
    }


def run_final_ship_gates_audit(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    tflite_path: str | Path = DEFAULT_TFLITE,
    test_csv: str | Path = DEFAULT_TEST_CSV,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    ood_fungi_csv: str | Path = DEFAULT_OOD_FUNGI_CSV,
    ood_nonfungi_csv: str | Path = DEFAULT_OOD_NONFUNGI_CSV,
    karty_path: str | Path = DEFAULT_KARTY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Execute complete Stage P6/P8 final Ship Gates audit."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # 1. Load species rules
    edible_slugs, toxic_slugs, _lethal = load_species_edibility_sets(karty_path)
    dangerous_pairs = load_dangerous_pairs(karty_path)
    high_stakes_slugs = load_high_stakes_species(karty_path)

    # 2. Load PyTorch model and TFLite runner
    wrapped_model, meta = load_and_prepare_model(checkpoint_path, exclude_verna=True)
    input_size = meta["input_size"]
    class_to_idx = meta["class_to_idx"]
    idx_to_class = meta["idx_to_class"]

    # Unwrap for standard NCHW batch loader
    pt_model = wrapped_model.model.to(device).eval()
    tflite_runner = TFLiteRunner(tflite_path, num_threads=8)

    # 3. Load test.csv dataset (active 66 classes)
    test_ds = MushroomSplitDataset(
        csv_path=test_csv,
        image_dir=image_dir,
        transform=get_val_transforms(input_size),
        class_to_idx=class_to_idx,
        mock_if_missing=False,
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=device.type == "cuda")

    # 4. Extract PyTorch & TFLite logits on test.csv
    py_test_logits = extract_pytorch_test_logits(pt_model, test_loader, device)
    tf_test_logits = extract_tflite_test_logits(tflite_runner, test_loader)
    test_targets = np.array([class_to_idx[r["species_slug"]] for r in test_ds.records], dtype=np.int64)

    # 5. Single-Frame Ship Gates Audit
    single_audit = evaluate_single_frame_audit(
        py_logits=py_test_logits,
        tflite_logits=tf_test_logits,
        targets=test_targets,
        records=test_ds.records,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        edible_slugs=edible_slugs,
        toxic_slugs=toxic_slugs,
        high_stakes_slugs=high_stakes_slugs,
        dangerous_pairs=dangerous_pairs,
    )

    # 6. Amanita verna test samples (uncovered deadly species)
    verna_ds = MushroomSplitDataset(
        csv_path=test_csv,
        image_dir=image_dir,
        transform=get_val_transforms(input_size),
        class_to_idx={"amanita_verna": 0},
        mock_if_missing=False,
    )
    verna_loader = DataLoader(verna_ds, batch_size=64, shuffle=False, num_workers=4)
    verna_py_logits = extract_pytorch_test_logits(pt_model, verna_loader, device)
    verna_audit = evaluate_uncovered_species_verna(
        verna_logits=verna_py_logits,
        idx_to_class=idx_to_class,
        edible_slugs=edible_slugs,
    )

    # 7. OOD Test Audit (2,230 images)
    # Using the cached OOD logits from Stage P4 (which were precomputed on the exact same checkpoint)
    extracted_calib_cache = torch.load("artifacts/calibration/extracted_logits.pt", map_location="cpu", weights_only=False)
    ood_audit = evaluate_ood_test_audit(
        ood_fungi_logits=extracted_calib_cache["ood_fungi_logits"].float().numpy(),
        ood_fungi_records=extracted_calib_cache["ood_fungi_records"],
        ood_nonfungi_logits=extracted_calib_cache["ood_nonfungi_logits"].float().numpy(),
        ood_nonfungi_records=extracted_calib_cache["ood_nonfungi_records"],
        idx_to_class=idx_to_class,
        edible_slugs=edible_slugs,
    )

    # 8. Multi-Photo Pipeline Audit (P5) on test.csv
    multi_audit = evaluate_multi_photo_audit(
        records=test_ds.records,
        logits=tf_test_logits,  # evaluate on production TFLite FP16 logits
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        edible_slugs=edible_slugs,
        toxic_slugs=toxic_slugs,
        high_stakes_slugs=high_stakes_slugs,
    )

    # 9. Gate evaluation synthesis
    gate1_pass = bool(single_audit["confident_toxic_as_edible_accepted"] == 0)
    gate2_pass = bool(single_audit["dangerous_pairs"]["failing_pairs_count"] == 0)
    gate3_pass = bool(single_audit["high_stakes_recall"]["failing_species_count"] == 0)
    gate4_pass = bool(single_audit["tflite_agreement_overall"] >= 0.99 and single_audit["tflite_agreement_high_stakes"] >= 0.99)
    gate5_pass = bool(ood_audit["passed"] and verna_audit["passed"])
    gate6_pass = bool(multi_audit["passed"])

    overall_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass and gate5_pass and gate6_pass

    summary_table = [
        {
            "gate_id": "GATE-01",
            "name": "confidentToxicAsEdible (Single Frame)",
            "target": "== 0",
            "measured": f"{single_audit['confident_toxic_as_edible_accepted']}",
            "status": "PASS" if gate1_pass else "FAIL",
        },
        {
            "gate_id": "GATE-02",
            "name": "DANGEROUS_PAIRS Confusion",
            "target": "<= 0.05 across all pairs",
            "measured": f"Failing pairs: {single_audit['dangerous_pairs']['failing_pairs_count']}/42",
            "status": "PASS" if gate2_pass else "FAIL",
        },
        {
            "gate_id": "GATE-03",
            "name": "highStakesRecallMin (n >= 8)",
            "target": ">= 0.85 (85%)",
            "measured": f"{single_audit['high_stakes_recall']['min_conditional_recall']*100:.2f}% (Failing: {single_audit['high_stakes_recall']['failing_species_count']})",
            "status": "PASS" if gate3_pass else "FAIL",
        },
        {
            "gate_id": "GATE-04",
            "name": "tfliteAgreement (FP16 vs PyTorch)",
            "target": ">= 0.99 (99%)",
            "measured": f"Overall: {single_audit['tflite_agreement_overall']*100:.2f}%, HS: {single_audit['tflite_agreement_high_stakes']*100:.2f}%",
            "status": "PASS" if gate4_pass else "FAIL",
        },
        {
            "gate_id": "GATE-05",
            "name": "ood_test Confident Edible",
            "target": "<= 0.02 (Wilson <= 0.05)",
            "measured": f"{ood_audit['confident_edible_rate']*100:.2f}% (Wilson 95% UB: {ood_audit['wilson_upper_bound_95']*100:.2f}%)",
            "status": "PASS" if gate5_pass else "FAIL",
        },
        {
            "gate_id": "GATE-06",
            "name": "Multi-Photo Pipeline Safety (P5)",
            "target": "Toxic As Edible == 0, Acc >= 85%",
            "measured": f"Toxic As Edible: {multi_audit['fused_toxic_as_edible_count']}, Acc: {multi_audit['conditional_accuracy']*100:.2f}%",
            "status": "PASS" if gate6_pass else "FAIL",
        },
    ]

    report = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "tflite_model": str(tflite_path),
        "tflite_sha256": compute_file_sha256(tflite_path),
        "overall_ship_gates_pass": overall_pass,
        "frozen_configuration": {
            "temperature_single": FROZEN_TEMPERATURE_SINGLE,
            "temperature_multi": FROZEN_TEMPERATURE_MULTI,
            "energy_threshold_theta": FROZEN_ENERGY_THRESHOLD,
            "min_top1_prob": FROZEN_MIN_TOP1_PROB,
            "disagreement_threshold": FROZEN_DISAGREEMENT_THRESHOLD,
            "high_stakes_veto_threshold": FROZEN_HIGH_STAKES_VETO_THRESHOLD,
        },
        "gates_summary_table": summary_table,
        "single_frame_audit": single_audit,
        "ood_test_audit": ood_audit,
        "uncovered_amanita_verna_audit": verna_audit,
        "multi_photo_audit": multi_audit,
    }

    report_json_path = out_dir / "ship_gates_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Render Markdown Summary
    md_content = render_ship_gates_markdown(report)
    summary_md_path = out_dir / "ship_gates_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved final Ship Gates report to %s and summary to %s", report_json_path, summary_md_path)
    return report


def render_ship_gates_markdown(report: dict[str, Any]) -> str:
    """Generate Markdown report for Ship Gates audit."""
    lines = [
        "# Finalny Audyt Ship Gates — Etap P6 / P8",
        "",
        f"**Data audytu**: `{report['audit_timestamp']}`",
        f"**Checkpoint PyTorch**: `{report['checkpoint']}`",
        f"**Model TFLite (FP16)**: `{report['tflite_model']}` (`{report['tflite_sha256'][:16]}...`)",
        f"**Decyzja końcowa**: **{'ZALICZONY (PASS)' if report['overall_ship_gates_pass'] else 'NIEZALICZONY (FAIL)'}**",
        "",
        "## 1. Tabela Główna Ship Gates",
        "",
        "| ID | Nazwa Bramki | Wymóg (Target) | Wynik Zmierzony | Status |",
        "|---|---|---|---|:---:|",
    ]
    for row in report["gates_summary_table"]:
        status_badge = "✅ **PASS**" if row["status"] == "PASS" else "❌ **FAIL**"
        lines.append(f"| `{row['gate_id']}` | **{row['name']}** | `{row['target']}` | {row['measured']} | {status_badge} |")

    single = report["single_frame_audit"]
    lines.extend([
        "",
        "## 2. Metryki Ogólne na Zbiorze Testowym (`data/splits/test.csv`)",
        "",
        f"- **Liczba próbek w teście**: {single['dataset_size']} (66 aktywnych klas)",
        f"- **Dokładność Top-1 (Raw PyTorch vs TFLite FP16)**: {single['accuracy_raw_pytorch']*100:.2f}% vs {single['accuracy_raw_tflite']*100:.2f}%",
        f"- **Zgodność TFLite FP16 vs PyTorch (`tfliteAgreement`)**: **{single['tflite_agreement_overall']*100:.2f}%** (Wysokie ryzyko: **{single['tflite_agreement_high_stakes']*100:.2f}%**)",
        f"- **Błąd logitów TFLite FP16**: MAE = `{single['tflite_logit_mae']:.4f}`, Max Error = `{single['tflite_max_logit_error']:.4f}`",
        f"- **Kalibracja ECE (15 binów, T=0.7848)**: {single['ece_tflite']*100:.2f}% (MCE = {single['mce_tflite']*100:.2f}%)",
        f"- **Keep Rate bramki OOD (θ = -2.2082, minTop1Prob = 0.50)**: **{single['keep_rate_tflite']*100:.2f}%**",
        f"- **Dokładność po odcięciu (Conditional Accuracy)**: **{single['accuracy_conditional_tflite']*100:.2f}%**",
        "",
        "## 3. Audyt Bezpieczeństwa — Gatunki Wysokiego Ryzyka i Pary Niebezpieczne",
        "",
        f"- **Confident Toxic as Edible**: `{single['confident_toxic_as_edible_accepted']}` (Wymóg: `== 0`)",
        f"- **Liczba par DANGEROUS_PAIRS przekraczających próg 5%**: `{single['dangerous_pairs']['failing_pairs_count']}` / 42",
        f"- **Minimalna czułość na klasach High Stakes (n >= 8)**: `{single['high_stakes_recall']['min_conditional_recall']*100:.2f}%`",
        "",
        "### Tabela Czułości Klas Śmiertelnych / Wysokiego Ryzyka (Wybrane)",
        "",
        "| Gatunek | n | TP (acc) | Czułość Warunkowa | Wilson 95% CI | Status |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for slug, s_data in report["single_frame_audit"]["high_stakes_recall"]["species"].items():
        if s_data["n"] >= 8:
            ci = s_data["wilson_ci_conditional"]
            status_str = "✅ PASS" if s_data["passed"] else "❌ FAIL"
            lines.append(
                f"| *{slug}* | {s_data['n']} | {s_data['accepted_tp']} | {s_data['conditional_recall']*100:.2f}% | [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%] | {status_str} |"
            )

    pairs_data = single["dangerous_pairs"]
    lines.extend([
        "",
        "### Tabela Par Niebezpiecznych (DANGEROUS_PAIRS)",
        "",
        "| Para Niebezpieczna (Gatunek A ↔ Gatunek B) | n_A (acc) | n_B (acc) | Pomyłka A→B | Pomyłka B→A | Max Pomyłka | Status (<= 5%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for p_name, p_info in sorted(pairs_data["pairs"].items(), key=lambda kv: kv[1]["max_confusion"], reverse=True):
        p_status = "❌ FAIL" if not p_info["passed"] else "✅ PASS"
        lines.append(
            f"| `{p_name}` | {p_info['n_a_accepted']}/{p_info['n_a']} | {p_info['n_b_accepted']}/{p_info['n_b']} | "
            f"{p_info['conditional_confusion_a_to_b']*100:.2f}% ({p_info['confusion_a_to_b_count']}) | "
            f"{p_info['conditional_confusion_b_to_a']*100:.2f}% ({p_info['confusion_b_to_a_count']}) | "
            f"**{p_info['max_confusion']*100:.2f}%** | {p_status} |"
        )

    ood = report["ood_test_audit"]
    verna = report["uncovered_amanita_verna_audit"]
    lines.extend([
        "",
        "## 4. Audyt OOD Test i Gatunków Niepokrytych",
        "",
        f"- **Próbki OOD test**: {ood['n_samples']} ({ood['n_fungi_test']} grzybów OOD, {ood['n_nonfungi_test']} nie-grzybów)",
        f"- **Odsetek odrzuconych przez energię**: {ood['energy_rejection_rate']*100:.2f}%",
        f"- **Odsetek Confident Edible na OOD test**: **{ood['confident_edible_rate']*100:.2f}%** (Wilson 95% UB: **{ood['wilson_upper_bound_95']*100:.2f}%**; Wymóg: rate <= 2%, UB <= 5%)",
        f"- **Amanita verna (11 próbek testowych)**: {verna['energy_rejected'] + verna['unclear_rejected']}/11 odrzuconych, Confident Edible: `{verna['confident_toxic_as_edible']}`",
        "",
        "## 5. Audyt Pipeline'u Wielokadrowego (P5)",
        "",
        f"- **Liczba obserwacji wielokadrowych**: {report['multi_photo_audit']['n_multi_observations']} ({report['multi_photo_audit']['n_multi_frames_total']} kadrów)",
        f"- **Czułość po fuzji (Conditional Accuracy)**: **{report['multi_photo_audit']['conditional_accuracy']*100:.2f}%**",
        f"- **Liczba konfliktów (`CONFLICT_DETECTED`)**: {report['multi_photo_audit']['conflict_detected_count']}",
        f"- **Liczba zadziałań veta bezpieczeństwa (`HIGH_STAKES_VETO`)**: {report['multi_photo_audit']['high_stakes_veto_count']}",
        f"- **Fused Confident Toxic as Edible**: **`{report['multi_photo_audit']['fused_toxic_as_edible_count']}`**",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P6/P8 Final Ship Gates Audit.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
    parser.add_argument("--tflite", type=str, default=str(DEFAULT_TFLITE), help="TFLite model path")
    parser.add_argument("--test-csv", type=str, default=str(DEFAULT_TEST_CSV), help="Test split CSV")
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR), help="Image directory")
    parser.add_argument("--ood-fungi-csv", type=str, default=str(DEFAULT_OOD_FUNGI_CSV), help="OOD fungi CSV")
    parser.add_argument("--ood-nonfungi-csv", type=str, default=str(DEFAULT_OOD_NONFUNGI_CSV), help="OOD nonfungi CSV")
    parser.add_argument("--karty-path", type=str, default=str(DEFAULT_KARTY_PATH), help="Path to karty_sobowtory.json")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    run_final_ship_gates_audit(
        checkpoint_path=args.checkpoint,
        tflite_path=args.tflite,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        ood_fungi_csv=args.ood_fungi_csv,
        ood_nonfungi_csv=args.ood_nonfungi_csv,
        karty_path=args.karty_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
