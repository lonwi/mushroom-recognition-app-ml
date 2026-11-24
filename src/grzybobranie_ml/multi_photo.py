"""Stage P5: Multi-photo prediction fusion with safety veto rules and OOD gating.

Module: grzybobranie_ml.multi_photo
CLI: python -m grzybobranie_ml.multi_photo [options]

Features:
- OOD filter per frame (reject frames with E(x) > theta_frame).
- Disagreement check (detect conflicts if >= 2 accepted frames have p >= 0.70 with different species).
- Logit averaging fusion: bar_z = mean(z_i) from accepted frames.
- Calibration with separate T_multi temperature (fit on val multi-photo observations).
- High-Stakes Safety Veto: blocks confident edible prediction if any frame has p_toxic >= 0.20 for high-stakes species.
- OOD threshold sweep on val.csv and ood_calib to produce trade-off table for tighter thresholds.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grzybobranie_ml.calibrate_ood import (
    compute_ece_mce,
    compute_energy_vector,
    compute_softmax,
    fit_temperature_scaling,
)
from grzybobranie_ml.train import (
    load_dangerous_pairs,
    load_high_stakes_species,
)

logger = logging.getLogger("grzybobranie_ml.multi_photo")

DEFAULT_CACHED_LOGITS = Path("artifacts/calibration/extracted_logits.pt")
DEFAULT_KARTY_PATH = Path("configs/karty_sobowtory.json")
DEFAULT_CALIBRATION_RESULTS = Path("artifacts/calibration/calibration_results.json")
DEFAULT_MULTI_PHOTO_OUTPUT_DIR = Path("artifacts/multi_photo")
DEFAULT_CALIBRATION_OUTPUT_DIR = Path("artifacts/calibration")


@dataclass
class FramePrediction:
    """Prediction summary for a single frame of an observation."""

    frame_idx: int
    energy: float
    is_ood: bool
    top1_idx: int
    top1_slug: str
    top1_prob: float
    high_stakes_probs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiPhotoResult:
    """Outcome of 4-step multi-photo fusion pipeline."""

    status: str  # OK, HIGH_STAKES_VETO, CONFLICT_DETECTED, ALL_FRAMES_OOD, OOD, UNCLEAR
    predicted_slug: str | None
    predicted_idx: int | None
    confidence: float
    energy: float
    is_edible: bool
    confident_edible: bool
    accepted_frame_indices: list[int]
    rejected_frame_indices: list[int]
    total_frames: int
    frame_predictions: list[FramePrediction]
    conflict_species: list[str] = field(default_factory=list)
    veto_risk_species: list[dict[str, Any]] = field(default_factory=list)
    fused_probabilities: list[float] | None = None
    fused_logits: list[float] | None = None

    @property
    def is_ood(self) -> bool:
        """Return True if observation was rejected by per-frame OOD or fused OOD gate."""
        return self.status in ("OOD", "ALL_FRAMES_OOD")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_ood"] = self.is_ood
        if self.fused_probabilities is not None:
            d["fused_probabilities"] = [float(p) for p in self.fused_probabilities]
        if self.fused_logits is not None:
            d["fused_logits"] = [float(z) for z in self.fused_logits]
        return d


def load_species_edibility_sets(karty_path: str | Path = DEFAULT_KARTY_PATH) -> tuple[set[str], set[str], list[str]]:
    """Load edible, toxic/deadly, and lethal species slugs from karty_sobowtory.json."""
    karty_p = Path(karty_path)
    edible_slugs: set[str] = set()
    toxic_or_deadly_slugs: set[str] = set()
    lethal_slugs: list[str] = []

    if not karty_p.exists():
        raise FileNotFoundError(f"Species cards configuration not found at: {karty_p}")

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

    return edible_slugs, toxic_or_deadly_slugs, lethal_slugs


def fuse_observation_frames(
    logits: np.ndarray | torch.Tensor | list[Any],
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    edible_slugs: set[str],
    high_stakes_slugs: set[str] | list[str],
    temperature_frame: float = 0.7848,
    temperature_multi: float = 0.5146,
    energy_threshold_frame: float = -1.8890,
    energy_threshold_fused: float | None = None,
    min_top1_prob: float = 0.55,
    disagreement_prob_threshold: float = 0.70,
    high_stakes_veto_threshold: float = 0.20,
    veto_check_all_frames: bool = True,
) -> MultiPhotoResult:
    """Execute 4-step multi-photo fusion pipeline on a batch of logits for one specimen.

    Pipeline specification:
    1. Per-frame OOD filter: reject frames with E(x_i) > energy_threshold_frame. If all rejected, return ALL_FRAMES_OOD.
    2. Disagreement check: if >= 2 accepted frames have p_top1 >= 0.70 for different species, return CONFLICT_DETECTED.
    3. Logit Averaging + Calibration: bar_z = mean(z_accepted), calibrate with temperature_multi, gate on E and min_top1_prob.
    4. High-Stakes Veto: if fused prediction is edible, check if any single frame (default: all frames) had
       p_toxic >= 0.20 for any high-stakes species. If so, return HIGH_STAKES_VETO and block confident edible.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    elif not isinstance(logits, np.ndarray):
        logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    elif logits.ndim != 2:
        raise ValueError(f"Logits must be 1D or 2D array, got shape {logits.shape}")

    total_frames = logits.shape[0]
    if total_frames == 0:
        raise ValueError("Cannot fuse empty logits: observation must contain at least 1 frame.")

    if np.isnan(logits).any() or np.isinf(logits).any():
        raise ValueError("Logits contain NaN or Inf values.")

    if temperature_frame <= 0 or temperature_multi <= 0:
        raise ValueError("Temperatures must be strictly positive.")

    high_stakes_set = set(high_stakes_slugs)
    high_stakes_indices = {class_to_idx[s]: s for s in high_stakes_set if s in class_to_idx}

    if energy_threshold_fused is None:
        energy_threshold_fused = energy_threshold_frame

    # Compute per-frame energy and probabilities using single-frame temperature
    frame_energies = compute_energy_vector(logits, temperature_frame)
    frame_probs = compute_softmax(logits, temperature_frame)

    frame_predictions: list[FramePrediction] = []
    accepted_frames: list[int] = []
    rejected_frames: list[int] = []

    for i in range(total_frames):
        e_i = float(frame_energies[i])
        is_ood = bool(e_i > energy_threshold_frame)
        top1_idx = int(np.argmax(frame_probs[i]))
        top1_prob = float(frame_probs[i, top1_idx])
        top1_slug = idx_to_class.get(top1_idx, f"unknown_{top1_idx}")

        hs_probs = {
            s: float(frame_probs[i, c_idx])
            for c_idx, s in high_stakes_indices.items()
            if frame_probs[i, c_idx] >= 0.05
        }

        frame_predictions.append(
            FramePrediction(
                frame_idx=i,
                energy=e_i,
                is_ood=is_ood,
                top1_idx=top1_idx,
                top1_slug=top1_slug,
                top1_prob=top1_prob,
                high_stakes_probs=hs_probs,
            )
        )

        if is_ood:
            rejected_frames.append(i)
        else:
            accepted_frames.append(i)

    # Step 1: Per-frame OOD filter check
    if len(accepted_frames) == 0:
        return MultiPhotoResult(
            status="ALL_FRAMES_OOD",
            predicted_slug=None,
            predicted_idx=None,
            confidence=0.0,
            energy=float(np.mean(frame_energies)),
            is_edible=False,
            confident_edible=False,
            accepted_frame_indices=[],
            rejected_frame_indices=rejected_frames,
            total_frames=total_frames,
            frame_predictions=frame_predictions,
        )

    # Step 2: Disagreement Check
    confident_calls: list[tuple[int, int, str, float]] = []
    for i in accepted_frames:
        top1_idx = frame_predictions[i].top1_idx
        top1_prob = frame_predictions[i].top1_prob
        top1_slug = frame_predictions[i].top1_slug
        if top1_prob >= disagreement_prob_threshold:
            confident_calls.append((i, top1_idx, top1_slug, top1_prob))

    if len(confident_calls) >= 2:
        distinct_classes = sorted({c[2] for c in confident_calls})
        if len(distinct_classes) > 1:
            return MultiPhotoResult(
                status="CONFLICT_DETECTED",
                predicted_slug=None,
                predicted_idx=None,
                confidence=0.0,
                energy=float(np.mean([frame_energies[i] for i in accepted_frames])),
                is_edible=False,
                confident_edible=False,
                accepted_frame_indices=accepted_frames,
                rejected_frame_indices=rejected_frames,
                total_frames=total_frames,
                frame_predictions=frame_predictions,
                conflict_species=distinct_classes,
            )

    # Step 3: Logit Averaging + Calibration
    accepted_logits = logits[accepted_frames]
    bar_z = np.mean(accepted_logits, axis=0, keepdims=True)

    # Fused energy (using frame temperature or single scale)
    fused_energy = float(compute_energy_vector(bar_z, temperature_frame)[0])

    # Calibrated probabilities with T_multi (or temperature_frame if single photo)
    eff_temp = temperature_multi if len(accepted_frames) > 1 else temperature_frame
    fused_probs = compute_softmax(bar_z, eff_temp)[0]

    fused_top1_idx = int(np.argmax(fused_probs))
    fused_confidence = float(fused_probs[fused_top1_idx])
    fused_slug = idx_to_class.get(fused_top1_idx, f"unknown_{fused_top1_idx}")
    is_edible = fused_slug in edible_slugs

    # Gating on fused prediction
    if fused_energy > energy_threshold_fused:
        return MultiPhotoResult(
            status="OOD",
            predicted_slug=fused_slug,
            predicted_idx=fused_top1_idx,
            confidence=fused_confidence,
            energy=fused_energy,
            is_edible=is_edible,
            confident_edible=False,
            accepted_frame_indices=accepted_frames,
            rejected_frame_indices=rejected_frames,
            total_frames=total_frames,
            frame_predictions=frame_predictions,
            fused_probabilities=fused_probs.tolist(),
            fused_logits=bar_z[0].tolist(),
        )

    if fused_confidence < min_top1_prob:
        return MultiPhotoResult(
            status="UNCLEAR",
            predicted_slug=fused_slug,
            predicted_idx=fused_top1_idx,
            confidence=fused_confidence,
            energy=fused_energy,
            is_edible=is_edible,
            confident_edible=False,
            accepted_frame_indices=accepted_frames,
            rejected_frame_indices=rejected_frames,
            total_frames=total_frames,
            frame_predictions=frame_predictions,
            fused_probabilities=fused_probs.tolist(),
            fused_logits=bar_z[0].tolist(),
        )

    # Step 4: High-Stakes Safety Veto
    # If top-1 is edible, check whether on any single frame any high-stakes species >= veto threshold
    veto_risks: list[dict[str, Any]] = []
    frames_to_check = range(total_frames) if veto_check_all_frames else accepted_frames

    if is_edible:
        for f_idx in frames_to_check:
            for hs_idx, hs_slug in high_stakes_indices.items():
                p_toxic = float(frame_probs[f_idx, hs_idx])
                if p_toxic >= high_stakes_veto_threshold:
                    veto_risks.append({
                        "slug": hs_slug,
                        "class_idx": hs_idx,
                        "probability": p_toxic,
                        "frame_idx": f_idx,
                    })

    if len(veto_risks) > 0:
        return MultiPhotoResult(
            status="HIGH_STAKES_VETO",
            predicted_slug=fused_slug,
            predicted_idx=fused_top1_idx,
            confidence=fused_confidence,
            energy=fused_energy,
            is_edible=is_edible,
            confident_edible=False,  # BLOCKED by veto
            accepted_frame_indices=accepted_frames,
            rejected_frame_indices=rejected_frames,
            total_frames=total_frames,
            frame_predictions=frame_predictions,
            veto_risk_species=veto_risks,
            fused_probabilities=fused_probs.tolist(),
            fused_logits=bar_z[0].tolist(),
        )

    # All checks passed: clean OK result
    return MultiPhotoResult(
        status="OK",
        predicted_slug=fused_slug,
        predicted_idx=fused_top1_idx,
        confidence=fused_confidence,
        energy=fused_energy,
        is_edible=is_edible,
        confident_edible=is_edible,
        accepted_frame_indices=accepted_frames,
        rejected_frame_indices=rejected_frames,
        total_frames=total_frames,
        frame_predictions=frame_predictions,
        fused_probabilities=fused_probs.tolist(),
        fused_logits=bar_z[0].tolist(),
    )


def run_ood_threshold_sweep(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    ood_fungi_logits: np.ndarray,
    ood_fungi_records: list[dict[str, Any]],
    ood_nonfungi_logits: np.ndarray,
    ood_nonfungi_records: list[dict[str, Any]],
    idx_to_class: dict[int, str],
    class_to_idx: dict[str, int],
    edible_slugs: set[str],
    toxic_slugs: set[str],
    temperature: float = 0.7848,
    target_id_tprs: list[float] | None = None,
    min_top1_probs: list[float] | None = None,
) -> dict[str, Any]:
    """Execute grid sweep over energy threshold theta and minTop1Prob strictly on val.csv and ood_calib.

    STRICT SAFETY: Does NOT touch ood_test or test.csv!
    """
    if target_id_tprs is None:
        target_id_tprs = [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]
    if min_top1_probs is None:
        min_top1_probs = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]

    val_e = compute_energy_vector(val_logits, temperature)
    val_probs = compute_softmax(val_logits, temperature)
    val_p = np.max(val_probs, axis=1)
    val_preds = np.argmax(val_logits, axis=1)

    # Filter calib partition ONLY from OOD records
    f_calib = np.array([r.get("proposed_split") == "calib" for r in ood_fungi_records])
    nf_calib = np.array([r.get("proposed_split") == "calib" for r in ood_nonfungi_records])

    f_calib_logits = ood_fungi_logits[f_calib]
    nf_calib_logits = ood_nonfungi_logits[nf_calib]

    f_e = compute_energy_vector(f_calib_logits, temperature)
    f_probs = compute_softmax(f_calib_logits, temperature)
    f_p = np.max(f_probs, axis=1)
    f_preds = np.argmax(f_calib_logits, axis=1)
    f_is_edible = np.array([idx_to_class.get(p, "") in edible_slugs for p in f_preds])

    nf_e = compute_energy_vector(nf_calib_logits, temperature)
    nf_probs = compute_softmax(nf_calib_logits, temperature)
    nf_p = np.max(nf_probs, axis=1)
    nf_preds = np.argmax(nf_calib_logits, axis=1)
    nf_is_edible = np.array([idx_to_class.get(p, "") in edible_slugs for p in nf_preds])

    n_nf = len(nf_e)
    n_f = len(f_e)
    n_calib_total = n_nf + n_f

    # Baseline operating point (P4: 95% TPR, min_p=0.40)
    th_baseline = float(np.percentile(val_e, 95.0, method="higher"))
    base_val_ok = (val_e <= th_baseline) & (val_p >= 0.40)
    base_val_keep_rate = float(np.mean(base_val_ok))

    base_nf_ce_cnt = int(np.sum((nf_e <= th_baseline) & (nf_p >= 0.40) & nf_is_edible))
    base_nf_ce_rate = float(base_nf_ce_cnt / n_nf)

    base_f_ce_cnt = int(np.sum((f_e <= th_baseline) & (f_p >= 0.40) & f_is_edible))
    base_f_ce_rate = float(base_f_ce_cnt / n_f)

    base_tot_ce_cnt = base_nf_ce_cnt + base_f_ce_cnt
    base_tot_ce_rate = float(base_tot_ce_cnt / n_calib_total)

    grid_results: list[dict[str, Any]] = []

    toxic_indices = {class_to_idx[s] for s in toxic_slugs if s in class_to_idx}
    edible_indices = {class_to_idx[s] for s in edible_slugs if s in class_to_idx}
    val_is_toxic = np.isin(val_targets, list(toxic_indices))
    val_pred_is_edible = np.isin(val_preds, list(edible_indices))

    for tpr in target_id_tprs:
        th = float(np.percentile(val_e, tpr * 100.0, method="higher"))
        for min_p in min_top1_probs:
            val_energy_ok = val_e <= th
            val_unclear = val_energy_ok & (val_p < min_p)
            val_ok = val_energy_ok & (~val_unclear)

            val_keep = float(np.mean(val_ok))
            val_energy_rej = float(np.mean(~val_energy_ok))
            val_unclear_rej = float(np.mean(val_unclear))

            val_acc = float(np.mean(val_preds[val_ok] == val_targets[val_ok])) if np.sum(val_ok) > 0 else 0.0
            val_conf_toxic_as_ed = int(np.sum(val_ok & val_is_toxic & val_pred_is_edible))

            # Nonfungi calib
            nf_energy_ok = nf_e <= th
            nf_ok = nf_energy_ok & (nf_p >= min_p)
            nf_ce_cnt = int(np.sum(nf_ok & nf_is_edible))
            nf_ce_rate = float(nf_ce_cnt / n_nf)
            nf_ce_rel_drop = float((base_nf_ce_rate - nf_ce_rate) / base_nf_ce_rate) if base_nf_ce_rate > 0 else 0.0

            # Fungi calib
            f_energy_ok = f_e <= th
            f_ok = f_energy_ok & (f_p >= min_p)
            f_ce_cnt = int(np.sum(f_ok & f_is_edible))
            f_ce_rate = float(f_ce_cnt / n_f)
            f_ce_rel_drop = float((base_f_ce_rate - f_ce_rate) / base_f_ce_rate) if base_f_ce_rate > 0 else 0.0

            # Combined calib
            tot_ce_cnt = nf_ce_cnt + f_ce_cnt
            tot_ce_rate = float(tot_ce_cnt / n_calib_total)
            tot_ce_rel_drop = float((base_tot_ce_rate - tot_ce_rate) / base_tot_ce_rate) if base_tot_ce_rate > 0 else 0.0

            tot_accepted = int(np.sum(nf_ok)) + int(np.sum(f_ok))
            tot_accept_rate = float(tot_accepted / n_calib_total)

            grid_results.append({
                "target_id_tpr": tpr,
                "energy_threshold": th,
                "min_top1_prob": min_p,
                "val": {
                    "keep_rate": val_keep,
                    "energy_rejection_rate": val_energy_rej,
                    "unclear_rejection_rate": val_unclear_rej,
                    "conditional_accuracy": val_acc,
                    "confident_toxic_as_edible_count": val_conf_toxic_as_ed,
                },
                "ood_calib_nonfungi": {
                    "n_samples": n_nf,
                    "confident_edible_count": nf_ce_cnt,
                    "confident_edible_rate": nf_ce_rate,
                    "relative_drop_from_baseline": nf_ce_rel_drop,
                },
                "ood_calib_fungi": {
                    "n_samples": n_f,
                    "confident_edible_count": f_ce_cnt,
                    "confident_edible_rate": f_ce_rate,
                    "relative_drop_from_baseline": f_ce_rel_drop,
                },
                "ood_calib_total": {
                    "n_samples": n_calib_total,
                    "confident_edible_count": tot_ce_cnt,
                    "confident_edible_rate": tot_ce_rate,
                    "relative_drop_from_baseline": tot_ce_rel_drop,
                    "acceptance_rate": tot_accept_rate,
                },
            })

    # Pick recommended operating point (e.g., target val_keep in 80-85% with balanced drop)
    candidates = [
        r for r in grid_results
        if 0.80 <= r["val"]["keep_rate"] <= 0.85 and r["min_top1_prob"] >= 0.50
    ]
    # Sort candidates by lowest total confident edible rate on ood_calib
    candidates.sort(key=lambda x: x["ood_calib_total"]["confident_edible_rate"])
    recommended = candidates[0] if candidates else grid_results[0]

    return {
        "baseline": {
            "energy_threshold": th_baseline,
            "min_top1_prob": 0.40,
            "val_keep_rate": base_val_keep_rate,
            "ood_calib_nonfungi_confident_edible_rate": base_nf_ce_rate,
            "ood_calib_fungi_confident_edible_rate": base_f_ce_rate,
            "ood_calib_total_confident_edible_rate": base_tot_ce_rate,
        },
        "recommended": recommended,
        "grid": grid_results,
    }


def evaluate_val_multi_photo(
    cached_logits_path: str | Path = DEFAULT_CACHED_LOGITS,
    karty_path: str | Path = DEFAULT_KARTY_PATH,
    output_dir: str | Path = DEFAULT_MULTI_PHOTO_OUTPUT_DIR,
    energy_threshold_frame: float = -1.8890,
    min_top1_prob: float = 0.55,
    disagreement_prob_threshold: float = 0.70,
    high_stakes_veto_threshold: float = 0.20,
    veto_check_all_frames: bool = True,
) -> dict[str, Any]:
    """Evaluate multi-photo fusion pipeline on val.csv observation groups."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(cached_logits_path, map_location="cpu", weights_only=False)
    val_records: list[dict[str, Any]] = cache["val_records"]
    val_logits = cache["val_logits"].double().numpy()
    val_targets = np.array(cache["val_targets"], dtype=np.int64)
    idx_to_class: dict[int, str] = cache["idx_to_class_66"]
    class_to_idx: dict[str, int] = cache["class_to_idx_66"]

    edible_slugs, toxic_slugs, _lethal_slugs = load_species_edibility_sets(karty_path)
    high_stakes_slugs = sorted([s for s in load_high_stakes_species(karty_path) if s in class_to_idx])

    # 1. Group val records by observation
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

    multi_obs = {oid: idxs for oid, idxs in obs_map.items() if len(idxs) > 1}
    single_obs = {oid: idxs for oid, idxs in obs_map.items() if len(idxs) == 1}

    logger.info(
        "Loaded %d val records: %d observations (%d single-photo, %d multi-photo with %d total frames).",
        len(val_records),
        len(obs_map),
        len(single_obs),
        len(multi_obs),
        sum(len(idxs) for idxs in multi_obs.values()),
    )

    t_single = 0.78479995600372

    # 2. Fit T_multi specifically on fused multi-photo observations
    multi_fused_logits_raw: list[np.ndarray] = []
    multi_targets: list[int] = []
    for oid, idxs in multi_obs.items():
        multi_fused_logits_raw.append(np.mean(val_logits[idxs], axis=0))
        multi_targets.append(val_targets[idxs[0]])

    multi_fused_logits_arr = np.array(multi_fused_logits_raw)
    multi_targets_arr = np.array(multi_targets, dtype=np.int64)

    t_multi_opt = fit_temperature_scaling(multi_fused_logits_arr, multi_targets_arr, initial_t=1.0)
    logger.info("Fitted T_multi on multi-photo observations: %.4f (vs single T = %.4f)", t_multi_opt, t_single)

    # Calibration comparison on multi-photo subset
    multi_single_frame_indices = [idx for idxs in multi_obs.values() for idx in idxs]
    ece_multi_single, mce_multi_single, _ = compute_ece_mce(
        val_logits[multi_single_frame_indices],
        val_targets[multi_single_frame_indices],
        temperature=t_single,
        n_bins=15,
    )
    ece_multi_fused_tsingle, mce_multi_fused_tsingle, _ = compute_ece_mce(
        multi_fused_logits_arr,
        multi_targets_arr,
        temperature=t_single,
        n_bins=15,
    )
    ece_multi_fused_tmulti, mce_multi_fused_tmulti, _ = compute_ece_mce(
        multi_fused_logits_arr,
        multi_targets_arr,
        temperature=t_multi_opt,
        n_bins=15,
    )

    # 3. Run multi-photo fusion pipeline on all multi-photo observations
    multi_results: dict[str, MultiPhotoResult] = {}
    status_counts: dict[str, int] = defaultdict(int)

    for oid, idxs in multi_obs.items():
        res = fuse_observation_frames(
            logits=val_logits[idxs],
            idx_to_class=idx_to_class,
            class_to_idx=class_to_idx,
            edible_slugs=edible_slugs,
            high_stakes_slugs=high_stakes_slugs,
            temperature_frame=t_single,
            temperature_multi=t_multi_opt,
            energy_threshold_frame=energy_threshold_frame,
            energy_threshold_fused=energy_threshold_frame,
            min_top1_prob=min_top1_prob,
            disagreement_prob_threshold=disagreement_prob_threshold,
            high_stakes_veto_threshold=high_stakes_veto_threshold,
            veto_check_all_frames=veto_check_all_frames,
        )
        multi_results[oid] = res
        status_counts[res.status] += 1

    # Also run on all observations (single + multi) to evaluate system-wide observation level metrics
    all_obs_results: dict[str, MultiPhotoResult] = {}
    for oid, idxs in obs_map.items():
        res = fuse_observation_frames(
            logits=val_logits[idxs],
            idx_to_class=idx_to_class,
            class_to_idx=class_to_idx,
            edible_slugs=edible_slugs,
            high_stakes_slugs=high_stakes_slugs,
            temperature_frame=t_single,
            temperature_multi=t_multi_opt,
            energy_threshold_frame=energy_threshold_frame,
            energy_threshold_fused=energy_threshold_frame,
            min_top1_prob=min_top1_prob,
            disagreement_prob_threshold=disagreement_prob_threshold,
            high_stakes_veto_threshold=high_stakes_veto_threshold,
            veto_check_all_frames=veto_check_all_frames,
        )
        all_obs_results[oid] = res

    # 4. Evaluation of 8 Problematic Pairs
    problematic_8 = [
        ("macrolepiota_procera", "lepiota_brunneoincarnata"),
        ("agaricus_campestris", "agaricus_xanthodermus"),
        ("agaricus_sylvaticus", "agaricus_moelleri"),
        ("kuehneromyces_mutabilis", "hypholoma_fasciculare"),
        ("lactarius_deliciosus", "lactarius_torminosus"),
        ("russula_vesca", "russula_emetica"),
        ("hypholoma_capnoides", "hypholoma_fasciculare"),
        ("neoboletus_luridiformis", "suillellus_luridus"),
    ]

    all_pairs = load_dangerous_pairs(karty_path)
    active_pairs = [(a, b) for a, b in all_pairs if a in class_to_idx and b in class_to_idx]

    pairs_evaluation: dict[str, Any] = {}
    for a, b in active_pairs:
        ia, ib = class_to_idx[a], class_to_idx[b]
        is_prob_8 = (a, b) in problematic_8 or (b, a) in problematic_8

        # Frame-level (single frames)
        mask_a_frame = val_targets == ia
        mask_b_frame = val_targets == ib
        na_frame, nb_frame = int(np.sum(mask_a_frame)), int(np.sum(mask_b_frame))
        raw_preds = np.argmax(val_logits, axis=1)

        c_ab_frame = int(np.sum(raw_preds[mask_a_frame] == ib))
        c_ba_frame = int(np.sum(raw_preds[mask_b_frame] == ia))
        err_ab_frame = float(c_ab_frame / na_frame) if na_frame > 0 else 0.0
        err_ba_frame = float(c_ba_frame / nb_frame) if nb_frame > 0 else 0.0

        # Multi-photo subset only
        multi_a_oids = [oid for oid, idxs in multi_obs.items() if val_targets[idxs[0]] == ia]
        multi_b_oids = [oid for oid, idxs in multi_obs.items() if val_targets[idxs[0]] == ib]
        multi_a_frames = [idx for oid in multi_a_oids for idx in multi_obs[oid]]
        multi_b_frames = [idx for oid in multi_b_oids for idx in multi_obs[oid]]

        c_ab_multi_single = int(np.sum(raw_preds[multi_a_frames] == ib)) if multi_a_frames else 0
        c_ba_multi_single = int(np.sum(raw_preds[multi_b_frames] == ia)) if multi_b_frames else 0

        # Multi-photo fused
        fused_a_preds = [multi_results[oid].predicted_idx for oid in multi_a_oids]
        fused_b_preds = [multi_results[oid].predicted_idx for oid in multi_b_oids]
        c_ab_multi_fused = sum(1 for p in fused_a_preds if p == ib)
        c_ba_multi_fused = sum(1 for p in fused_b_preds if p == ia)

        # Observation level with P5 pipeline (all observations)
        obs_a_list = [res for oid, res in all_obs_results.items() if val_targets[obs_map[oid][0]] == ia]
        obs_b_list = [res for oid, res in all_obs_results.items() if val_targets[obs_map[oid][0]] == ib]

        ok_obs_a = [r for r in obs_a_list if r.status == "OK"]
        ok_obs_b = [r for r in obs_b_list if r.status == "OK"]

        c_ab_obs_ok = sum(1 for r in ok_obs_a if r.predicted_idx == ib)
        c_ba_obs_ok = sum(1 for r in ok_obs_b if r.predicted_idx == ia)

        err_ab_obs = float(c_ab_obs_ok / len(ok_obs_a)) if len(ok_obs_a) > 0 else 0.0
        err_ba_obs = float(c_ba_obs_ok / len(ok_obs_b)) if len(ok_obs_b) > 0 else 0.0

        vetoes_a = sum(1 for r in obs_a_list if r.status == "HIGH_STAKES_VETO")
        vetoes_b = sum(1 for r in obs_b_list if r.status == "HIGH_STAKES_VETO")
        conflicts_a = sum(1 for r in obs_a_list if r.status == "CONFLICT_DETECTED")
        conflicts_b = sum(1 for r in obs_b_list if r.status == "CONFLICT_DETECTED")

        pair_key = f"{a} <-> {b}"
        pairs_evaluation[pair_key] = {
            "is_problematic_8": is_prob_8,
            "single_frame": {
                "n_a": na_frame,
                "n_b": nb_frame,
                "c_ab": c_ab_frame,
                "c_ba": c_ba_frame,
                "err_ab": err_ab_frame,
                "err_ba": err_ba_frame,
                "max_err": max(err_ab_frame, err_ba_frame),
            },
            "multi_photo_subset": {
                "n_obs_a": len(multi_a_oids),
                "n_obs_b": len(multi_b_oids),
                "n_frames_a": len(multi_a_frames),
                "n_frames_b": len(multi_b_frames),
                "c_ab_single_frames": c_ab_multi_single,
                "c_ba_single_frames": c_ba_multi_single,
                "c_ab_fused": c_ab_multi_fused,
                "c_ba_fused": c_ba_multi_fused,
                "err_ab_single_frames": float(c_ab_multi_single / len(multi_a_frames)) if len(multi_a_frames) > 0 else 0.0,
                "err_ba_single_frames": float(c_ba_multi_single / len(multi_b_frames)) if len(multi_b_frames) > 0 else 0.0,
                "err_ab_fused": float(c_ab_multi_fused / len(multi_a_oids)) if len(multi_a_oids) > 0 else 0.0,
                "err_ba_fused": float(c_ba_multi_fused / len(multi_b_oids)) if len(multi_b_oids) > 0 else 0.0,
                "max_err_single_frames": max(
                    float(c_ab_multi_single / len(multi_a_frames)) if len(multi_a_frames) > 0 else 0.0,
                    float(c_ba_multi_single / len(multi_b_frames)) if len(multi_b_frames) > 0 else 0.0,
                ),
                "max_err_fused": max(
                    float(c_ab_multi_fused / len(multi_a_oids)) if len(multi_a_oids) > 0 else 0.0,
                    float(c_ba_multi_fused / len(multi_b_oids)) if len(multi_b_oids) > 0 else 0.0,
                ),
            },
            "all_observations_pipeline": {
                "total_obs_a": len(obs_a_list),
                "total_obs_b": len(obs_b_list),
                "accepted_ok_a": len(ok_obs_a),
                "accepted_ok_b": len(ok_obs_b),
                "c_ab_ok": c_ab_obs_ok,
                "c_ba_ok": c_ba_obs_ok,
                "err_ab": err_ab_obs,
                "err_ba": err_ba_obs,
                "max_err": max(err_ab_obs, err_ba_obs),
                "vetoes_a": vetoes_a,
                "vetoes_b": vetoes_b,
                "conflicts_a": conflicts_a,
                "conflicts_b": conflicts_b,
            },
        }

    # 5. High-Stakes Species Recall
    hs_recall_eval: dict[str, Any] = {}
    for s in high_stakes_slugs:
        idx = class_to_idx[s]

        # Single frame recall
        mask = val_targets == idx
        n_frames = int(np.sum(mask))
        correct_frames = int(np.sum(raw_preds[mask] == idx))
        rec_frame = float(correct_frames / n_frames) if n_frames > 0 else 0.0

        # Multi-photo subset
        s_multi_oids = [oid for oid, idxs in multi_obs.items() if val_targets[idxs[0]] == idx]
        s_multi_frames = [i for oid in s_multi_oids for i in multi_obs[oid]]
        n_mf = len(s_multi_frames)
        c_mf_single = int(np.sum(raw_preds[s_multi_frames] == idx)) if n_mf > 0 else 0
        rec_mf_single = float(c_mf_single / n_mf) if n_mf > 0 else 0.0

        fused_s_preds = [multi_results[oid].predicted_idx for oid in s_multi_oids]
        c_mf_fused = sum(1 for p in fused_s_preds if p == idx)
        rec_mf_fused = float(c_mf_fused / len(s_multi_oids)) if len(s_multi_oids) > 0 else 0.0

        # All observations pipeline (accepted OK conditional recall)
        s_obs_list = [res for oid, res in all_obs_results.items() if val_targets[obs_map[oid][0]] == idx]
        s_obs_ok = [r for r in s_obs_list if r.status == "OK"]
        c_obs_ok = sum(1 for r in s_obs_ok if r.predicted_idx == idx)
        rec_obs_conditional = float(c_obs_ok / len(s_obs_ok)) if len(s_obs_ok) > 0 else 0.0
        rec_obs_effective = float(c_obs_ok / len(s_obs_list)) if len(s_obs_list) > 0 else 0.0

        hs_recall_eval[s] = {
            "single_frame": {
                "n_samples": n_frames,
                "correct": correct_frames,
                "recall": rec_frame,
            },
            "multi_photo_subset": {
                "n_observations": len(s_multi_oids),
                "n_frames": n_mf,
                "single_frame_recall": rec_mf_single,
                "fused_recall": rec_mf_fused,
            },
            "all_observations_pipeline": {
                "total_observations": len(s_obs_list),
                "accepted_ok": len(s_obs_ok),
                "correct_ok": c_obs_ok,
                "conditional_recall": rec_obs_conditional,
                "effective_recall": rec_obs_effective,
            },
        }

    # Summary metrics across high-stakes species and overall safety
    hs_cond_recalls = [v["all_observations_pipeline"]["conditional_recall"] for v in hs_recall_eval.values()]
    min_hs_cond_recall = float(min(hs_cond_recalls)) if hs_cond_recalls else 0.0
    mean_hs_cond_recall = float(np.mean(hs_cond_recalls)) if hs_cond_recalls else 0.0

    hs_fused_recalls = [
        v["multi_photo_subset"]["fused_recall"]
        for v in hs_recall_eval.values()
        if v["multi_photo_subset"]["n_observations"] > 0
    ]
    min_hs_fused_recall = float(min(hs_fused_recalls)) if hs_fused_recalls else 0.0
    mean_hs_fused_recall = float(np.mean(hs_fused_recalls)) if hs_fused_recalls else 0.0

    conf_toxic_obs_cnt = sum(
        1 for oid, res in all_obs_results.items()
        if idx_to_class[val_targets[obs_map[oid][0]]] in toxic_slugs and res.confident_edible
    )

    results = {
        "parameters": {
            "temperature_single": float(t_single),
            "temperature_multi": float(t_multi_opt),
            "energy_threshold_frame": float(energy_threshold_frame),
            "min_top1_prob": float(min_top1_prob),
            "disagreement_prob_threshold": float(disagreement_prob_threshold),
            "high_stakes_veto_threshold": float(high_stakes_veto_threshold),
            "veto_check_all_frames": bool(veto_check_all_frames),
        },
        "dataset_summary": {
            "total_val_frames": len(val_records),
            "total_observations": len(obs_map),
            "single_photo_observations": len(single_obs),
            "multi_photo_observations": len(multi_obs),
            "multi_photo_frames": sum(len(idxs) for idxs in multi_obs.values()),
        },
        "summary": {
            "high_stakes_recall_min_conditional": float(min_hs_cond_recall),
            "high_stakes_recall_mean_conditional": float(mean_hs_cond_recall),
            "high_stakes_recall_min_fused": float(min_hs_fused_recall),
            "high_stakes_recall_mean_fused": float(mean_hs_fused_recall),
            "confident_toxic_as_edible_observations_count": int(conf_toxic_obs_cnt),
            "confident_toxic_as_edible_observations_rate": float(conf_toxic_obs_cnt / len(all_obs_results)),
        },
        "multi_photo_subset": {
            "n_observations": len(multi_obs),
            "status_distribution": {k: int(v) for k, v in status_counts.items()},
            "status_percentages": {k: float(v / len(multi_obs)) for k, v in status_counts.items()},
            "calibration": {
                "ece_single_frames": float(ece_multi_single),
                "mce_single_frames": float(mce_multi_single),
                "ece_fused_tsingle": float(ece_multi_fused_tsingle),
                "mce_fused_tsingle": float(mce_multi_fused_tsingle),
                "ece_fused_tmulti": float(ece_multi_fused_tmulti),
                "mce_fused_tmulti": float(mce_multi_fused_tmulti),
            },
            "accuracy": {
                "single_frames": float(np.mean(raw_preds[multi_single_frame_indices] == val_targets[multi_single_frame_indices])),
                "fused_raw": float(np.mean(np.argmax(multi_fused_logits_arr, axis=1) == multi_targets_arr)),
            },
        },
        "problematic_8_pairs": {k: v for k, v in pairs_evaluation.items() if v["is_problematic_8"]},
        "dangerous_pairs_all": pairs_evaluation,
        "high_stakes_recall": hs_recall_eval,
    }

    out_json = out_dir / "multi_photo_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved multi-photo evaluation results to %s", out_json)

    return results


def main() -> None:
    """CLI entry point for grzybobranie_ml.multi_photo."""
    parser = argparse.ArgumentParser(description="Multi-photo prediction fusion and OOD threshold sweep.")
    parser.add_argument("--sweep", action="store_true", help="Run 2D OOD threshold sweep on val.csv and ood_calib.")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation of multi-photo fusion on val.csv.")
    parser.add_argument("--cached-logits", type=str, default=str(DEFAULT_CACHED_LOGITS), help="Path to cached logits .pt file.")
    parser.add_argument("--karty-path", type=str, default=str(DEFAULT_KARTY_PATH), help="Path to karty_sobowtory.json.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_MULTI_PHOTO_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--energy-threshold", type=float, default=-1.8890, help="Energy threshold theta.")
    parser.add_argument("--min-top1-prob", type=float, default=0.55, help="Minimum top1 confidence.")
    parser.add_argument("--disagreement-threshold", type=float, default=0.70, help="Disagreement confidence threshold.")
    parser.add_argument("--veto-threshold", type=float, default=0.20, help="High-stakes veto probability threshold.")
    parser.add_argument(
        "--no-veto-all-frames",
        action="store_true",
        help="Only check accepted frames for high stakes veto (default checks all frames for fail-closed safety).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # If neither flag specified, run both sweep and eval
    run_sweep = args.sweep or (not args.sweep and not args.eval)
    run_eval = args.eval or (not args.sweep and not args.eval)

    cached_p = Path(args.cached_logits)
    if not cached_p.exists():
        raise FileNotFoundError(f"Cached logits not found at {cached_p}. Run calibrate_ood first.")

    cache = torch.load(cached_p, map_location="cpu", weights_only=False)

    if run_sweep:
        logger.info("Executing 2D OOD threshold sweep on val.csv and ood_calib...")
        edible_slugs, toxic_slugs, _ = load_species_edibility_sets(args.karty_path)
        sweep_res = run_ood_threshold_sweep(
            val_logits=cache["val_logits"].double().numpy(),
            val_targets=np.array(cache["val_targets"], dtype=np.int64),
            ood_fungi_logits=cache["ood_fungi_logits"].double().numpy(),
            ood_fungi_records=cache["ood_fungi_records"],
            ood_nonfungi_logits=cache["ood_nonfungi_logits"].double().numpy(),
            ood_nonfungi_records=cache["ood_nonfungi_records"],
            idx_to_class=cache["idx_to_class_66"],
            class_to_idx=cache["class_to_idx_66"],
            edible_slugs=edible_slugs,
            toxic_slugs=toxic_slugs,
            temperature=0.78479995600372,
        )
        sweep_out = DEFAULT_CALIBRATION_OUTPUT_DIR / "ood_sweep_results.json"
        sweep_out.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_out, "w", encoding="utf-8") as f:
            json.dump(sweep_res, f, indent=2)
        logger.info("Saved OOD sweep results to %s", sweep_out)

    if run_eval:
        logger.info("Executing Multi-Photo Fusion evaluation on val.csv...")
        evaluate_val_multi_photo(
            cached_logits_path=args.cached_logits,
            karty_path=args.karty_path,
            output_dir=args.output_dir,
            energy_threshold_frame=args.energy_threshold,
            min_top1_prob=args.min_top1_prob,
            disagreement_prob_threshold=args.disagreement_threshold,
            high_stakes_veto_threshold=args.veto_threshold,
            veto_check_all_frames=not args.no_veto_all_frames,
        )


if __name__ == "__main__":
    main()
