"""Unit tests for Stage P5: Multi-photo prediction fusion and safety veto rules."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch

from grzybobranie_ml.multi_photo import (
    MultiPhotoResult,
    build_edible_to_toxic_twins,
    fuse_observation_frames,
    run_ood_threshold_sweep,
)


@pytest.fixture
def mock_species_env() -> dict[str, Any]:
    """Fixture with mock species mapping and edibility sets."""
    idx_to_class = {
        0: "macrolepiota_procera",     # Edible
        1: "lepiota_brunneoincarnata", # Deadly / high stakes
        2: "agaricus_campestris",      # Edible
        3: "agaricus_xanthodermus",    # Toxic / high stakes
        4: "amanita_phalloides",       # Deadly / high stakes
        5: "boletus_edulis",           # Edible
    }
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    edible_slugs = {"macrolepiota_procera", "agaricus_campestris", "boletus_edulis"}
    toxic_slugs = {"lepiota_brunneoincarnata", "agaricus_xanthodermus", "amanita_phalloides"}
    high_stakes_slugs = ["lepiota_brunneoincarnata", "agaricus_xanthodermus", "amanita_phalloides"]

    return {
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
        "edible_slugs": edible_slugs,
        "toxic_slugs": toxic_slugs,
        "high_stakes_slugs": high_stakes_slugs,
    }


def test_case1_single_frame_ood_rejection(mock_species_env: dict[str, Any]) -> None:
    """Case 1: Test rejection of an OOD frame prior to fusion."""
    # 3 frames for macrolepiota_procera (idx 0)
    # Frame 0 and 1: strong in-distribution logits (E <= theta)
    # Frame 2: flat near-zero logits (very high energy, E > theta)
    k = 6
    logits = np.zeros((3, k), dtype=np.float64)

    # Frame 0: clean procera
    logits[0, 0] = 5.0
    # Frame 1: clean procera
    logits[1, 0] = 4.5
    # Frame 2: blurry / OOD photo with flat low activations -> Liu energy = -T * log(sum(exp(0))) ~ -T * log(6) ~ -1.40
    # While frame 0 has energy ~ -T * log(exp(5)) ~ -5.0 * T
    logits[2, :] = 0.0

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-2.5,  # Frame 0 & 1 have E ~ -3.8, frame 2 has E ~ -1.43
        min_top1_prob=0.50,
    )

    # Frame 2 must be rejected, frames 0 and 1 accepted
    assert res.accepted_frame_indices == [0, 1]
    assert res.rejected_frame_indices == [2]
    assert res.status == "OK"
    assert res.predicted_slug == "macrolepiota_procera"
    assert res.confident_edible is True


def test_case1_all_frames_ood_rejection(mock_species_env: dict[str, Any]) -> None:
    """Case 1 (b): Test when all frames are outside distribution."""
    k = 6
    # All 2 frames have flat zero logits (OOD)
    logits = np.zeros((2, k), dtype=np.float64)

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        energy_threshold_frame=-2.5,
    )

    assert res.status == "ALL_FRAMES_OOD"
    assert res.accepted_frame_indices == []
    assert res.rejected_frame_indices == [0, 1]
    assert res.confidence == 0.0
    assert res.predicted_slug is None
    assert res.confident_edible is False


def test_case2_disagreement_conflict_detection(mock_species_env: dict[str, Any]) -> None:
    """Case 2: Test disagreement check when 2 frames confidently disagree."""
    k = 6
    logits = np.zeros((2, k), dtype=np.float64)

    # Frame 0: very confident macrolepiota_procera (idx 0), p > 0.80
    logits[0, 0] = 6.0
    # Frame 1: very confident agaricus_campestris (idx 2), p > 0.80
    logits[1, 2] = 6.0

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        energy_threshold_frame=-1.0,
        disagreement_prob_threshold=0.70,
    )

    assert res.status == "CONFLICT_DETECTED"
    assert "macrolepiota_procera" in res.conflict_species
    assert "agaricus_campestris" in res.conflict_species
    assert res.predicted_slug is None
    assert res.confident_edible is False


def test_case2_no_false_conflict_when_one_frame_uncertain(mock_species_env: dict[str, Any]) -> None:
    """Case 2 (b): No conflict if one frame is confident and another is uncertain."""
    k = 6
    logits = np.zeros((2, k), dtype=np.float64)

    # Frame 0: confident macrolepiota_procera (idx 0), p ~ 0.85
    logits[0, 0] = 5.0
    # Frame 1: uncertain, small difference between idx 2 and idx 0 -> p < 0.50
    logits[1, 2] = 1.0
    logits[1, 0] = 0.8

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-0.5,
        disagreement_prob_threshold=0.70,
    )

    # Disagreement threshold is 0.70, Frame 1 has p < 0.70, so no conflict detected
    assert res.status == "OK"
    assert res.predicted_slug == "macrolepiota_procera"


def test_case3_valid_multi_photo_fusion(mock_species_env: dict[str, Any]) -> None:
    """Case 3: Test valid fusion where multiple noisy frames reinforce the true class."""
    k = 6
    logits = np.zeros((3, k), dtype=np.float64)

    # 3 frames pointing to boletus_edulis (idx 5) with moderate confidence
    # Frame 0: boletus=3.0, noise on idx 0=2.0
    logits[0, 5] = 3.0
    logits[0, 0] = 2.0
    # Frame 1: boletus=3.2, noise on idx 2=2.1
    logits[1, 5] = 3.2
    logits[1, 2] = 2.1
    # Frame 2: boletus=2.9, noise on idx 3=1.8
    logits[2, 5] = 2.9
    logits[2, 3] = 1.8

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.55,
    )

    assert res.status == "OK"
    assert res.predicted_slug == "boletus_edulis"
    assert res.predicted_idx == 5
    assert res.is_edible is True
    assert res.confident_edible is True
    assert res.confidence > 0.80  # Reinforced by averaging
    assert len(res.accepted_frame_indices) == 3


def test_case4_high_stakes_veto(mock_species_env: dict[str, Any]) -> None:
    """Case 4: Test High-Stakes Veto when fused is edible but one frame detected toxic risk."""
    k = 6
    logits = np.zeros((2, k), dtype=np.float64)

    # Frame 0: Macrolepiota procera (edible, idx 0) = 4.5
    logits[0, 0] = 4.5
    # Frame 1: Fused will still favor procera (idx 0 = 3.5), BUT Lepiota brunneoincarnata (deadly, idx 1)
    # reaches p_toxic >= 0.20
    logits[1, 0] = 3.0
    logits[1, 1] = 2.5  # With T=0.8, exp(2.5/0.8)/exp(3.0/0.8) ~ exp(-0.625) ~ 0.53 -> p_toxic ~ 0.30

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
        high_stakes_veto_threshold=0.20,
    )

    # VETO must trigger: status is HIGH_STAKES_VETO and confident_edible is BLOCKED (False)
    assert res.status == "HIGH_STAKES_VETO"
    assert res.predicted_slug == "macrolepiota_procera"
    assert res.is_edible is True
    assert res.confident_edible is False
    assert len(res.veto_risk_species) >= 1
    assert res.veto_risk_species[0]["slug"] == "lepiota_brunneoincarnata"
    assert res.veto_risk_species[0]["probability"] >= 0.20
    assert res.veto_risk_species[0]["frame_idx"] == 1


def test_case4_negative_veto_when_fused_already_toxic(mock_species_env: dict[str, Any]) -> None:
    """Case 4 (b): No veto needed if fused prediction is already classified as toxic/deadly."""
    k = 6
    logits = np.zeros((2, k), dtype=np.float64)

    # Both frames classify Amanita phalloides (idx 4)
    logits[0, 4] = 4.0
    logits[1, 4] = 4.5

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
    )

    assert res.status == "OK"
    assert res.predicted_slug == "amanita_phalloides"
    assert res.is_edible is False
    assert res.confident_edible is False


def test_single_frame_observation(mock_species_env: dict[str, Any]) -> None:
    """Test that pipeline works seamlessly on 1-frame observations."""
    k = 6
    logits = np.zeros(k, dtype=np.float64)
    logits[5] = 4.0  # Boletus edulis

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
    )

    assert res.status == "OK"
    assert res.predicted_slug == "boletus_edulis"
    assert res.accepted_frame_indices == [0]
    assert res.total_frames == 1


def test_result_serialization(mock_species_env: dict[str, Any]) -> None:
    """Test serialization of MultiPhotoResult to JSON dictionary."""
    k = 6
    logits = np.zeros((2, k), dtype=np.float64)
    logits[:, 0] = 3.0

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-0.5,
    )

    d = res.to_dict()
    assert isinstance(d, dict)
    json_str = json.dumps(d)
    assert "macrolepiota_procera" in json_str


def test_ood_sweep_logic(mock_species_env: dict[str, Any]) -> None:
    """Test 2D OOD threshold sweep on synthetic calib partitions."""
    rng = np.random.default_rng(42)
    n_val = 100
    n_ood = 50
    k = 6

    val_logits = rng.standard_normal((n_val, k))
    val_targets = rng.integers(0, k, size=n_val)

    ood_f_logits = rng.standard_normal((n_ood, k))
    ood_f_records = [{"proposed_split": "calib"} for _ in range(n_ood)]

    ood_nf_logits = rng.standard_normal((n_ood, k))
    ood_nf_records = [{"proposed_split": "calib"} for _ in range(n_ood)]

    sweep = run_ood_threshold_sweep(
        val_logits=val_logits,
        val_targets=val_targets,
        ood_fungi_logits=ood_f_logits,
        ood_fungi_records=ood_f_records,
        ood_nonfungi_logits=ood_nf_logits,
        ood_nonfungi_records=ood_nf_records,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        toxic_slugs=mock_species_env["toxic_slugs"],
        temperature=0.8,
        target_id_tprs=[0.95, 0.90, 0.85],
        min_top1_probs=[0.40, 0.50, 0.60],
    )

    assert "baseline" in sweep
    assert "recommended" in sweep
    assert len(sweep["grid"]) == 9


def test_torch_tensor_support(mock_species_env: dict[str, Any]) -> None:
    """Test that fuse_observation_frames seamlessly accepts PyTorch tensors without TypeError."""
    k = 6
    logits_tensor = torch.zeros(2, k, dtype=torch.float32)
    logits_tensor[0, 5] = 4.0
    logits_tensor[1, 5] = 4.5

    res = fuse_observation_frames(
        logits=logits_tensor,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
    )

    assert res.status == "OK"
    assert res.predicted_slug == "boletus_edulis"
    assert res.is_ood is False


def test_empty_logits_raises_value_error(mock_species_env: dict[str, Any]) -> None:
    """Test that passing 0 frames raises ValueError instead of RuntimeWarning and NaN."""
    k = 6
    empty_logits = np.empty((0, k), dtype=np.float64)

    with pytest.raises(ValueError, match="Cannot fuse empty logits"):
        fuse_observation_frames(
            logits=empty_logits,
            idx_to_class=mock_species_env["idx_to_class"],
            class_to_idx=mock_species_env["class_to_idx"],
            edible_slugs=mock_species_env["edible_slugs"],
            high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        )


def test_invalid_logits_and_temperatures(mock_species_env: dict[str, Any]) -> None:
    """Test validation of NaN/Inf logits and negative temperatures."""
    k = 6
    nan_logits = np.zeros((2, k), dtype=np.float64)
    nan_logits[0, 0] = np.nan

    with pytest.raises(ValueError, match="contain NaN or Inf"):
        fuse_observation_frames(
            logits=nan_logits,
            idx_to_class=mock_species_env["idx_to_class"],
            class_to_idx=mock_species_env["class_to_idx"],
            edible_slugs=mock_species_env["edible_slugs"],
            high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        )

    valid_logits = np.zeros((2, k), dtype=np.float64)
    with pytest.raises(ValueError, match="Temperatures must be strictly positive"):
        fuse_observation_frames(
            logits=valid_logits,
            idx_to_class=mock_species_env["idx_to_class"],
            class_to_idx=mock_species_env["class_to_idx"],
            edible_slugs=mock_species_env["edible_slugs"],
            high_stakes_slugs=mock_species_env["high_stakes_slugs"],
            temperature_frame=-1.0,
        )


def test_veto_triggered_from_rejected_frame(mock_species_env: dict[str, Any]) -> None:
    """Test that safety veto triggers even when toxic traits appear on an OOD-rejected frame."""
    k = 6
    logits = np.zeros((2, k), dtype=np.float64)

    # Frame 0: clean accepted photo indicating edible macrolepiota_procera (idx 0)
    logits[0, 0] = 5.0
    # Frame 1: blurry / high energy photo, rejected by per-frame OOD (E > theta)
    # BUT model detects deadly Amanita phalloides (idx 4) with p_toxic >= 0.20
    logits[1, 4] = 1.6  # E ~ -2.01 > -2.5 (rejected), but p_toxic ~ 0.60 >= 0.20
    logits[1, 0] = 0.0

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-2.5,  # Frame 0 has E ~ -3.8 (accepted), Frame 1 has E ~ -2.1 (rejected)
        min_top1_prob=0.50,
        high_stakes_veto_threshold=0.20,
        veto_check_all_frames=True,
    )

    # Frame 1 is rejected from logit averaging
    assert res.accepted_frame_indices == [0]
    assert res.rejected_frame_indices == [1]
    # BUT veto MUST trigger to fail closed on toxic lookalike
    assert res.status == "HIGH_STAKES_VETO"
    assert res.is_edible is True
    assert res.confident_edible is False
    assert any(risk["slug"] == "amanita_phalloides" for risk in res.veto_risk_species)


def test_unclear_and_fused_ood_statuses(mock_species_env: dict[str, Any]) -> None:
    """Test gating into UNCLEAR (low prob) and OOD (high fused energy)."""
    k = 6
    # 1. Low confidence -> UNCLEAR
    logits_unclear = np.zeros((2, k), dtype=np.float64)
    logits_unclear[0, 0] = 1.0
    logits_unclear[1, 0] = 1.0

    res_unclear = fuse_observation_frames(
        logits=logits_unclear,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=10.0,
        energy_threshold_fused=10.0,
        min_top1_prob=0.80,  # fused confidence ~ 0.53 < 0.80
    )
    assert res_unclear.status == "UNCLEAR"
    assert res_unclear.confident_edible is False
    assert res_unclear.is_ood is False

    # 2. High fused energy -> OOD
    logits_ood = np.zeros((2, k), dtype=np.float64)
    logits_ood[0, 0] = 3.0
    logits_ood[1, 0] = 3.0

    res_ood = fuse_observation_frames(
        logits=logits_ood,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=10.0,
        energy_threshold_fused=-5.0,  # Fused energy is ~ -2.4 > -5.0 -> OOD
        min_top1_prob=0.50,
    )
    assert res_ood.status == "OOD"
    assert res_ood.confident_edible is False
    assert res_ood.is_ood is True


def test_five_frame_observation(mock_species_env: dict[str, Any]) -> None:
    """Test observation with M=5 frames."""
    k = 6
    logits = np.zeros((5, k), dtype=np.float64)
    logits[:, 2] = 4.0  # agaricus_campestris

    res = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.55,
    )

    assert res.status == "OK"
    assert res.predicted_slug == "agaricus_campestris"
    assert res.total_frames == 5
    assert len(res.accepted_frame_indices) == 5
    assert res.is_ood is False
    assert res.confident_edible is True


def test_is_ood_property_coverage(mock_species_env: dict[str, Any]) -> None:
    """Test is_ood helper property on MultiPhotoResult across statuses."""
    r_ood = MultiPhotoResult(
        status="OOD",
        predicted_slug="boletus_edulis",
        predicted_idx=5,
        confidence=0.8,
        energy=-1.0,
        is_edible=True,
        confident_edible=False,
        accepted_frame_indices=[0],
        rejected_frame_indices=[],
        total_frames=1,
        frame_predictions=[],
    )
    assert r_ood.is_ood is True
    assert r_ood.to_dict()["is_ood"] is True

    r_all_ood = MultiPhotoResult(
        status="ALL_FRAMES_OOD",
        predicted_slug=None,
        predicted_idx=None,
        confidence=0.0,
        energy=-0.5,
        is_edible=False,
        confident_edible=False,
        accepted_frame_indices=[],
        rejected_frame_indices=[0],
        total_frames=1,
        frame_predictions=[],
    )
    assert r_all_ood.is_ood is True
    assert r_all_ood.to_dict()["is_ood"] is True

    r_ok = MultiPhotoResult(
        status="OK",
        predicted_slug="boletus_edulis",
        predicted_idx=5,
        confidence=0.9,
        energy=-3.0,
        is_edible=True,
        confident_edible=True,
        accepted_frame_indices=[0],
        rejected_frame_indices=[],
        total_frames=1,
        frame_predictions=[],
    )
    assert r_ok.is_ood is False
    assert r_ok.to_dict()["is_ood"] is False


def test_asymmetric_twin_pair_threshold_veto(mock_species_env: dict[str, Any]) -> None:
    """Direct toxic twin triggers HIGH_STAKES_VETO at p_t in [0.10, 0.20), while non-twin HS does not."""
    pairs = [("macrolepiota_procera", "lepiota_brunneoincarnata")]
    twins = build_edible_to_toxic_twins(
        pairs, mock_species_env["edible_slugs"], set(mock_species_env["high_stakes_slugs"])
    )
    assert twins == {"macrolepiota_procera": {"lepiota_brunneoincarnata"}}

    # Single frame: macrolepiota_procera (idx 0) ~0.85, lepiota_brunneoincarnata (idx 1) ~0.14 in [0.10, 0.20)
    # Note: p_edible / p_twin ~ 6.07 >= 5.0, so ONLY the lowered 0.10 twin threshold catches it!
    p_target = np.array([0.85, 0.14, 0.0025, 0.0025, 0.0025, 0.0025], dtype=np.float64)
    logits = 0.8 * np.log(p_target) + 6.0

    # Without dangerous_pairs, 0.14 < 0.20 would pass as OK
    res_legacy = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
        high_stakes_veto_threshold=0.20,
    )
    assert res_legacy.status == "OK"

    # With dangerous_pairs, direct twin at 0.14 >= 0.10 must trigger HIGH_STAKES_VETO
    res_asym = fuse_observation_frames(
        logits=logits,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
        high_stakes_veto_threshold=0.20,
        dangerous_pairs=pairs,
        twin_pair_veto_threshold=0.10,
        twin_pair_min_ratio=5.0,
    )
    assert res_asym.status == "HIGH_STAKES_VETO"
    assert res_asym.confident_edible is False
    assert any(
        r["slug"] == "lepiota_brunneoincarnata" and r["reason"] == "twin_pair_threshold"
        for r in res_asym.veto_risk_species
    )

    # Non-twin high-stakes species (amanita_phalloides idx 4) at 0.14 should NOT trigger twin veto on boletus_edulis
    p_boletus = np.array([0.0025, 0.0025, 0.0025, 0.0025, 0.14, 0.85], dtype=np.float64)
    logits_boletus = 0.8 * np.log(p_boletus) + 6.0
    res_non_twin = fuse_observation_frames(
        logits=logits_boletus,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
        high_stakes_veto_threshold=0.20,
        dangerous_pairs=pairs,
        twin_pair_veto_threshold=0.10,
        twin_pair_min_ratio=5.0,
    )
    assert res_non_twin.status == "OK"
    assert res_non_twin.confident_edible is True


def test_asymmetric_twin_pair_ratio_veto_fused_and_frame(mock_species_env: dict[str, Any]) -> None:
    """Block confident edible when p_edible < 5 * p_toxic_pair even when p_toxic_pair < 0.10."""
    pairs = [("agaricus_campestris", "agaricus_xanthodermus")]

    # Case A: Single/fused prob for agaricus_campestris (idx 2) = 0.42, agaricus_xanthodermus (idx 3) = 0.09
    # p_t = 0.09 < 0.10, BUT 0.42 < 5 * 0.09 = 0.45 -> ratio is 4.67 < 5.0 -> HIGH_STAKES_VETO!
    p_fused_viol = np.array([0.20, 0.10, 0.42, 0.09, 0.10, 0.09], dtype=np.float64)
    logits_fused_viol = 0.8 * np.log(p_fused_viol) + 6.0
    res_fused = fuse_observation_frames(
        logits=logits_fused_viol,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.40,
        dangerous_pairs=pairs,
        twin_pair_veto_threshold=0.10,
        twin_pair_min_ratio=5.0,
    )
    assert res_fused.status == "HIGH_STAKES_VETO"
    assert res_fused.confident_edible is False
    assert any(r["reason"] == "twin_pair_ratio_fused" for r in res_fused.veto_risk_species)

    # Case B: 2 accepted frames — frame 0 is strong campestris (0.95), frame 1 is ambiguous (0.25 vs 0.08 twin).
    # Fused passes 5x dominance, and p_twin < 0.10 everywhere, BUT frame 1 violates 0.25 < 5 * 0.08!
    p_f0 = np.array([0.01, 0.01, 0.95, 0.01, 0.01, 0.01], dtype=np.float64)
    p_f1 = np.array([0.30, 0.10, 0.25, 0.08, 0.15, 0.12], dtype=np.float64)
    logits_2 = np.stack([0.8 * np.log(p_f0) + 6.0, 0.8 * np.log(p_f1) + 6.0], axis=0)
    res_frame = fuse_observation_frames(
        logits=logits_2,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
        dangerous_pairs=pairs,
        twin_pair_veto_threshold=0.10,
        twin_pair_min_ratio=5.0,
    )
    assert res_frame.status == "HIGH_STAKES_VETO"
    assert any(r["reason"] == "twin_pair_ratio_frame" and r["frame_idx"] == 1 for r in res_frame.veto_risk_species)

    # Case C: Positive control — p_twin = 0.08 < 0.10 and p_edible = 0.70 >= 5 * 0.08 -> OK!
    p_ok = np.array([0.10, 0.05, 0.70, 0.08, 0.04, 0.03], dtype=np.float64)
    res_ok = fuse_observation_frames(
        logits=0.8 * np.log(p_ok) + 6.0,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.5,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
        dangerous_pairs=pairs,
        twin_pair_veto_threshold=0.10,
        twin_pair_min_ratio=5.0,
    )
    assert res_ok.status == "OK"
    assert res_ok.confident_edible is True

    # Case D: Per-frame p_twin < 0.10 and ratio >= 5.0 on every frame, but after logit averaging
    # non-twin distractors cancel out so fused p_twin >= 0.10 (even if ratio >= 5.0) -> HIGH_STAKES_VETO!
    p_d0 = np.array([0.35, 0.005, 0.55, 0.085, 0.005, 0.005], dtype=np.float64)
    p_d1 = np.array([0.005, 0.35, 0.55, 0.085, 0.005, 0.005], dtype=np.float64)
    logits_d = np.stack([0.8 * np.log(p_d0) + 6.0, 0.8 * np.log(p_d1) + 6.0], axis=0)
    res_fused_thresh = fuse_observation_frames(
        logits=logits_d,
        idx_to_class=mock_species_env["idx_to_class"],
        class_to_idx=mock_species_env["class_to_idx"],
        edible_slugs=mock_species_env["edible_slugs"],
        high_stakes_slugs=mock_species_env["high_stakes_slugs"],
        temperature_frame=0.8,
        temperature_multi=0.9,
        energy_threshold_frame=-1.0,
        min_top1_prob=0.50,
        dangerous_pairs=pairs,
        twin_pair_veto_threshold=0.10,
        twin_pair_min_ratio=5.0,
    )
    assert res_fused_thresh.status == "HIGH_STAKES_VETO"
    assert res_fused_thresh.confident_edible is False
    assert any(r["reason"] == "twin_pair_threshold_fused" for r in res_fused_thresh.veto_risk_species)

