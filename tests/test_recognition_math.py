"""Unit tests for the P2 v1 reference math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from grzybobranie_ml.recognition_math import (
    STATUS_OK,
    STATUS_OOD,
    STATUS_UNCLEAR,
    decide_single,
    energy,
    fuse,
    resolve_temperatures,
    softmax,
)

TOL = 1e-5

# A generic 5-class peaked vector.
PEAKED = np.array([4.0, 1.0, 0.2, -0.5, -1.0], dtype=np.float64)
FLAT = np.array([0.1, 0.0, -0.05, 0.05, -0.02], dtype=np.float64)


def test_softmax_sums_to_one_and_is_shift_invariant() -> None:
    p = softmax(PEAKED, 1.3)
    assert p.dtype == np.float64
    assert abs(float(p.sum()) - 1.0) < 1e-15
    assert np.all(p > 0)
    shifted = softmax(PEAKED + 12.5, 1.3)
    assert np.allclose(p, shifted, atol=1e-15)


def test_softmax_recovers_constructed_distribution() -> None:
    target = np.array([0.5, 0.2, 0.15, 0.1, 0.05], dtype=np.float64)
    t = 1.7
    logits = t * np.log(target)
    got = softmax(logits, t)
    assert np.allclose(got, target, atol=1e-15)


def test_energy_matches_stable_logsumexp_definition() -> None:
    t = 1.4
    e = energy(PEAKED, t)
    scaled = PEAKED / t
    peak = float(np.max(scaled))
    expected = float(-t * (peak + math.log(float(np.sum(np.exp(scaled - peak))))))
    assert abs(e - expected) < 1e-15
    # A positive shift of every logit lowers energy by the same amount.
    assert abs(energy(PEAKED + 3.0, t) - (e - 3.0)) < 1e-12


def test_energy_and_softmax_are_stable_for_large_logits() -> None:
    huge = np.array([1500.0, 1490.0, 0.0, -1490.0, -1500.0], dtype=np.float64)
    p = softmax(huge, 1.0)
    e = energy(huge, 1.0)
    assert np.all(np.isfinite(p))
    assert math.isfinite(e)
    assert abs(float(p.sum()) - 1.0) < 1e-15
    # The gap of 10 between the top two at T=1 is enough for a near-one-hot.
    assert p[0] > 0.999


def test_invalid_temperature_and_logits_are_rejected() -> None:
    with pytest.raises(ValueError, match="T"):
        softmax(PEAKED, 0.0)
    with pytest.raises(ValueError, match="T"):
        energy(PEAKED, -1.0)
    with pytest.raises(ValueError, match="at least 2"):
        softmax([1.0], 1.0)
    with pytest.raises(ValueError, match="finite"):
        energy([1.0, float("nan")], 1.0)


def test_resolve_temperatures_rules() -> None:
    t_p, t_e = resolve_temperatures(
        {"calibration": {"temperature": 1.5}, "ood": {"energyTemperature": 0.8}}
    )
    assert (t_p, t_e) == (1.5, 0.8)
    t_p, t_e = resolve_temperatures({"calibration": {"temperature": 1.5}, "ood": {}})
    assert (t_p, t_e) == (1.5, 1.5)
    t_p, t_e = resolve_temperatures({"ood": {"energyTemperature": 0.8}})
    assert (t_p, t_e) == (0.8, 0.8)
    with pytest.raises(ValueError, match="temperature"):
        resolve_temperatures({"ood": {}})


def _manifest(**ood_extra: object) -> dict:
    ood = {
        "energyThreshold": -6.0,
        "minTop1Prob": 0.40,
        **ood_extra,
    }
    return {
        "calibration": {"temperature": 1.0},
        "ood": ood,
        "classes": [f"c{i}" for i in range(5)],
    }


def test_decide_ok_and_top3_candidates() -> None:
    # Force a known distribution and a passing energy.
    target = np.array([0.70, 0.15, 0.08, 0.05, 0.02], dtype=np.float64)
    logits = np.log(target) + 8.0  # energy shift, T_prob = T_energy = 1
    result = decide_single(logits, _manifest())
    assert result["status"] == STATUS_OK
    assert result["energy"] < -6.0
    assert result["p_top1"] == pytest.approx(0.70, abs=1e-12)
    ids = [c["classId"] for c in result["candidates"]]
    assert ids == ["c0", "c1", "c2"]
    assert [c["rank"] for c in result["candidates"]] == [1, 2, 3]
    assert result["candidates"][0]["index"] == 0


def test_decide_energy_gate_and_strict_boundary() -> None:
    target = np.array([0.80, 0.10, 0.05, 0.03, 0.02], dtype=np.float64)
    z0 = np.log(target)
    e0 = energy(z0, 1.0)
    # Put energy exactly on the threshold.
    z_eq = z0 + (e0 - (-6.0))
    assert energy(z_eq, 1.0) == pytest.approx(-6.0, abs=1e-12)
    equal = decide_single(z_eq, _manifest())
    assert equal["status"] == STATUS_OK
    # Just above the threshold (less negative / higher energy) is OOD.
    z_over = z_eq - 0.05
    over = decide_single(z_over, _manifest())
    assert over["status"] == STATUS_OOD
    assert over["candidates"] == []


def test_decide_min_top1_gate_and_strict_boundary() -> None:
    # top-1 == 0.40, energy comfortable.
    target = np.array([0.40, 0.25, 0.15, 0.12, 0.08], dtype=np.float64)
    logits = np.log(target) + 8.0
    result = decide_single(logits, _manifest())
    assert result["p_top1"] == pytest.approx(0.40, abs=1e-12)
    assert result["status"] == STATUS_OK
    # A slightly flatter top-1 is unclear (energy still passes).
    low = np.array([0.39, 0.25, 0.16, 0.12, 0.08], dtype=np.float64)
    low = low / low.sum()
    unclear = decide_single(np.log(low) + 8.0, _manifest())
    assert unclear["status"] == STATUS_UNCLEAR
    assert unclear["candidates"] == []


def test_decide_gate_order_energy_before_top1() -> None:
    # Both gates fail: flat distribution and high energy.
    target = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)
    logits = np.log(target)  # no downward shift → relatively high energy
    assert energy(logits, 1.0) > -6.0
    assert float(np.max(softmax(logits, 1.0))) < 0.40
    result = decide_single(logits, _manifest())
    assert result["status"] == STATUS_OOD


def test_differing_temperatures_change_energy_not_only_softmax() -> None:
    logits = PEAKED
    same = _manifest()
    split = _manifest(energyTemperature=2.5)
    a = decide_single(logits, same)
    b = decide_single(logits, split)
    # Softmax uses calibration.temperature=1.0 in both manifests.
    assert a["p_top1"] == pytest.approx(b["p_top1"], abs=1e-15)
    assert a["energy"] != pytest.approx(b["energy"], abs=1e-8)


def test_fuse_m0_is_ood_without_zbar() -> None:
    # Three high-energy photos.
    target = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)
    photo = np.log(target)
    result = fuse([photo, photo, photo], _manifest())
    assert result["m"] == 0
    assert result["status"] == STATUS_OOD
    assert result["energy"] is None
    assert result["p_top1"] is None
    assert result["candidates"] == []
    assert result["weights"] == []
    assert len(result["per_photo"]) == 3


def test_fuse_m1_uses_single_photo_thresholds() -> None:
    ok = np.log(np.array([0.80, 0.10, 0.05, 0.03, 0.02], dtype=np.float64)) + 8.0
    bad = np.log(np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64))
    manifest = _manifest(fusion={"2": {"energyThreshold": -6.3, "minTop1Prob": 0.45}})
    result = fuse([bad, ok], manifest)
    assert result["m"] == 1
    assert result["status"] == STATUS_OK
    single = decide_single(ok, manifest)
    assert result["energy"] == pytest.approx(single["energy"], abs=1e-15)
    assert result["p_top1"] == pytest.approx(single["p_top1"], abs=1e-15)
    assert abs(result["weights"][0] - 1.0) < 1e-15


def test_fuse_m2_uses_fusion_thresholds_and_entropy_weights() -> None:
    peaked = np.log(np.array([0.90, 0.04, 0.03, 0.02, 0.01], dtype=np.float64)) + 8.0
    milder = np.log(np.array([0.60, 0.20, 0.10, 0.06, 0.04], dtype=np.float64)) + 8.0
    manifest = _manifest(fusion={"2": {"energyThreshold": -6.3, "minTop1Prob": 0.45}})
    result = fuse([peaked, milder], manifest)
    assert result["m"] == 2
    assert result["status"] == STATUS_OK
    # The more peaked photo must receive the larger entropy weight.
    assert result["weights"][0] > result["weights"][1]
    assert abs(sum(result["weights"]) - 1.0) < 1e-12


def test_fuse_equal_weights_when_all_accepted_are_uniform() -> None:
    uniform = np.zeros(5, dtype=np.float64) + 8.0  # softmax is uniform; energy is low
    manifest = _manifest(
        minTop1Prob=0.10, fusion={"2": {"energyThreshold": -6.3, "minTop1Prob": 0.10}}
    )
    # Uniform 5-way top-1 is 0.2, so lower the minTop1 floor to accept them.
    result = fuse([uniform, uniform], manifest)
    assert result["m"] == 2
    assert result["weights"][0] == pytest.approx(0.5, abs=1e-12)
    assert result["weights"][1] == pytest.approx(0.5, abs=1e-12)


def test_fuse_rejects_more_than_max_photos() -> None:
    photo = PEAKED
    with pytest.raises(ValueError, match="at most 3"):
        fuse([photo, photo, photo, photo], _manifest())


def test_fuse_missing_fusion_key_for_m2() -> None:
    ok = np.log(np.array([0.80, 0.10, 0.05, 0.03, 0.02], dtype=np.float64)) + 8.0
    with pytest.raises(ValueError, match=r"ood.fusion\[2\]"):
        fuse([ok, ok], _manifest())
